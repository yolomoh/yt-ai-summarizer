import os
import re
import asyncio
import logging

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, FSInputFile
from aiogram.enums import ParseMode
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.utils.token import TokenValidationError

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

import openai
from transformers import pipeline

# === Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# === API Keys ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not BOT_TOKEN or not OPENAI_API_KEY:
    raise ValueError("Missing BOT_TOKEN or OPENAI_API_KEY in environment variables")

openai.api_key = OPENAI_API_KEY

# === Load Fallback Transformers ===
logger.info("🔁 Loading local fallback models…")
summarizer_fallback = pipeline("summarization", model="facebook/bart-large-cnn")
qa_fallback = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
logger.info("✅ Local models ready.")

# === Telegram Setup ===
router = Router()
chat_context = {}

# === Markdown Escaper ===
def escape_markdown(text: str) -> str:
    escape_chars = r"_*[]()~`>#+-=|{}.!\\"
    return "".join(f"\\{c}" if c in escape_chars else c for c in text)

# === Helper Functions ===

def get_video_id(url: str) -> str:
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|\&|$)", url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

def fetch_transcript(video_id: str, skip_silences=True, min_words=2) -> list[str]:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    lines = [
        e['text'].strip()
        for e in transcript
        if e['text'].strip() and (not skip_silences or len(e['text'].split()) >= min_words and e['duration'] <= 7)
    ]
    return lines

async def summarize_openai(text: str) -> str:
    try:
        response = await asyncio.wait_for(
            openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Summarize the following transcript."},
                    {"role": "user", "content": text}
                ],
                temperature=0.7,
                max_tokens=512
            ),
            timeout=30
        )
        return response.choices[0].message.content.strip()
    except asyncio.TimeoutError:
        logger.warning("OpenAI summary timed out")
        raise openai.OpenAIError("Summary request timed out")

async def answer_openai(context: str, question: str) -> str:
    try:
        response = await asyncio.wait_for(
            openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Answer based solely on the user-provided transcript."},
                    {"role": "assistant", "content": context},
                    {"role": "user", "content": question}
                ],
                temperature=0
            ),
            timeout=30
        )
        return response.choices[0].message.content.strip()
    except asyncio.TimeoutError:
        logger.warning("OpenAI Q&A timed out")
        raise openai.OpenAIError("Q&A request timed out")

def summarize_fallback(lines: list[str]) -> str:
    text = " ".join(lines)
    if len(text.split()) < 50:
        return text
    chunks = [" ".join(text.split()[i:i+900]) for i in range(0, len(text.split()), 900)]
    res = []
    for c in chunks:
        out = summarizer_fallback(c, max_length=130, min_length=30, do_sample=False)
        res.append(out[0]['summary_text'])
    return "\n".join(res)

def answer_fallback(context: str, question: str) -> str:
    res = qa_fallback(question=question, context=context)
    return res["answer"]

# === Telegram Handlers ===

@router.message(F.text == "/start")
async def cmd_start(msg: Message):
    logger.info(f"[{msg.chat.id}] /start received")
    await msg.answer("Hi! Send a YouTube link to summarize and interact with.\n\nUse /t <link> to get full transcript as a .txt file.")

@router.message(F.text.func(lambda t: 'youtube.com' in t or 'youtu.be' in t))
async def handle_video(msg: Message):
    logger.info(f"[{msg.chat.id}] YouTube link: {msg.text}")
    await msg.answer("⏳ Fetching and summarizing…")
    try:
        vid = get_video_id(msg.text)
        lines = fetch_transcript(vid)
        logger.info(f"Fetched {len(lines)} lines")
        context_text = " ".join(lines[:3000])  # avoid token explosion

        transcript_text = "\n".join(lines)
        filename = f"transcript_{vid}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        try:
            summary = await summarize_openai(context_text)
        except openai.OpenAIError as e:
            logger.warning(f"OpenAI summary failed: {e}. Using fallback")
            summary = summarize_fallback(lines)

        chat_context[msg.chat.id] = context_text

        await msg.answer(f"📄 Summary:\n{escape_markdown(summary)}", parse_mode=ParseMode.MARKDOWN_V2)

        file_to_send = FSInputFile(filename)
        await msg.answer_document(file_to_send, caption="📄 Full transcript as text file")

        await msg.answer("💬 Now ask me questions about the video!")

    except (TranscriptsDisabled, NoTranscriptFound):
        await msg.answer("❌ This video has no transcript available.")
    except Exception as e:
        logger.error(f"Video handling error: {e}")
        await msg.answer(f"❌ Error: {escape_markdown(str(e))}", parse_mode=ParseMode.MARKDOWN_V2)

@router.message(F.text.func(lambda t: t.startswith("/t ")))
async def cmd_transcript(msg: Message):
    logger.info(f"[{msg.chat.id}] /t command received: {msg.text}")
    url = msg.text[3:].strip()
    if not url:
        return await msg.answer("❌ Please provide a YouTube link after /t")

    await msg.answer("⏳ Fetching transcript…")
    try:
        vid = get_video_id(url)
        lines = fetch_transcript(vid)
        logger.info(f"Transcript fetched with {len(lines)} lines")

        transcript_text = "\n".join(lines)
        filename = f"transcript_{vid}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        file_to_send = FSInputFile(filename)
        await msg.answer_document(file_to_send, caption="📄 Full transcript as text file")

    except (TranscriptsDisabled, NoTranscriptFound):
        await msg.answer("❌ This video has no transcript available.")
    except Exception as e:
        logger.error(f"/t command error: {e}")
        await msg.answer(f"❌ Error: {escape_markdown(str(e))}", parse_mode=ParseMode.MARKDOWN_V2)

@router.message()
async def handle_question(msg: Message):
    ctx = chat_context.get(msg.chat.id)
    if not ctx:
        return await msg.answer("Send a YouTube link first!")
    await msg.chat.do("typing")
    try:
        question = msg.text.strip()
        logger.info(f"[{msg.chat.id}] Q: {question}")
        try:
            ans = await answer_openai(ctx, question)
        except openai.OpenAIError as e:
            logger.warning(f"OpenAI Q&A failed: {e}. Using fallback")
            ans = answer_fallback(ctx, question)
        logger.info(f"A: {ans}")
        await msg.answer(f"*A:* {escape_markdown(ans)}", parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        logger.error(f"Q&A error: {e}")
        await msg.answer(f"❌ Couldn't answer: {escape_markdown(str(e))}", parse_mode=ParseMode.MARKDOWN_V2)

# === Async Runner ===

async def main():
    session = AiohttpSession()
    bot = Bot(token=BOT_TOKEN, session=session)
    dp = Dispatcher()
    dp.include_router(router)
    logger.info("🚀 Bot started")
    try:
        await dp.start_polling(bot)
    finally:
        await session.close()
        logger.info("🔚 Bot stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except TokenValidationError:
        logger.error("Invalid BOT_TOKEN")
    except Exception:
        logger.exception("Unexpected error")
