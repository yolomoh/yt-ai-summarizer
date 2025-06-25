import os
import re
import asyncio
import logging
from collections import Counter

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, FSInputFile
from aiogram.enums import ParseMode
from aiogram.client.session.aiohttp import AiohttpSession
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

# === Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("Missing BOT_TOKEN in environment variables")

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

# === Lightweight Fallbacks ===

def simple_summary(lines: list[str], max_lines=5) -> str:
    long_lines = [l for l in lines if len(l.split()) > 5]
    most_common = Counter(long_lines).most_common(max_lines)
    return "\n".join([line for line, _ in most_common]) or "Summary not available."

def simple_answer(context: str, question: str) -> str:
    sentences = context.split(". ")
    keywords = question.lower().split()
    matches = [s for s in sentences if any(k in s.lower() for k in keywords)]
    return ". ".join(matches[:3]) or "No answer found in the transcript."

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
        context_text = " ".join(lines[:3000])

        transcript_text = "\n".join(lines)
        filename = f"transcript_{vid}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        summary = simple_summary(lines)
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
        ans = simple_answer(ctx, question)
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
    except Exception:
        logger.exception("Unexpected error")
