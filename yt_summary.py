import re
import heapq
from collections import Counter
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline

def get_video_id(url):
    pattern = r"(?:v=|youtu\.be/|/embed/|/v/|/watch\?v=)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube URL or video ID not found.")

def fetch_transcript(video_id, skip_silences=True, min_words=2):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        print("Available transcripts:", [t.language_code for t in transcript_list])
        transcript = transcript_list.find_transcript(['en'])
        data = transcript.fetch()

        lines = []
        for entry in data:
            if isinstance(entry, dict):
                text = entry.get('text', '').strip()
                duration = entry.get('duration', 0)
            else:
                text = getattr(entry, 'text', '').strip()
                duration = getattr(entry, 'duration', 0)

            if skip_silences and (not text or len(text.split()) < min_words or duration > 7):
                continue

            lines.append(text)

        return lines

    except TranscriptsDisabled:
        raise Exception("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise Exception("No transcript found in the requested language.")
    except Exception as e:
        raise Exception(f"Error fetching transcript: {str(e)}")

def save_transcript(lines, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.strip() + "\n")
    print(f"\n✅ Full transcript saved to: {filename}")

def summarize_text(lines):
    """Abstractive summarization using transformers (e.g., BART)."""
    text = " ".join(lines)

    if len(text.split()) < 50:
        return text

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    max_chunk_size = 900
    words = text.split()
    chunks = [" ".join(words[i:i+max_chunk_size]) for i in range(0, len(words), max_chunk_size)]

    final_summary = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        final_summary.append(summary[0]['summary_text'])

    return "\n".join(final_summary)

def ask_question(context, question):
    """Simple Q&A from context using transformers-based model."""
    qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    result = qa(question=question, context=context)
    return result['answer']

def main():
    video_url = input("Enter the YouTube video URL: ").strip()

    try:
        video_id = get_video_id(video_url)
        print(f"Extracted Video ID: {video_id}")

        transcript_lines = fetch_transcript(video_id)
        print(f"Fetched {len(transcript_lines)} usable transcript lines.")

        save_transcript(transcript_lines)
        summary = summarize_text(transcript_lines)
        print("\n📝 Summary of the Video Transcript:\n")
        print(summary)

        # === Interactive Q&A Loop ===
        context = " ".join(transcript_lines)
        print("\n💬 Ask questions about the video. Type 'exit' to quit.\n")
        while True:
            q = input("Q: ")
            if q.lower() in ['exit', 'quit']:
                break
            try:
                answer = ask_question(context, q)
                print("A:", answer)
            except Exception as e:
                print("Sorry, couldn't answer that. Error:", str(e))

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
