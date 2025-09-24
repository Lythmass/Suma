from openai import OpenAI
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import gradio
import os
import re
import math
from datetime import timedelta

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")

openai = OpenAI()
gemini = OpenAI(
    api_key=google_api_key, 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
ytt_api = YouTubeTranscriptApi()

system_message = "You are an assistant specialized in summarizing a YouTube video transcript. \n" \
"You're responsible for providing a summary of the video that emphasizes the structure, keypoints, tone and returns a very detailed result in markdown."

class Video:
    def __init__(self, title, url):
        self.url = url 
        self.title = title
        pattern = (
            r'(?:https?://)?'
            r'(?:[0-9A-Z-]+\.)?'
            r'(?:youtube(?:-nocookie)?\.com|youtu\.be)'
            r'(?:/(?:watch\?v=|watch\?.+&v=|embed/|v/|shorts/|.+\?v=))?'
            r'([A-Za-z0-9_-]{11})'
        )
        match = re.search(pattern, url, re.IGNORECASE)
        self.video_id = match.group(1) if match else None
    def get_transcript(self):
        snippets = ytt_api.fetch(self.video_id).snippets
        transcript = ""
        for snippet in snippets:
            transcript = transcript + f"[{self.format_timestamp(snippet.start)}] " + snippet.text + "\n"
        return transcript
    def format_timestamp(self, seconds: float) -> str:
        td = timedelta(seconds=math.floor(seconds))
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def summarize(title, url, model="GPT"):
    video = Video(title, url)
    prompt = f"Summarize the video: ${video.title}, this is the transcript: ${video.get_transcript()}"
    if model == "GPT":
        stream = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
    if model == "Gemini":
        stream = gemini.chat.completions.create(
            model="gemini-2.5-pro",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        yield response

app = gradio.Interface(
    fn=summarize,
    inputs=[
        gradio.Textbox(label="Video Title: "),
        gradio.Textbox(label="Video URL: "),
        gradio.Dropdown(["GPT", "Gemini"], label="Select model"),
    ],
    outputs=[gradio.Markdown(label="Transcript:")],
    flagging_mode="never"
)

app.launch()