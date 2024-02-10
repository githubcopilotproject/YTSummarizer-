#import streamlit as st
from dotenv import load_dotenv
load_dotenv() ##load all the evironment variables
import os
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube

import assemblyai as aai
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
#aai.settings.api_key = "ASSEMBLYAI_API_KEY"
#aai.settings.api_key = st.secrets["ASSEMBLYAI_API_KEY"]

import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#genai.configure(api_key="GOOGLE_API_KEY")
#genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

prompt="""You are Yotube video summarizer. You will be taking the transcript text
and summarize the entire video and provide the important summary.
Please provide the summary of the text given here:  """

generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 8192,
  "candidate_count": 1,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  }
]

# Getting the transcript data from yt videos
def extract_transcript_details(youtube_video_url):
	try:
		video_id = youtube_video_url.split("v=")[1].split("&")[0]
		transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
		for transcript1 in transcript_list:
			transcript = transcript1.translate('en').fetch()
			transcriptnew = ""
			for i in transcript:
				transcriptnew += " " + i["text"]
		return transcriptnew
	except Exception as e:
		print ("---------------- FOUND Error: ----------------\n" + str(e))
		return ""

def download_youtube_audio(url, transcript_file):
	yt = YouTube(url)
	audio_stream = yt.streams.filter(only_audio=True).first()
	audio_stream.download(output_path=".", filename=transcript_file)


def convert_audio_to_text(transcript_file):
	try:
		transcriber = aai.Transcriber()
		transcript = transcriber.transcribe(transcript_file)
		return transcript.text
	except Exception as e:
		print ("---------------- FOUND Error While convert_audio_to_text: ----------------\n" + str(e))
		return ""	

def generate_gemini_content(transcript_text, prompt):
	try:
		model = genai.GenerativeModel(model_name = "gemini-pro", generation_config=generation_config, safety_settings = safety_settings)
		response = model.generate_content(prompt + transcript_text)
		return response.text
	except Exception as e:
		print ("---------------- FOUND Error: ----------------\n" + str(e))
		print ("---------------- response.prompt_feedback: ----------------\n" + str(response.prompt_feedback))
		return ""

st.title("YouTube Video Detailed Notes")
youtube_link = st.text_input("Enter YouTube Video Link:")

if st.button("Get Notes"):
	video_id = youtube_link.split("=")[1]
	st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", width=100)
	transcript_text = extract_transcript_details(youtube_link)
	if transcript_text:
		summary = generate_gemini_content(transcript_text, prompt)
		summary.replace("$", "\$")
		if summary:
			st.markdown("## Detailed Notes:")
			st.write(summary)
		else:
			st.write("## LLM couldn't generate transcript summary")
	else:
		transcript_file = "yt_transcript_" + video_id + ".txt"
		download_youtube_audio(youtube_link, transcript_file)
		transcript_text = convert_audio_to_text(transcript_file)
		os.remove(transcript_file)
		if transcript_text:
			summary = generate_gemini_content(transcript_text, prompt)
			summary.replace("$", "\$")
			if summary:
				st.markdown("## Detailed Notes:")
				st.write(summary)
			else:
				st.write("## LLM couldn't generate transcript summary")
		else:
			st.write("## No Transcript found for this video")
