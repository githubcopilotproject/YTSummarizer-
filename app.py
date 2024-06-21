import streamlit as st
import os
#from dotenv import load_dotenv
#load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
from langchain_google_genai import (ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory)

from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube

import assemblyai as aai
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
#aai.settings.api_key = st.secrets["ASSEMBLYAI_API_KEY"]

import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Initialize "Get Note" button clicked state [Else UI will not work]
if 'clicked' not in st.session_state:
	st.session_state.clicked = False

# Initialize session state to store output
if 'summary' not in st.session_state:
    st.session_state.summary = ""

# Initialize chat history
if "messages" not in st.session_state:
	st.session_state.messages = []


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


safety_settings1 = {
  HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
  HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
  HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
  HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
}



def click_button():
	print("Inside click_button() so that means button is clicked")
	st.session_state.clicked = True
	with st.spinner():
		transcript_text = process_yt_transcript(youtube_video_url)
		print("Inside click_button() FINISHED getting summerized transcript")
		print("------------------ Inside click_button() START ------------------\n")
		print (transcript_text)
		print("------------------ Inside click_button() END ------------------\n")
		return transcript_text


def process_yt_transcript(youtube_video_url):
	print("Inside process_yt_transcript()")
	video_id = youtube_video_url.split("=")[1]
	st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", width=100)
	
	transcript_text = extract_transcript_from_youtube(youtube_video_url)
	print("------------------ Inside process_yt_transcript() START ------------------\n")
	print (transcript_text)
	print("------------------ Inside process_yt_transcript() END ------------------\n")
	#Trying to get transcript manually by extracting audio and using speech to text
	if not transcript_text:
		extract_transcript_manually(youtube_video_url, video_id)

	if transcript_text:
		summarize_transcript_text(transcript_text, prompt)
		embed_transcript_text(transcript_text)
		return transcript_text
	else:
		st.write("## No Transcript found for this video")


# Getting the transcript data from yt videos
def extract_transcript_from_youtube(youtube_video_url):
	print("Inside extract_transcript_from_youtube()")
	#print ("youtube_video_url: " + youtube_video_url)
	try:
		video_id = youtube_video_url.split("v=")[1].split("&")[0]
		transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
		for transcript1 in transcript_list:
			transcript = transcript1.translate('en').fetch()
			print("------------------ Inside extract_transcript_from_youtube() transcript1 START ------------------\n")
			print(transcript1)
			print("------------------ Inside extract_transcript_from_youtube() transcript1 START ------------------\n")
			transcriptnew = ""
			for i in transcript:
				transcriptnew += " " + i["text"]
		return transcriptnew
	except Exception as e:
		print ("---------------- FOUND Error: ----------------\n" + str(e))
		return ""


def extract_transcript_manually(youtube_video_url, video_id):
	print("Inside extract_transcript_manually()")
	transcript_file = "yt_transcript_" + video_id + ".txt"
	download_youtube_audio(youtube_video_url, transcript_file)
	transcript_text = convert_audio_to_text(transcript_file)
	os.remove(transcript_file)		


def download_youtube_audio(url, transcript_file):
	print("Inside download_youtube_audio()")
	yt = YouTube(url)
	audio_stream = yt.streams.filter(only_audio=True).first()
	audio_stream.download(output_path=".", filename=transcript_file)


def convert_audio_to_text(transcript_file):
	print("Inside convert_audio_to_text()")
	try:
		transcriber = aai.Transcriber()
		transcript = transcriber.transcribe(transcript_file)	# Working
		return transcript.text
	except Exception as e:
		print ("---------------- FOUND Error While convert_audio_to_text: ----------------\n" + str(e))
		return ""	


# Getting the summary based on Prompt from Google Gemini Pro
def generate_gemini_content(transcript_text, prompt):
	print("Inside generate_gemini_content()")
	try:
		model = genai.GenerativeModel(model_name = "gemini-1.5-pro-latest", generation_config=generation_config, safety_settings=safety_settings)
		response = model.generate_content(prompt + transcript_text)
		print("---------------- model.count_tokens: ----------------\n" + str(model.count_tokens(transcript_text)))
		print ("---------------- response.prompt_feedback: ----------------\n" + str(response.prompt_feedback))
		print("---------------- response.candidates: ----------------\n" + str(response.candidates))
		print ("---------------- response: ----------------\n" + str(response))
		print ("---------------- response.text: ----------------\n" + str(response.text))
		return response.text
	except Exception as e:
		print ("---------------- FOUND Error: ----------------\n" + str(e))
		print ("---------------- response.prompt_feedback: ----------------\n" + str(response.prompt_feedback))
		return ""


def summarize_transcript_text(transcript_text, prompt):
	print("Inside summarize_transcript_text()")
	summary_temp = generate_gemini_content(transcript_text, prompt)
	summary_temp.replace("$", "\$")
	st.session_state.summary = summary_temp
	if summary_temp:
		st.write(st.session_state.summary)
	else:
		st.write("## LLM couldn't generate transcript summary")


def embed_transcript_text(transcript_text):
	text_chunks = get_text_chunks(transcript_text)
	get_vector_store(text_chunks)


def start_chat_session():
	print("Inside start_chat_session()")
	with st.sidebar:
		print ("Entered sidebar")
		
		# Container for chat messages
		chat_container = st.container(height=600)
		
		# Display chat messages from history on app rerun
		for message in st.session_state.messages:
			with chat_container:
				with st.chat_message(message["role"]):
					st.markdown(message["content"])
					
		user_question = st.chat_input("Ask a Question from this video")
		print ("Got user_question: " + str(user_question))
		if user_question:
			print("---------------- user_question inside start_chat_session: ----------------\n" + str(user_question))
			with chat_container:
				with st.chat_message("user"):
					st.markdown(user_question)
					# Add user message to chat history
					st.session_state.messages.append({"role": "user", "content": user_question})
				with st.spinner():
					handle_user_input(user_question)


def get_text_chunks(text):
	print("Inside get_text_chunks()")
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=1000)
	chunks = text_splitter.split_text(text)
	return chunks


def get_vector_store(text_chunks):
	print("Inside get_vector_store()")
	vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
	vector_store.save_local("faiss_index")


def handle_user_input(user_question):
	print("---------------- user_question Inside handle_user_input: ----------------\n" + str(user_question))
	new_db = FAISS.load_local("faiss_index", embeddings)
	docs = new_db.similarity_search(user_question)
	chain = get_conversational_chain()
	try:
		response = chain(
			{"input_documents":docs, "question": user_question},
			return_only_outputs=True)
	except Exception as e:
		response = {'output_text': "I am not supposed to answer that question, it might be harmful. Please ask me something different."}
	print("---------------- response: ----------------\n" + str(response))
	with st.chat_message("assistant"):
		st.markdown(response["output_text"])
		# Add assistant response to chat history
		st.session_state.messages.append({"role": "assistant", "content": response["output_text"]})


def get_conversational_chain():
	print("Inside get_conversational_chain()")
	prompt_template = """
	Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
	provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
	Context:\n {context}?\n
	Question: \n{question}\n

	Answer:
	"""

	model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", generation_config=generation_config, safety_settings=safety_settings1, google_api_key=os.getenv("GOOGLE_API_KEY"))
	prompt_conversation = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
	chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_conversation)
	return chain


st.set_page_config( page_title="YouTube Video Chat", page_icon="🧊")
st.title("YouTube Video Detailed Notes")
youtube_video_url = st.text_input("Enter YouTube Video Link:", value="https://www.youtube.com/watch?v=fYrAl8BIzJo")
#print ("youtube_video_url: " + youtube_video_url)

transcript_text = ""
#if st.button('Get Notes', on_click=click_button):
if st.button('Get Notes'):
	transcript_text = click_button()
	print("Entered st.button, so that means \"Get Notes\" is clicked and got transcript_text")
print("Exited st.button")

if st.session_state.clicked:
	print("------------------ Inside st.session_state.clicked START ------------------\n")
	print(transcript_text)
	print("------------------ Inside st.session_state.clicked END ------------------\n")
	start_chat_session()

st.write(st.session_state.summary)


