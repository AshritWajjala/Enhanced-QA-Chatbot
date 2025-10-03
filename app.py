import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

# LangSmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OpenAI"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful personal assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(choice, question, api_key, llm, temperature, max_tokens):
    if choice == "OpenAI(Paid)":
        openai.api_key = api_key
        llm = ChatOpenAI(model=llm)
    elif choice == "Ollama(Free)":
        llm = ChatOllama(model=llm)
    else:
        pass
    parser = StrOutputParser()
    chain = prompt|llm|parser
    answer = chain.invoke({"question": question})

    return answer

# Streamlit

# Title
st.title("Enhanced Q&A Chatbot with OpenAI/Ollama")

# Sidebar for settings
st.sidebar.title("Settings")

# Selection of OpenAI/Ollama
choice = st.sidebar.selectbox("Select your choice.", options=["Select...", "OpenAI(Paid)", "Ollama(Free)"], )

api_key = None
llm = None

if choice == "OpenAI(Paid)":
    api_key = st.sidebar.text_input("Enter your OpenAI API key.", type="password")
    # Dropdown --> select OpenAI models
    llm = st.selectbox("Select an OpenAI model.", ["gpt-5", "gpt-4o", "gpt-4-turbo", "gpt-4"])
elif choice == "Ollama(Free)":
    # Dropdown --> select OpenAI models
    llm = st.selectbox("Select an OpenAI model.", ["llama3", "llama2"])  
else:
    st.sidebar.info("Please select a provider to continue.") 


# Slider --> adjust parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Ask anything.")
user_input = st.text_input("You: ")

if choice == "Select...":
    st.write("Please provide the query.")
else:
    response = generate_response(choice, user_input, api_key, llm, temperature, max_tokens)
    st.write(response)