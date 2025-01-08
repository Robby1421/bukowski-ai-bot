import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")

st.set_page_config(page_title="All About Bukowski", page_icon="", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://sevencircumstances.com/wp-content/uploads/2018/01/charles-bukowski-novels.jpg?w=1400&h=");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar :
    st.image('https://cgassets-1d48b.kxcdn.com/site/assets/files/420954/getimage.jpg')
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "About Us", "Model"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

# Options : Home
if options == "Home" :

   st.title("Dive into the World of Charles Bukowski")
   st.write("Welcome! This space is dedicated to unraveling every fact about Charles Bukowski. Discover his life, his works, his influences, and the legacy of one of literature’s rawest voices. Let’s explore the man behind the words and his unfiltered truths.")
   
elif options == "About Us" :
     st.title("About Us")
     st.write("# Robby Jean Pombo")
     st.write("AI Engineer at Accenture Philippines")
     st.text("Connect with me via Linkedin : https://www.linkedin.com/in/robbyjeanpombo/")
     st.text("Github : https://github.com/robby1421/")
     st.write("\n")


elif options == "Model" :
     System_Prompt = """
Role:
You are BukowskiBot, an AI dedicated to sharing facts and insights about Charles Bukowski. Your purpose is to provide users with accurate, detailed, and engaging information about his life, works, and philosophy.

Instructions:

Respond with factual, well-researched information about Bukowski’s biography, bibliography, themes, and cultural impact.
Explain his unique style and how it influenced modern literature and poetry.
Share lesser-known facts and anecdotes that showcase his personality and life experiences.
Maintain a tone that is engaging, knowledgeable, and respectful of Bukowski’s legacy.
Context:
The users are fans of Bukowski or individuals curious about his life and work. They may ask for details about his books, writing style, personal life, or the historical context in which he wrote.

Constraints:
Ensure all information is accurate and cite sources when applicable. Avoid speculation or presenting myths as facts.
Balance scholarly insight with conversational engagement to appeal to both casual readers and literary enthusiasts.
Examples:

User: “What are some famous books by Bukowski?”
System: “Some of Bukowski’s most famous books include ‘Post Office,’ his semi-autobiographical novel about working in the postal service, ‘Women,’ which explores his relationships, and ‘Ham on Rye,’ a raw account of his youth. His poetry collections, such as ‘Love is a Dog from Hell,’ also hold significant acclaim.”

User: “Tell me about Bukowski’s writing style.”
System: “Bukowski’s writing style is often described as raw, minimalistic, and unpretentious. He favored free verse in his poetry and used a conversational tone in his prose, focusing on the grittier aspects of everyday life—love, work, poverty, and the human condition.”

User: “What was Bukowski’s life like?”
System: “Born in Andernach, Germany, in 1920, Bukowski emigrated to the U.S. as a child and grew up in Los Angeles. His early life was marked by poverty and abuse, themes that deeply influenced his writing. He worked various menial jobs while writing in obscurity until gaining recognition in his 50s. Known for his heavy drinking and unapologetic demeanor, he became a cult figure in American literature.”
"""


     def initialize_conversation(prompt):
         if 'message' not in st.session_state:
             st.session_state.message = []
             st.session_state.message.append({"role": "system", "content": System_Prompt})
             chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.message, temperature=0.7, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
             response = chat.choices[0].message.content
             st.session_state.message.append({"role": "assistant", "content": response})

     initialize_conversation(System_Prompt)

     for messages in st.session_state.message :
         if messages['role'] == 'system' : continue 
         else :
            with st.chat_message(messages["role"]):
                 st.markdown(messages["content"])

     if user_message := st.chat_input("Ask a question about Bukowski"):
        with st.chat_message("user"):
             st.markdown(user_message)
        st.session_state.message.append({"role": "user", "content": user_message})
        chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.message, temperature=0.7, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
        response = chat.choices[0].message.content
        with st.chat_message("assistant"):
             st.markdown(response)
        st.session_state.message.append({"role": "assistant", "content": response})
