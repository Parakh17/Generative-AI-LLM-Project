import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv

os.environ['OPENAI_API_KEY']='YOUR_API_KEY_HERE'
llm = OpenAI(temperature=0.9, max_tokens=500)

st.title("News Research Langchain Model")

st.sidebar.title("Article URLs")

filepath = "faissmodelnew"

embeddings = OpenAIEmbeddings()

urls=[]
for i in range(3):

    url=st.sidebar.text_input(f" URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

if process_url_clicked:

    #load_data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading..........Started✅✅✅✅✅")
    data=loader.load()

    #split data
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000
    )
    main_placeholder.text("Text Splitter..........Started✅✅✅✅✅")

    docs = text_splitter.split_documents(data)

    #create emberddings and load FAISS

    

    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building..........Started✅✅✅✅✅")
    time.sleep(2)

    vectorstore_openai.save_local("faissmodelnew")

query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(filepath):
        #with open(filepath, "rb") as f:
        vectorstore = FAISS.load_local("faissmodelnew", embeddings, allow_dangerous_deserialization=True)

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question":query}, return_only_outputs=True)


        st.header("Answer")
        st.subheader(result["answer"])





