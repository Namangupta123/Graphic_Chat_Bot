# Using Streamlit
import streamlit as st
from langchain_community.llms import Cohere
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

st.set_page_config(
    page_title="Graphic Era ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'About': 'Hello'
    }
)
TEXT_FILE_PATH = "GraphicEra.txt"

def text_file_to_text(text_file):
    with open(text_file, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def text_splitter(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        separators=['\n', '\n\n', ' ', ',']
    )
    chunks = text_splitter.split_text(text=raw_text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_answer(question, retriever):
    cohere_llm = Cohere(model="command", temperature=0.1, cohere_api_key=os.getenv("COHERE_API_KEY"))

    prompt_template = """Answer the question as broadly as possible using the provided context. If the answer is
                    not contained in the context, say "Sorry the answer is not available in context" "\n\n
                    Context: \n {context} \n\n
                    Question: \n {question} \n
                    Answer:"""

    prompt = PromptTemplate.from_template(template=prompt_template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | cohere_llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)

def main():
    st.header("Graphic Era ChatBot")
    st.write("Hello, welcome! Feel free to ask me anything about Uttarakhand top private university.")

    question = st.text_input("Ask a question:")

    if st.button("Ask Questions"):
        with st.spinner("Please have patience for a moment I am searching the available information :)"):
            raw_text = text_file_to_text(TEXT_FILE_PATH)
            text_chunks = text_splitter(raw_text)
            vectorstore = get_vector_store(text_chunks)
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

            if question:
                if(generate_answer(question, retriever)):
                   answer = generate_answer(question, retriever)
                   st.write(answer)
                else:
                   st.write("Sorry, but I don't have that information.")
            else:
                st.warning("Please enter a question.")


if __name__ == "__main__":
    main()

