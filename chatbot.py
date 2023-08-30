import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

def get_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # In case you want to save money (Local Embeddings Model via instructor-xl)
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vector_store

def get_memory_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history'] 
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    
def main():
    load_dotenv()
    st.set_page_config(page_title='Explore your PDFs', page_icon=':robot_face:')

    st.write(css, unsafe_allow_html=True)
    # persistent state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.conversation_history = None
        
    st.header('Explore your PDFs:robot_face:')
    user_question = st.text_input('Ask a question about your PDFs:')
    if user_question:
        handle_user_input(user_question)
    
    with st.sidebar:
        st.subheader('Your Documents')
        pdf_docs = st.file_uploader(
            'Upload your PDFs here:', accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing your PDFs...'):
                # get text from pdfs
                raw_text = get_text(pdf_docs)

                # get text chunks from pdfs
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
                # create vector store with pdf embeddings
                vector_store = get_vectorstore(text_chunks)

                # create memory chain for conversation
                st.session_state.conversation = get_memory_chain(vector_store)

if __name__ == '__main__':
    main()