import streamlit as st
import os
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import chromadb

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="AskDocX",
    page_icon="📚",
    initial_sidebar_state="expanded",
    layout="wide"
)

st.title("📚 AskDocX")
st.caption("A Document Question-Answering System")

st.info("Upload your PDFs, ask questions, and get detailed responses from your documents! 🥳")

# Define prompt template
prompt_template = """
Answer the question as detailed as possible from the provided context, make sure to provide all the details from the given context only.

Break your answer up into nicely readable paragraphs.

Context: {context}

Question: {question}

Answer: """

def embed(uploaded_pdfs):
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Read PDF content from all uploaded files
    text = ""
    for uploaded_pdf in uploaded_pdfs:
        pdf_reader = PdfReader(uploaded_pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # Create Chroma vector store
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        client=chroma_client
    ).as_retriever()
    
    return vector_store

def query_pdf(question, vector_store):
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8, google_api_key=api_key)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    docs = vector_store.get_relevant_documents(question)
    response = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True
    )
    
    return response['output_text']

def main():
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = None

    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_pdfs = st.file_uploader(
            "Upload your PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="You can upload multiple PDF files"
        )
        
        if uploaded_pdfs:
            with st.spinner("Processing PDFs..."):
                try:
                    st.session_state['vector_store'] = embed(uploaded_pdfs)
                    st.success("✅ PDFs processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")

    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_input(
            "🤔 Ask your question",
            placeholder="Type your question here...",
            key="input"
        )
    
    with col2:
        st.write("")  # Added for vertical alignment
        st.write("")  # Added for vertical alignment
        submit = st.button("Ask Question 🚀", type="primary", use_container_width=True)

    # Display response
    if submit and st.session_state['vector_store']:
        try:
            with st.spinner("Generating response..."):
                response = query_pdf(question, st.session_state['vector_store'])
            
            st.subheader("💡 Response")
            st.write(response)
            
            # Update chat history
            st.session_state['chat_history'].insert(0, ("Bot 🤖", response))
            st.session_state['chat_history'].insert(0, ("You 🙋‍♂️", question))
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

    # Chat history in an expander
    with st.expander("📝 Chat History"):
        for i, (role, text) in enumerate(st.session_state['chat_history']):
            if role == "You 🙋‍♂️":
                st.write(f"**{role}**")
                st.write(text)
            else:
                st.write(f"**{role}**")
                st.write(text)
            
            if i < len(st.session_state['chat_history']) - 1:
                st.divider()

if __name__ == "__main__":
    main()