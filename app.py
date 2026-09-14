import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Page configuration
st.set_page_config(page_title="PDF RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Search Assistant")
st.caption("Powered by Groq (`openai/gpt-oss-120b`), FAISS & LangChain")

# Retrieve Groq API Key from Streamlit Secrets or Environment Variable
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.warning("Please configure `GROQ_API_KEY` in Streamlit Secrets or as an environment variable.")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Cache the lightweight embedding model to avoid reloading on every rerun."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


def process_pdf(uploaded_file):
    """Processes PDF file: extracts text, splits into chunks, and returns a FAISS vector store."""
    # Write uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 1. Extract text from PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # 2. Tokenize/Chunk text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)

        # 3. Create embeddings & store in FAISS index
        vector_store = FAISS.from_documents(chunks, embedding_model)
        return vector_store
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# Sidebar - PDF Upload Section
with st.sidebar:
    st.header("Document Upload")
    uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_pdf and "vector_store" not in st.session_state:
        with st.spinner("Extracting PDF, generating chunks, and indexing into FAISS..."):
            st.session_state.vector_store = process_pdf(uploaded_pdf)
            st.session_state.file_name = uploaded_pdf.name
            st.success("PDF successfully indexed!")

    if "file_name" in st.session_state:
        st.info(f"Active Document: **{st.session_state.file_name}**")


# Main Interface - Question Answering
if "vector_store" in st.session_state:
    # Initialize Groq Chat Model
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="openai/gpt-oss-120b",
        temperature=0.2,
    )

    # Define Prompt Template
    prompt_template = ChatPromptTemplate.from_template(
        """Answer the following question based strictly on the provided context.
If the answer cannot be found in the context, say "I cannot find the answer in the provided document."

<context>
{context}
</context>

Question: {input}"""
    )

    # Set up Retriever & Chain
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    user_query = st.text_input("Ask a question about your PDF:")

    if user_query:
        with st.spinner("Retrieving relevant context and generating answer..."):
            response = rag_chain.invoke({"input": user_query})
            st.markdown("### Answer")
            st.write(response["answer"])

            with st.expander("View Retrieved Chunks"):
                for idx, doc in enumerate(response["context"]):
                    st.markdown(f"**Chunk {idx + 1} (Page {doc.metadata.get('page', 'N/A')}):**")
                    st.text(doc.page_content)
else:
    st.info("👈 Please upload a PDF document from the sidebar to start asking questions.")
