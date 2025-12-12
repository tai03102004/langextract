"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

This application allows users to upload a PDF, process it,
and then ask questions about the content using a selected language model.
"""

import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
import warnings

# Suppress torch warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend


# Set protobuf environment variable to avoid error messages
# This might cause some issues with latency but it's a tradeoff
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Define persistent directory for ChromaDB
PERSIST_DIRECTORY = os.path.join("data", "vectors")

# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def extract_model_names(models_info: Any) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.

    Args:
        models_info: Response from ollama.list()

    Returns:
        Tuple[str, ...]: A tuple of model names.
    """
    logger.info("Extracting model names from models_info")
    try:
        # The new response format returns a list of Model objects
        if hasattr(models_info, "models"):
            # Extract model names from the Model objects
            model_names = tuple(model.model for model in models_info.models)
        else:
            # Fallback for any other format
            model_names = tuple()
            
        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()

def create_vector_db(file_upload) -> Chroma:
    """
    Create a vector database from an uploaded PDF file using Docling.
    Better table extraction with TableFormer.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info(f"Creating vector DB with Docling from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
    
    try:
        # Configure Docling pipeline
        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )
        
        result = converter.convert(path)
        logger.info("PDF converted with Docling")
        
        # Convert to LangChain documents
        full_markdown = result.document.export_to_markdown()

        doc = Document(
            page_content=full_markdown,
            metadata={
                "source": file_upload.name,
                "extraction_method": "docling_tableformer",
                "total_pages": len(result.document.pages) if hasattr(result.document, 'pages') else 1
            }
        )
        documents = [doc]
        
        logger.info(f"Converted {len(documents)} pages to LangChain documents")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=7500, 
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Document split into {len(chunks)} chunks")

        # Create vector database
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=f"docling_pdf_{hash(file_upload.name)}"
        )
        logger.info("Vector DB created with Docling extraction")

    except Exception as e:
        logger.error(f"Error during Docling processing: {e}")
        raise
    finally:
        shutil.rmtree(temp_dir)
        logger.info(f"Temporary directory {temp_dir} removed")
    
    return vector_db

def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model.

    Args:
        question (str): The user's question.
        vector_db (Chroma): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    logger.info(f"Processing question: {question} using model: {selected_model}")
    
    # Initialize LLM
    llm = ChatOllama(model=selected_model)
    
    # Query prompt template
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 2
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    # Set up retriever
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    # RAG prompt template
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Create chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.

    Args:
        vector_db (Optional[Chroma]): The vector database to be deleted.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        try:
            # Delete the collection
            vector_db.delete_collection()
            
            # Clear session state
            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_upload", None)
            st.session_state.pop("vector_db", None)
            
            st.success("Collection and temporary files deleted successfully.")
            logger.info("Vector DB and related session state cleared")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")
            logger.error(f"Error deleting collection: {e}")
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")


def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    st.subheader("🧠 Ollama PDF RAG playground", divider="gray", anchor=False)

    # Get available models
    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    # Create layout
    col1, col2 = st.columns([1.5, 2])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "use_sample" not in st.session_state:
        st.session_state["use_sample"] = False
    if "processed_files" not in st.session_state: 
        st.session_state["processed_files"] = []

    # Model selection
    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓", 
            available_models,
            key="model_select"
        )

    # Add checkbox for sample PDF
    use_sample = col1.toggle(
        "Use sample PDF (Scammer Agent Paper)", 
        key="sample_checkbox"
    )
    
    # Clear vector DB if switching between sample and upload
    if use_sample != st.session_state.get("use_sample"):
        if st.session_state["vector_db"] is not None:
            st.session_state["vector_db"].delete_collection()
            st.session_state["vector_db"] = None
            st.session_state["pdf_pages"] = None
            st.session_state["processed_files"] = []
        st.session_state["use_sample"] = use_sample

    if use_sample:
        # Use the sample PDF
        sample_path = "../data/scammer-agent.pdf"
        if os.path.exists(sample_path):
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing sample PDF with Docling..."):
                    try:
                        # Configure Docling pipeline
                        pipeline_options = PdfPipelineOptions(do_table_structure=True)
                        pipeline_options.do_table_structure = True
                        pipeline_options.table_structure_options.do_cell_matching = True
                        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
                        converter = DocumentConverter(
                            format_options={
                                InputFormat.PDF: PdfFormatOption(
                                    pipeline_options=pipeline_options,
                                    backend=PyPdfiumDocumentBackend
                                )
                            }
                        )
                        result = converter.convert(sample_path)
                        
                        # Convert to LangChain documents
                        full_markdown = result.document.export_to_markdown()

                        
                        # Split and create vector DB
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=7500, 
                            chunk_overlap=100
                        )

                        doc = Document(
                            page_content=full_markdown,
                            metadata={
                                "source": "scammer-agent.pdf",
                                "extraction_method": "docling_tableformer",
                                "total_pages": len(result.document.pages) if hasattr(result.document, 'pages') else 1
                            }
                        )
                        documents = [doc]
                        chunks = text_splitter.split_documents(documents)
                        
                        st.session_state["vector_db"] = Chroma.from_documents(
                            documents=chunks,
                            embedding=OllamaEmbeddings(model="nomic-embed-text"),
                            persist_directory=PERSIST_DIRECTORY,
                            collection_name="docling_sample_pdf"
                        )
                        
                        # Open and display the sample PDF
                        with pdfplumber.open(sample_path) as pdf:
                            st.session_state["pdf_pages"] = [
                                page.to_image().original for page in pdf.pages
                            ]
                        
                        st.success("✅ Sample PDF processed with Docling TableFormer!")
                        
                    except Exception as e:
                        st.error(f"Error processing with Docling: {str(e)}")
                        logger.error(f"Docling processing error: {e}")
        else:
            st.error("Sample PDF file not found in the current directory.")
    else:
        # Regular file upload with unique key
        file_uploads = col1.file_uploader(
            "Upload a PDF file ↓", 
            type="pdf", 
            accept_multiple_files=True,
            key="pdf_uploader"
        )

        if file_uploads:
            # Hiển thị số files đã upload
            col1.info(f"📄 {len(file_uploads)} file(s) uploaded")
            
            # Lấy danh sách files chưa xử lý
            new_files = [f for f in file_uploads if f.name not in st.session_state["processed_files"]]
            
            if new_files:
                with st.spinner(f"Processing {len(new_files)} new PDF(s) with Docling..."):
                    try:
                        all_chunks = []
                        all_pdf_pages = []
                        
                        for file_upload in new_files:
                            logger.info(f"Processing file: {file_upload.name}")
                            
                            # Process từng file
                            temp_dir = tempfile.mkdtemp()
                            path = os.path.join(temp_dir, file_upload.name)
                            
                            with open(path, "wb") as f:
                                f.write(file_upload.getvalue())
                            
                            try:
                                # Configure Docling
                                pipeline_options = PdfPipelineOptions(do_table_structure=True)
                                pipeline_options.table_structure_options.do_cell_matching = True
                                pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
                                
                                converter = DocumentConverter(
                                    format_options={
                                        InputFormat.PDF: PdfFormatOption(
                                            pipeline_options=pipeline_options,
                                            backend=PyPdfiumDocumentBackend
                                        )
                                    }
                                )
                                
                                result = converter.convert(path)
                                full_markdown = result.document.export_to_markdown()
                                
                                doc = Document(
                                    page_content=full_markdown,
                                    metadata={
                                        "source": file_upload.name,
                                        "extraction_method": "docling_tableformer",
                                        "total_pages": len(result.document.pages) if hasattr(result.document, 'pages') else 1
                                    }
                                )
                                
                                # Split document
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=7500, 
                                    chunk_overlap=100
                                )
                                chunks = text_splitter.split_documents([doc])
                                all_chunks.extend(chunks)
                                
                                # Extract pages for display
                                with pdfplumber.open(file_upload) as pdf:
                                    pages = [page.to_image().original for page in pdf.pages]
                                    all_pdf_pages.extend(pages)
                                
                                # Mark as processed
                                st.session_state["processed_files"].append(file_upload.name)
                                logger.info(f"Successfully processed: {file_upload.name}")
                                
                            finally:
                                shutil.rmtree(temp_dir)
                        
                        # Create or update vector DB
                        embeddings = OllamaEmbeddings(model="nomic-embed-text")
                        
                        if st.session_state["vector_db"] is None:
                            # Create new vector DB
                            st.session_state["vector_db"] = Chroma.from_documents(
                                documents=all_chunks,
                                embedding=embeddings,
                                persist_directory=PERSIST_DIRECTORY,
                                collection_name="multi_pdf_collection"
                            )
                            st.session_state["pdf_pages"] = all_pdf_pages
                        else:
                            # Add to existing vector DB
                            st.session_state["vector_db"].add_documents(all_chunks)
                            if "pdf_pages" not in st.session_state:
                                st.session_state["pdf_pages"] = []
                            st.session_state["pdf_pages"].extend(all_pdf_pages)
                        
                        st.success(f"✅ {len(new_files)} PDF(s) processed with Docling TableFormer!")
                        st.info(f"📚 Total documents in database: {len(st.session_state['processed_files'])}")
                        
                    except Exception as e:
                        st.error(f"Error processing PDFs: {str(e)}")
                        logger.error(f"Error processing PDFs: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                st.info(f"✅ All {len(file_uploads)} file(s) already processed")
                
            # Hiển thị danh sách files đã xử lý
            with col1.expander("📚 Processed files"):
                for idx, filename in enumerate(st.session_state["processed_files"], 1):
                    st.text(f"{idx}. {filename}")

    # Display PDF if pages are available
    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        # PDF display controls
        zoom_level = col1.slider(
            "Zoom Level", 
            min_value=100, 
            max_value=1000, 
            value=700, 
            step=50,
            key="zoom_slider"
        )

        # Display PDF pages
        with col1:
            with st.container(height=410, border=True):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width=zoom_level)

    # Delete collection button
    delete_collection = col1.button(
        "⚠️ Delete collection", 
        type="secondary",
        key="delete_button"
    )

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    # Chat interface
    with col2:
        message_container = st.container(height=500, border=True)

        # Display chat history
        for i, message in enumerate(st.session_state["messages"]):
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Chat input and processing
        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            try:
                # Add user message to chat
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)

                # Process and display assistant response
                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a PDF file first.")

                # Add assistant response to chat history
                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file or use the sample PDF to begin chat...")


if __name__ == "__main__":
    main()