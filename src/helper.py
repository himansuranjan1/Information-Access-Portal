import os
import logging
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory  # Keep this for now, will need update
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from huggingface_hub import HfFolder
from PyPDF2.errors import PdfReadError

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()
hf_api_key = os.getenv("hf_api_key")  # Match the name exactly as defined in the .env file

# Set the Hugging Face API key globally using HfFolder
if hf_api_key:
    HfFolder.save_token(hf_api_key)  # Save the token globally
    logging.info("Hugging Face API key successfully set globally.")
else:
    logging.error("HUGGINGFACE_API_KEY not found in environment variables.")
    raise ValueError("Hugging Face API key is required.")

# Initialize variables for lazy model loading
tokenizer = None
model = None

# Lazy loading of the model
def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        logging.info("Loading the model...")
        try:
            # Load GPT-Neo 1.3B
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
            model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
            
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

# Extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except PdfReadError:
            # Handle the error gracefully
            logging.error(f"Error reading {pdf.name}: PDF might be corrupted or incomplete.")
            text += f"\n[Error: Unable to read {pdf.name}]\n"
    return text

# Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

# Create a FAISS vector store from text chunks
def get_vector_store(text_chunks):
    # Load Hugging Face embeddings model without the token parameter
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Updated embedding model
    
    # Create FAISS vector store from the embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# Create a conversational chain for Q&A
def get_conversational_chain(vector_store):
    logging.info("Creating conversational chain...")
    
    # Set up conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # Deprecation warning here
    
    # Load the model and create a text generation pipeline
    load_model()
    
    # Create a HuggingFace text generation pipeline
    text_generation_pipeline = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        max_new_tokens=200,  # Generate up to 200 new tokens
        pad_token_id=tokenizer.eos_token_id  # Ensure proper padding
    )
    
    # Wrap the HuggingFace pipeline in LangChain's HuggingFacePipeline
    hf_pipeline = HuggingFacePipeline(pipeline=text_generation_pipeline)
    
    # Create the conversational retrieval chain
    try:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=hf_pipeline,  # Use the HuggingFacePipeline as the LLM
            retriever=vector_store.as_retriever(),
            memory=memory
        )
        logging.info("Conversational chain created successfully.")
    except Exception as e:
        logging.error(f"Error creating conversational chain: {e}")
        raise
    return conversation_chain