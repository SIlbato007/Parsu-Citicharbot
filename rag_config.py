import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings

# Load and set environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

def load_document(pdf_path):
    #Load a PDF document using UnstructuredPDFLoader.
    loader = UnstructuredPDFLoader(pdf_path)
    return loader.load()

def recursive_chunk(data, chunk_size=512, chunk_overlap=100):
   #Split loaded document into chunks using RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(data)

def create_embedding_model(api_key, model_name="BAAI/bge-base-en-v1.5"):
    #Create an embedding model instance.
    return HuggingFaceInferenceAPIEmbeddings(api_key=api_key, model_name=model_name)

def create_vector_store(chunks, embeddings, persist_directory):
    #Create or load a Chroma vector store.
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

def setup_retrievers(vector_store, chunks):
    """Create an ensemble retriever combining BM25 and vector search."""
    kb_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 5
    return EnsembleRetriever(retrievers=[kb_retriever, keyword_retriever], weights=[0.5, 0.5])

def setup_llm():
    #Initialize the LLM via HuggingFaceHub with specified parameters.
    return HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={
            "max_new_tokens": 512,
            "temperature": 0.5,
            "repetition_penalty": 1.1,
            "return_full_text": False
        }
    )

def setup_prompt_template():
    """Create a prompt template and output parser for the chain."""
    template = (
        "<|system|>\n"
        "You are a friendly AI Assistant for Partido State University that understands users extremely well and always responds professionally.\n"
        "Please be truthful and give direct answers. If the user query is not in CONTEXT, then ask again for clarification.\n"
        "If the query does not find a match, then let the user know.</s>\n"
        "CONTEXT: {context}</s>\n"
        "<|user|> {query} </s>\n"
        "<|assistant|>"
    )
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    return prompt, output_parser

def assemble_chain(retriever, prompt, llm, output_parser):
    """Assemble the chain using retriever, prompt, LLM, and output parser."""
    return (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )

class PSUChatBackend:
    def __init__(self, persist_directory="chroma_db"):
        """Initialize backend with a persistent directory for ChromaDB."""
        self.chain = None
        self.persist_directory = persist_directory

    def initialize_system(self, pdf_path="data/charter_data.pdf"):
        """Initialize the entire system, reusing stored embeddings if available."""
        try:
            embeddings = create_embedding_model(HF_TOKEN)

            # Check if ChromaDB exists; load or create as needed
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings)
                chunks = None  # No need to process document again
            #perform the indexing again
            else:
                data = load_document(pdf_path)
                chunks = recursive_chunk(data)
                vector_store = create_vector_store(chunks, embeddings, self.persist_directory)

            # Setup retrievers
            retriever = setup_retrievers(vector_store, chunks) if chunks else vector_store.as_retriever()

            # Setup LLM and prompt chain components
            llm = setup_llm()
            prompt, output_parser = setup_prompt_template()
            self.chain = assemble_chain(retriever, prompt, llm, output_parser)

            return True, "System initialized successfully!"
        except Exception as e:
            return False, f"Error during initialization: {str(e)}"
    
    def generate_response(self, query):
        if not self.chain:
            return False, "System not initialized. Please initialize first."
        try:
            response = self.chain.invoke(query)
            return True, response
        except Exception as e:
            return False, f"Error generating response: {str(e)}"
