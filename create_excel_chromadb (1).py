import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain.chains.retrieval_qa.base import RetrievalQA
import os

def create_chromadb_from_excel(excel_file_path, persist_directory="excel_chroma_db", collection_name="excel_data"):
    """
    Create ChromaDB from Excel file
    
    Args:
        excel_file_path: Path to your Excel file
        persist_directory: Directory to store ChromaDB
        collection_name: Name for the collection
    """
    
    # Step 1: Load Excel file
    print(f"Loading Excel file: {excel_file_path}")
    try:
        if excel_file_path.endswith('.xls'):
            df = pd.read_excel(excel_file_path, engine='xlrd')
        else:
            df = pd.read_excel(excel_file_path, engine='openpyxl')
        
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None
    
    # Step 2: Convert DataFrame to documents
    print("Converting data to documents...")
    documents = []
    
    for idx, row in df.iterrows():
        # Create text content from all columns
        text_parts = []
        metadata = {'row_index': idx}
        
        for col in df.columns:
            if pd.notna(row[col]):
                text_parts.append(f"{col}: {str(row[col])}")
                metadata[col] = str(row[col])
        
        if text_parts:
            full_text = " | ".join(text_parts)
            
            # Split into chunks if text is too long
            if len(full_text) > 500:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                chunks = text_splitter.split_text(full_text)
                
                for chunk_idx, chunk in enumerate(chunks):
                    from langchain.schema import Document
                    doc = Document(
                        page_content=chunk,
                        metadata={**metadata, 'chunk_index': chunk_idx}
                    )
                    documents.append(doc)
            else:
                from langchain.schema import Document
                doc = Document(
                    page_content=full_text,
                    metadata=metadata
                )
                documents.append(doc)
    
    print(f"Created {len(documents)} documents")
    
    # Step 3: Create embeddings and ChromaDB
    print("Creating embeddings and ChromaDB...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create directory if it doesn't exist
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    # Create ChromaDB
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    vector_store.persist()
    print(f"ChromaDB created successfully in: {persist_directory}")
    
    return vector_store

def query_excel_data(vector_store, question, k=3):
    """
    Query the Excel data using RAG
    
    Args:
        vector_store: ChromaDB vector store
        question: Your question
        k: Number of documents to retrieve
    """
    
    # Create LLM
    llm = OllamaLLM(model="mistral", base_url="http://localhost:11434")
    
    # Create retriever and RAG chain
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    
    # Query
    result = rag_chain.invoke({"query": question})
    return result["result"]

def load_existing_chromadb(persist_directory="excel_chroma_db", collection_name="excel_data"):
    """
    Load existing ChromaDB collection
    
    Args:
        persist_directory: Directory where ChromaDB is stored
        collection_name: Name of the collection
    """
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=collection_name
    )
    
    return vector_store

# Example usage
if __name__ == "__main__":
    # Replace with your Excel file path
    excel_file = "BILLING_EFFICIENCY_DATA_2706.xlsx"  # Change this to your file path
    
    # Create ChromaDB from Excel
    print("="*50)
    print("Creating ChromaDB from Excel file")
    print("="*50)
    
    vector_store = create_chromadb_from_excel(excel_file)
    
    if vector_store:
        print("\n" + "="*50)
        print("Interactive Query Mode (type 'exit' to quit)")
        print("="*50)
        try:
            while True:
                user_question = input("\nEnter your question: ").strip()
                if user_question.lower() in ("exit", "quit"): 
                    print("Exiting.")
                    break
                if not user_question:
                    print("Please enter a question or type 'exit' to quit.")
                    continue
                try:
                    answer = query_excel_data(vector_store, user_question)
                    print(f"Answer: {answer}")
                except Exception as e:
                    print(f"Error: {e}")
                    print("Make sure Ollama is running with: ollama serve")
        except KeyboardInterrupt:
            print("\nExiting.")
    else:
        print("Failed to create ChromaDB. Please check your Excel file path.") 