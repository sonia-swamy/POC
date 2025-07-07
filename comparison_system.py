import pandas as pd
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains.retrieval_qa.base import RetrievalQA
import os
from typing import Dict, Any, Tuple
import re

class DataComparisonSystem:
    def __init__(self, excel_file_path: str, persist_directory: str = "excel_chroma_db"):
        """
        Initialize comparison system with both pandas and ChromaDB approaches
        
        Args:
            excel_file_path: Path to Excel file
            persist_directory: Directory for ChromaDB storage
        """
        self.excel_file_path = excel_file_path
        self.persist_directory = persist_directory
        self.df = None
        self.vector_store = None
        self.rag_chain = None
        
        # Load data
        self._load_data()
        self._setup_chromadb()
    
    def _load_data(self):
        """Load Excel data into pandas DataFrame"""
        print(f"Loading Excel file: {self.excel_file_path}")
        try:
            if self.excel_file_path.endswith('.xls'):
                self.df = pd.read_excel(self.excel_file_path, engine='xlrd')
            else:
                self.df = pd.read_excel(self.excel_file_path, engine='openpyxl')
            
            print(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
            print(f"Columns: {list(self.df.columns)}")
            
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            raise
    
    def _setup_chromadb(self):
        """Setup ChromaDB vector store"""
        print("Setting up ChromaDB...")
        
        # Check if ChromaDB already exists
        if os.path.exists(self.persist_directory):
            print("Loading existing ChromaDB...")
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embedding_model,
                collection_name="excel_data"
            )
        else:
            print("Creating new ChromaDB...")
            self._create_chromadb()
        
        # Setup RAG chain
        llm = OllamaLLM(model="mistral", base_url="http://localhost:11434")
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )
    
    def _create_chromadb(self):
        """Create ChromaDB from DataFrame"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
        
        print("Converting data to documents...")
        documents = []
        
        for idx, row in self.df.iterrows():
            text_parts = []
            metadata = {'row_index': idx}
            
            for col in self.df.columns:
                if pd.notna(row[col]):
                    text_parts.append(f"{col}: {str(row[col])}")
                    metadata[col] = str(row[col])
            
            if text_parts:
                full_text = " | ".join(text_parts)
                
                if len(full_text) > 500:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )
                    chunks = text_splitter.split_text(full_text)
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={**metadata, 'chunk_index': chunk_idx}
                        )
                        documents.append(doc)
                else:
                    doc = Document(
                        page_content=full_text,
                        metadata=metadata
                    )
                    documents.append(doc)
        
        print(f"Created {len(documents)} documents")
        
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
        
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=self.persist_directory,
            collection_name="excel_data"
        )
        self.vector_store.persist()
    
    def _parse_natural_language_query(self, question: str) -> Dict[str, Any]:
        """
        Parse natural language question into pandas operations
        This is a simplified version - in practice, you'd use more sophisticated NLP
        """
        question_lower = question.lower()
        
        # Define patterns for different types of queries
        patterns = {
            'count': r'how many|count|total number',
            'filter': r'where|filter|show.*where|find.*where',
            'group': r'group by|by|per|each',
            'sort': r'sort|order|highest|lowest|top|bottom',
            'aggregate': r'average|mean|sum|total|maximum|minimum|max|min'
        }
        
        query_info = {
            'type': 'select',
            'columns': [],
            'filters': [],
            'group_by': [],
            'sort_by': [],
            'aggregation': None
        }
        
        # Detect query type
        for pattern_type, pattern in patterns.items():
            if re.search(pattern, question_lower):
                query_info['type'] = pattern_type
        
        # Extract column names mentioned in question
        for col in self.df.columns:
            if col.lower() in question_lower:
                query_info['columns'].append(col)
        
        return query_info
    
    def pandas_query(self, question: str) -> Tuple[Any, float]:
        """
        Execute pandas-based query (SQLAI.ai style)
        
        Returns:
            Tuple of (result, execution_time)
        """
        start_time = time.time()
        
        try:
            query_info = self._parse_natural_language_query(question)
            
            # Start with the full DataFrame
            result_df = self.df.copy()
            
            # Apply filters if any
            if query_info['filters']:
                for filter_condition in query_info['filters']:
                    result_df = result_df.query(filter_condition)
            
            # Apply aggregations
            if query_info['aggregation']:
                if query_info['aggregation'] == 'count':
                    result = len(result_df)
                elif query_info['aggregation'] == 'average':
                    result = result_df.mean()
                else:
                    result = result_df
            else:
                result = result_df
            
            execution_time = time.time() - start_time
            return result, execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            return f"Pandas query error: {str(e)}", execution_time
    
    def chromadb_query(self, question: str) -> Tuple[str, float]:
        """
        Execute ChromaDB RAG query
        
        Returns:
            Tuple of (result, execution_time)
        """
        start_time = time.time()
        
        try:
            result = self.rag_chain.invoke({"query": question})
            execution_time = time.time() - start_time
            return result["result"], execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            return f"ChromaDB query error: {str(e)}", execution_time
    
    def compare_queries(self, question: str) -> Dict[str, Any]:
        """
        Compare both approaches for a given question
        
        Returns:
            Dictionary with results and performance metrics
        """
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        # Run pandas query
        print("\n1. Pandas Query (SQLAI.ai style):")
        pandas_result, pandas_time = self.pandas_query(question)
        print(f"Execution time: {pandas_time:.3f} seconds")
        print(f"Result: {pandas_result}")
        
        # Run ChromaDB query
        print("\n2. ChromaDB RAG Query:")
        chromadb_result, chromadb_time = self.chromadb_query(question)
        print(f"Execution time: {chromadb_time:.3f} seconds")
        print(f"Result: {chromadb_result}")
        
        # Performance comparison
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON:")
        print(f"{'='*60}")
        print(f"Pandas Query Time: {pandas_time:.3f} seconds")
        print(f"ChromaDB Query Time: {chromadb_time:.3f} seconds")
        
        if pandas_time < chromadb_time:
            faster = "Pandas"
            speedup = chromadb_time / pandas_time
            print(f"Pandas is {speedup:.1f}x faster")
        else:
            faster = "ChromaDB"
            speedup = pandas_time / chromadb_time
            print(f"ChromaDB is {speedup:.1f}x faster")
        
        return {
            'question': question,
            'pandas_result': pandas_result,
            'pandas_time': pandas_time,
            'chromadb_result': chromadb_result,
            'chromadb_time': chromadb_time,
            'faster_method': faster,
            'speedup_factor': speedup
        }

def main():
    """Interactive comparison system"""
    excel_file = "BILLING_EFFICIENCY_DATA_2706.xlsx"
    
    try:
        # Initialize comparison system
        print("Initializing Data Comparison System...")
        comparison_system = DataComparisonSystem(excel_file)
        
        print(f"\n{'='*60}")
        print("COMPARISON SYSTEM READY")
        print("Ask questions to compare Pandas vs ChromaDB approaches")
        print("Type 'exit' to quit")
        print(f"{'='*60}")
        
        while True:
            question = input("\nEnter your question: ").strip()
            
            if question.lower() in ('exit', 'quit'):
                print("Exiting comparison system.")
                break
            
            if not question:
                print("Please enter a question or type 'exit' to quit.")
                continue
            
            try:
                comparison_system.compare_queries(question)
            except Exception as e:
                print(f"Error: {e}")
                print("Make sure Ollama is running with: ollama serve")
                
    except KeyboardInterrupt:
        print("\nExiting comparison system.")
    except Exception as e:
        print(f"Error initializing system: {e}")

if __name__ == "__main__":
    main() 