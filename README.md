# Excel to ChromaDB RAG System

A powerful Retrieval-Augmented Generation (RAG) system that converts Excel files (.xls/.xlsx) into searchable ChromaDB vector stores for intelligent querying and analysis.

## Features

- **Excel File Support**: Handles both .xls and .xlsx file formats
- **Automatic Document Processing**: Converts Excel rows into searchable documents with metadata
- **ChromaDB Integration**: Creates persistent vector stores for efficient retrieval
- **RAG Capabilities**: Query your Excel data using natural language
- **Metadata Preservation**: Maintains all column data as searchable metadata
- **Automatic Chunking**: Splits long content into manageable chunks
- **Persistent Storage**: Saves ChromaDB for reuse without reprocessing

## Installation

1. **Clone or download the project files**

2. **Install required dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install Ollama** (for local LLM inference):
```bash
# Download and install Ollama from https://ollama.ai/
ollama pull mistral
ollama serve
```

## Quick Start

1. **Place your Excel file** in the project directory

2. **Update the file path** in `create_excel_chromadb.py`:
```python
excel_file = "your_data.xlsx"  # Change to your file path
```

3. **Run the script**:
```bash
python3 create_excel_chromadb.py
```

## Usage

### Basic Usage

```python
from create_excel_chromadb import create_chromadb_from_excel, query_excel_data

# Create ChromaDB from Excel file
vector_store = create_chromadb_from_excel("your_data.xlsx")

# Query the data
answer = query_excel_data(vector_store, "What are the main trends in this data?")
print(answer)
```

### Advanced Usage

```python
from create_excel_chromadb import ExcelChromaDB

# Initialize with custom settings
excel_chroma = ExcelChromaDB(persist_directory="my_chroma_db")

# Load Excel file
df = excel_chroma.load_excel_file("data.xlsx")

# Convert to documents with custom chunking
documents = excel_chroma.dataframe_to_documents(
    df, 
    chunk_size=1000, 
    chunk_overlap=100
)

# Create ChromaDB
excel_chroma.create_chromadb(documents, collection_name="my_data")

# Create RAG chain
excel_chroma.create_rag_chain(k=5)

# Query
answer = excel_chroma.query("What patterns do you see in the data?")
print(answer)
```

## File Structure

```
RAG/
├── data/
│   └── Alice.txt                    # Sample text file
├── excel_chroma_db/                 # ChromaDB storage directory
│   ├── chroma.sqlite3
│   └── ...
├── faiss_index/                     # FAISS vector store files
│   ├── index.faiss
│   └── index.pkl
├── create_excel_chromadb.py         # Main Excel to ChromaDB script
├── rag1.py                          # Original RAG implementation
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Configuration

### Embedding Model
The system uses `all-MiniLM-L6-v2` by default. You can change this:

```python
self.embedding_model = HuggingFaceEmbeddings(model_name="your-model-name")
```

### LLM Model
Uses Ollama's Mistral model by default:

```python
self.llm = OllamaLLM(model="mistral", base_url="http://localhost:11434")
```

### Chunking Parameters
Adjust document chunking:

```python
documents = excel_chroma.dataframe_to_documents(
    df, 
    chunk_size=500,    # Size of each chunk
    chunk_overlap=50   # Overlap between chunks
)
```

### Retrieval Parameters
Control how many documents to retrieve:

```python
excel_chroma.create_rag_chain(k=3)  # Retrieve 3 documents
```

## API Reference

### ExcelChromaDB Class

#### `__init__(persist_directory="excel_chroma_db")`
Initialize the Excel to ChromaDB converter.

#### `load_excel_file(file_path, sheet_name=None)`
Load Excel file and return DataFrame.

#### `dataframe_to_documents(df, text_columns=None, chunk_size=500, chunk_overlap=50)`
Convert DataFrame to documents for ChromaDB.

#### `create_chromadb(documents, collection_name="excel_data")`
Create ChromaDB vector store from documents.

#### `load_existing_chromadb(collection_name="excel_data")`
Load existing ChromaDB collection.

#### `create_rag_chain(k=3)`
Create RAG chain for querying.

#### `query(question)`
Query the RAG system.

#### `get_collection_info()`
Get information about the ChromaDB collection.

### Standalone Functions

#### `create_chromadb_from_excel(excel_file_path, persist_directory="excel_chroma_db", collection_name="excel_data")`
Create ChromaDB from Excel file in one step.

#### `query_excel_data(vector_store, question, k=3)`
Query existing vector store.

#### `load_existing_chromadb(persist_directory="excel_chroma_db", collection_name="excel_data")`
Load existing ChromaDB collection.

## Example Use Cases

### 1. Customer Data Analysis
```python
# Analyze customer billing data
vector_store = create_chromadb_from_excel("customer_billing.xlsx")
answer = query_excel_data(vector_store, "What are the most common billing issues?")
```

### 2. Sales Data Insights
```python
# Get insights from sales data
vector_store = create_chromadb_from_excel("sales_data.xlsx")
answer = query_excel_data(vector_store, "What are the top performing products?")
```

### 3. Inventory Management
```python
# Query inventory data
vector_store = create_chromadb_from_excel("inventory.xlsx")
answer = query_excel_data(vector_store, "Which items are running low on stock?")
```

## Troubleshooting

### Common Issues

#### Ollama Connection Error
```
Error: Connection refused
```
**Solution**: Make sure Ollama is running:
```bash
ollama serve
```

#### Excel File Not Found
```
Error loading Excel file: [Errno 2] No such file or directory
```
**Solution**: Check the file path in the script and ensure the file exists.

#### Memory Issues
For large Excel files, consider:
- Reducing chunk size
- Processing in batches
- Using a machine with more RAM

#### ChromaDB Persistence Issues
If ChromaDB fails to load:
- Delete the `excel_chroma_db` directory and recreate
- Check file permissions
- Ensure sufficient disk space

### Performance Tips

1. **Optimize chunk size** based on your data characteristics
2. **Use appropriate embedding models** for your domain
3. **Monitor memory usage** when processing large files
4. **Consider batch processing** for very large datasets
5. **Use SSD storage** for better ChromaDB performance

## Dependencies

- **pandas**: Excel file processing
- **xlrd**: .xls file support
- **openpyxl**: .xlsx file support
- **chromadb**: Vector database
- **langchain**: RAG framework
- **langchain-huggingface**: Embeddings
- **langchain-ollama**: LLM integration
- **sentence-transformers**: Text embeddings
- **ollama**: Local LLM server

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the example code
3. Open an issue on the repository

## Changelog

### Version 1.0.0
- Initial release
- Excel file support (.xls/.xlsx)
- ChromaDB integration
- RAG capabilities
- Persistent storage
- Metadata preservation 