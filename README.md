# Agentic AI RAG System for Customer Visit Data

This Jupyter notebook implements an **Agentic AI Retrieval-Augmented Generation (RAG) system** designed to analyze and answer queries related to customer visit data. The system leverages natural language processing, vector embeddings, and a multi-agent workflow to provide insightful responses based on structured and unstructured data.

## Overview

The notebook covers the following key components:

- **Environment Setup:** Downloads necessary NLP resources and configures environment variables.
- **Data Loading and Preprocessing:** Loads customer visit data from an Excel file, cleans and normalizes text fields, handles missing values, and extracts key entities.
- **Embedding Generation:** Uses the `SentenceTransformer` model (`all-MiniLM-L6-v2`) to generate dense vector embeddings for combined textual and metadata fields.
- **Vector Database Setup:** Uses ChromaDB to store embeddings and metadata for efficient similarity search.
- **Agent Workflow:** Implements a multi-agent system using AutoGen and Ollama LLM to:
  - Analyze user queries to extract intent and entities.
  - Retrieve relevant data from the vector database.
  - Generate comprehensive, structured responses.
- **Query Processing Pipeline:** Processes user queries end-to-end, including temporal reference extraction, vector search, context augmentation, and response generation.
- **Interactive Query Interface:** Allows users to input natural language queries and receive AI-generated answers based on the customer visit data.

## Setup Instructions

1. **Install Dependencies**

   Ensure you have Python 3.8+ installed. Install required packages using:

   ```bash
   pip install -r requirements.txt
   ```

   Required packages include:
   - `nltk`
   - `pandas`
   - `sentence-transformers`
   - `chromadb`
   - `autogen-agentchat`
   - `autogen-ext`
   - `ollama`
   - `dateparser`
   - `tqdm`
   - `torch`

2. **Download NLTK Data**

   The notebook automatically downloads necessary NLTK data such as `punkt`, `stopwords`, and `wordnet`.

3. **Prepare Dataset**

   Place your customer visit data Excel file named `SMData.xlsx` in the working directory. The dataset should include columns such as:
   - `Next Steps`
   - `Outcome of meeting`
   - `Visit Plan: Owner Name Customer`
   - `Visit Plan: Product Division`
   - Other relevant metadata fields.

4. **Environment Variables**

   The notebook creates a `.env` file with paths for model storage and ChromaDB persistence:
   - `TRANSFORMERS_HOME=./models`
   - `CHROMADB_PATH=./chroma_db`

5. **Run the Notebook**

   Execute the notebook cells sequentially to:
   - Load and preprocess data.
   - Generate embeddings and populate the vector database.
   - Initialize the agent workflow.
   - Interactively query the system.

## Usage

- Enter natural language queries related to customer visits, such as:
  - "What next steps were planned for Ganges International Pvt Ltd?"
  - "Show visit outcomes for the last quarter."
- The system will analyze the query, retrieve relevant data, and generate a detailed response.

## Architecture Details

- **Data Preprocessing:** Cleans text, removes stopwords, lemmatizes tokens, and standardizes formats.
- **Embedding Model:** Uses `all-MiniLM-L6-v2` for efficient sentence embeddings.
- **Vector Store:** ChromaDB stores embeddings with associated metadata for fast similarity search.
- **Agents:**
  - **Query Analyzer:** Extracts intent, entities, time ranges, and filters from queries.
  - **Data Retriever:** Searches the vector database and filters results.
  - **Response Generator:** Produces human-readable answers based on retrieved data.
  - **Orchestrator:** Manages the workflow between agents.
- **RAG Processor:** Combines retrieval and generation for enhanced responses.

## Notes

- The system uses Ollama LLM models; ensure Ollama is installed and configured.
- The notebook includes error handling and fallback mechanisms for robustness.
- The vector database is persisted locally in the `./chroma_db` directory.

## License

This project is provided as-is under the MIT License.

---

For detailed code and explanations, please refer to the `main.ipynb` notebook.
