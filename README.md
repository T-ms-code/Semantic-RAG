# Semantic RAG with Metadata Extraction

## Overview

This project demonstrates how Retrieval-Augmented Generation (RAG) systems can be improved through the integration of **semantic metadata extraction** and a simplified **ontological structuring of text chunks**.

The main objective is to validate the hypothesis that:

> RAG optimization through Semantic Metadata Extraction and Ontological Structuring reduces hallucinations and improves answer precision compared to a standard RAG pipeline.

The project provides a direct comparison between:
- A **Standard RAG (Baseline)** system  
- A **Semantic RAG** system enhanced with metadata and consensus filtering  

---

## System Architecture

The system is organized into three main stages:

### 1. Ingestion & Enrichment (Data Lake → Search Index)

- Text documents are loaded from Azure Data Lake Storage.
- An LLM agent (`GPT-4.1`) processes each text fragment and extracts structured semantic metadata:
  - Topics
  - Characters
  - Emotions
  - Summary
- Each enriched chunk is indexed into Azure AI Search using:
  - Vector search  
  - Keyword search  
  - Metadata fields  
- Text embeddings are generated using `text-embedding-3-small`.

This step transforms raw text into semantically structured searchable data.

---

### 2. Standard RAG (Baseline)

- Performs hybrid similarity search (vector + keyword).
- Retrieves relevant chunks based purely on similarity.
- Susceptible to contextual drift and hallucinations when:
  - Concepts overlap across documents  
  - Similar embeddings belong to different narrative contexts  

This serves as the baseline for evaluation.

---

### 3. Semantic RAG (Proposed Method)

The proposed system enhances retrieval using **Consensus Filtering (Semantic Cohesion Algorithm)**.

Key improvements:
- Compares metadata across retrieved chunks.
- Filters out documents that do not align with the dominant context. 
- Provides enriched context (including metadata) to the LLM.

This method:
- Reduces cross-document contamination  
- Improves contextual alignment  
- Decreases hallucination rates  
- Increases response precision  

---

## Technologies Used

- Python 3.10+
- Azure OpenAI Service (GPT-4.1, text-embedding-3-small)
- Azure AI Search
- Azure Data Lake Storage
- Hybrid Retrieval
- Metadata-based Filtering

---

## Installation & Setup & RUNNING THE CODE

### Prerequisites

- Python 3.10 or higher
- An Azure account with access to:
  - Azure OpenAI Service
  - Azure AI Search
  - Azure Storage Account

---

### Clone the Repository

```bash
git clone <repository-url>
cd Semantic-RAG
```

--- 
### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Configure Environment Variables
Create a `.env` file in the project root with the following content:
```env
#The Brain
AZURE_OPENAI_ENDPOINT=your-azure-openai-endpoint
AZURE_OPENAI_KEY=your-azure-openai-key
AZURE_OPENAI_DEPLOYMENT_NAME=your-azure-openai-deployment-name##gpt-4.1  
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your-azure-openai-embedding-deployment-name##text-embedding-3-small 

# The Memory
AZURE_SEARCH_ENDPOINT=your-azure-search-endpoint
AZURE_SEARCH_KEY=your-azure-search-key
AZURE_SEARCH_INDEX_NAME=your-azure-search-index-name


# The Source Documents
AZURE_STORAGE_CONNECTION_STRING=your-azure-storage-connection-string
AZURE_STORAGE_CONTAINER_NAME=your-azure-storage-container-name
```

---

### RUNNING THE CODE
```bash
python ./chat_without_metadate.py ## Standard RAG (Baseline)
python ./chat_with_metadate.py ## SEMANTIC RAG (Proposed Method)
```
