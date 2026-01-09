import os
from dotenv import load_dotenv
from typing import List, Optional

#Biblioteci pentru Structura Date
from pydantic import BaseModel, Field

#Biblioteci Azure & LangChain
from azure.storage.blob import BlobServiceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import AzureSearch
from langchain_core.documents import Document

#Incarc variabilele din .env
load_dotenv()

#OpenAI (Creierul & Embeddings)
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
GPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")       
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") 

#Azure Search (Memoria)
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

#Data Lake (Stocarea)
STORAGE_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME") 



#DEFINIREA ONTOLOGIEI/METADATELE PENTRU ANALIZA SEMNATICA
#Asta defineate structura pe care GPT-4.1 trebuie sa o respecte
class SemanticMetadata(BaseModel):
    """Semantic metadata extracted from the text chunk for RAG optimization."""
    
    # Pot fi fragmente de unde nu se pot extrage metadatele!
    summary: Optional[str] = Field(
        default=None,
        description="A concise summary. If the text is meaningless (e.g. only numbers, headers), return None."
    )
    
    characters: List[str] = Field(
        default_factory=list, # Daca nu gaseste, pune lista goala []
        description="List of characters. Return empty list if none found."
    )
    
    emotions: List[str] = Field(
        default_factory=list,
        description="Predominant emotions. Return empty list if text is technical or neutral."
    )
    
    topics: List[str] = Field(
        default_factory=list,
        description="Key concepts. Return empty list if no clear topics."
    )



def main():
    print("START: Data Lake -> GPT -> Search")
    # Modelul de Chat (pentru extragere metadate)
    llm = AzureChatOpenAI(
        azure_deployment=GPT_DEPLOYMENT,
        openai_api_version="2024-12-01-preview",
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_KEY,
        temperature=0 # Vrem consistenta, nu creativitate/creativitate=0, strict ,matematic
    )

    #Creez un model care sa raspunda conform ontologiei 
    structured_llm = llm.with_structured_output(SemanticMetadata)

    #Modelul de Embedding (pentru vectori)
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=EMBED_DEPLOYMENT,
        openai_api_version="2024-12-01-preview",
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_KEY,
        max_retries=1,     #ca sa nu ramana blocat
        request_timeout=10
    )


    #Clientul de Stocare (Data Lake)
    try:
        blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONN_STR)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        print(f"Connected to Storage: {CONTAINER_NAME}")
    except Exception as e:
        print(f"Error connecting to Storage: {e}")
        return

    #Descarcarea fisierelor din Cloud si procesarea lor locala pentru crearea indexului
    raw_documents = []
    blobs = container_client.list_blobs()
    
    found_blobs = False
    for blob in blobs:
        found_blobs = True
        print(f"Downloading: {blob.name}...")
        try:
            blob_client = container_client.get_blob_client(blob)
            downloader = blob_client.download_blob()
            text_content = downloader.readall().decode('utf-8')
            
            doc = Document(page_content=text_content, metadata={"source": blob.name})
            raw_documents.append(doc)
        except Exception as e:
            print(f"Error reading {blob.name}: {e}")

    if not found_blobs:
        print("No files found in Data Lake.")
        return
    

    #Spargerea Textului (Chunking) in fragmente mai mici, mai ales pentru textele foarte mari
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    #Textul este impartit in bucati (chunks) de: 1000 caractere fiecare cu 200 caractere comune intre bucai consecutive
    #ca sa existe totusi un context comun
    chunks = text_splitter.split_documents(raw_documents)
    print(f"   -> {len(chunks)} chunks created.")

    #EXTRAGEREA SEMANTICA
    print("LLM is analyzing chunks...")
    
    enriched_docs = []
    
    #Procesam toate bucatile
    for i, chunk in enumerate(chunks):
        print(f"   LLM is analyzing chunk{i+1}/{len(chunks)}...", end="\r")
        
        try:
            #Invocam modelul
            metadata_res = structured_llm.invoke(
                "Analyze the following text at a high level and extract neutral semantic metadata. "
                "Do not include graphic or explicit details. Follow the schema strictly.\n\n"
                + chunk.page_content
            )

            new_metadata = chunk.metadata.copy()
            #Daca rezumatul e None (AI-ul a zis ca textul e inutil), pun un text standard
            if metadata_res.summary is None:
                new_metadata["summary"] = "Technical content or lacks semantic context."
            else:
                new_metadata["summary"] = metadata_res.summary
            
            #Listele sunt deja [] daca sunt goale (datorita default_factory), deci e safe
            new_metadata["characters"] = metadata_res.characters
            new_metadata["emotions"] = metadata_res.emotions
            new_metadata["topics"] = metadata_res.topics
            
            new_doc = Document(page_content=chunk.page_content, metadata=new_metadata)
            enriched_docs.append(new_doc)
            
        except Exception as e:
            #Daca crapa agentul de tot, punem valori default ca sa nu oprim tot procesul
            print(f"\n   Error chunk {i}: {e}. Using defaults.")
            fallback_meta = chunk.metadata.copy()
            fallback_meta["summary"] = "Error processing metadata."
            fallback_meta["characters"] = []
            fallback_meta["emotions"] = []
            fallback_meta["topics"] = []
            
            enriched_docs.append(Document(page_content=chunk.page_content, metadata=fallback_meta))

    print("\n Extraction complete.")


    #Indexarea
    print(f" Indexing {len(enriched_docs)} documents...")
    vector_store = AzureSearch(
        azure_search_endpoint=SEARCH_ENDPOINT,
        azure_search_key=SEARCH_KEY,
        index_name=INDEX_NAME,
        embedding_function=embeddings.embed_query
    )
    
    #Adaugam documentele (asta creeaza indexul automat daca nu exista)
    # Trimit documentele cate 50, nu toate odata, ca s-a blocat inainte...
    BATCH_SIZE = 10
    total_docs = len(enriched_docs)
    
    print(f" Starting batched upload ({BATCH_SIZE} docs per request)...")
    
    for i in range(0, total_docs, BATCH_SIZE):
        batch = enriched_docs[i : i + BATCH_SIZE]
        print(f"    Uploading batch {i} - {i + len(batch)}...", end="\r")
        
        try:
            vector_store.add_documents(documents=batch)
            print(f"   Batch {i} - {i + len(batch)} uploaded successfully.   ")
        except Exception as e:
            print(f"\n    Error uploading batch starting at {i}: {e}")
    print(f" SUCCESS! Pipeline finished.")



if __name__ == "__main__":
    main()