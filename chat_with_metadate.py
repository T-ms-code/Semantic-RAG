import os
from dotenv import load_dotenv
from collections import defaultdict

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch

#Intrucat in interfata din Azure Cloud din Playground nu pot folosi index-ul creat anterior
#(caci e considerat
#extern/cu metadate), dar si pentru ca nu voi alege la Search Service ceva mai mult de Free,
#sunt obligat sa
#fac un nou script in care sa implementez manual RAG-ul folosind metadate.

#Incarc variabilele din .env
load_dotenv()

#Configurare Variabile de Mediu
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
GPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")       
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") 

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")



def main():
    # ---------------------------------------------------------
    #1)INITIALIZARE CLIENTI (Azure OpenAI & Azure Search)
    # ---------------------------------------------------------
    llm = AzureChatOpenAI(
        azure_deployment=GPT_DEPLOYMENT,
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_KEY,
        openai_api_version="2024-12-01-preview",
        temperature=0 #temperatura 0 pentru raspunsuri precise
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=EMBED_DEPLOYMENT,
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_KEY,
        openai_api_version="2024-12-01-preview"
    )

    vector_store = AzureSearch(
        azure_search_endpoint=SEARCH_ENDPOINT,
        azure_search_key=SEARCH_KEY,
        index_name=INDEX_NAME,
        embedding_function=embeddings.embed_query
    )


    # ---------------------------------------------------------
    #2)PRELUARE INTREBARE
    # ---------------------------------------------------------
    query = input("\nQuestion: ")


    # ---------------------------------------------------------
    #3)HYBRID SEARCH (Retrieval Initial)
    # ---------------------------------------------------------
    #Aducem mai multe documente (k=10) pentru a avea de unde filtra
    print("...Searching index (Hybrid Search)...")
    
    retrieved_docs = vector_store.similarity_search(
        query, 
        k=15, 
        search_type="hybrid" 
    )#hybrid = Combină intelesul (Vectori) cu potrivirea exacta de cuvinte


    # ---------------------------------------------------------
    #4)POST-RETRIEVAL FILTERING (Calcul Scor Coeziune)
    # ---------------------------------------------------------
    #Aici folosim metadatele pentru a elimina "zgomotul".
    #Daca un document are personaje/subiecte comune cu celelalte,
    #primeste un scor mai mare. Practic, asta e elementul de noutate pe langa RAG:
    #pe langa similitudine matematica se vine si cu una (o legatra) de sens data de 
    #ontologie=schema metadatelor.
    
    #A)Numaram frecventa metadatelor in rezultatele brute
    topic_counts = defaultdict(int)
    char_counts = defaultdict(int)
    emotion_counts = defaultdict(int)

    for doc in retrieved_docs:
        for t in doc.metadata.get("topics", []): topic_counts[t] += 1
        for c in doc.metadata.get("characters", []): char_counts[c] += 1
        for e in doc.metadata.get("emotions", []): emotion_counts[e] += 1

    #B)Acordam puncte fiecarui document
    scored_docs = []
    
    for doc in retrieved_docs:
        score = 0

        #+2 puncte pentru topicuri comune
        doc_topics = doc.metadata.get("topics", [])
        for t in doc_topics:
            if topic_counts[t] >= 2: 
                score += 2

        #+3 puncte pentru personaje comune
        doc_chars = doc.metadata.get("characters", [])
        for c in doc_chars:
            if char_counts[c] >= 2:
                score += 3

        #+1 punct pentru emotii comune (dominante)
        doc_emotions = doc.metadata.get("emotions", [])
        for e in doc_emotions:
            if emotion_counts[e] >= 3:
                score += 1

        scored_docs.append((doc, score))

    #C)Sortare si Selectie Finala
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    selected_docs = []

    for doc, score in scored_docs[:3]:#Le iau doar pe primele 3
            selected_docs.append(doc)
            

    # ---------------------------------------------------------
    #5)CREARE CONTEXT IMBOGATIT (Enriched Context)
    # ---------------------------------------------------------
    # Construim prompt-ul folosind rezumatele si metadatele
    context_blocks = []
    for doc in selected_docs:
        meta = doc.metadata
        block = (
            f"--- FRAGMENT ---\n"
            f"SUMMARY: {meta.get('summary')}\n"
            f"KEY CHARACTERS: {', '.join(meta.get('characters', []))}\n"
            f"PREDOMINANT EMOTIONS: {', '.join(meta.get('emotions', []))}\n"
            f"PREDOMINANT TOPICS: {', '.join(meta.get('topics', []))}\n"
            f"CONTENT:\n{doc.page_content}\n"
        )
        context_blocks.append(block)

    context = "\n".join(context_blocks)
    

    # ---------------------------------------------------------
    #6)GENERARE RASPUNS CU LLM
    # ---------------------------------------------------------
    print("\n...Generating answer...\n")
    
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the context below.\n"
        "Pay attention to the metadata to understand the context.\n"
        "If the answer is not fully supported by the context, state that clearly.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}"
    )

    answer = llm.invoke(prompt)

    print("\n=== ANSWER ===\n")
    print(answer.content)

if __name__ == "__main__":
    main()