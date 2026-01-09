# Semantic RAG cu Extragere de Metadate (LangExtract)

Acest proiect demonstrează îmbunătățirea sistemelor RAG (Retrieval-Augmented Generation) prin utilizarea **metadatelor semantice** și a unei structuri ontologice simplificate.

Scopul principal este de a valida **Cerința 8: Optimizarea RAG prin extragerea de Metadate Semantice și Structurarea Ontologică a Fragmentelor**, arătând cum acest lucru reduce halucinațiile și îmbunătățește precizia răspunsurilor în comparație cu un RAG standard.

## Arhitectura

Sistemul este compus din trei etape principale:
1.  **Ingestie & îmbogățire (Data Lake -> Index):**
    * Textele sunt citite din Azure Data Lake Storage.
    * Un agent LLM (`GPT-4.1`) analizează fiecare fragment și extrage o ontologie: `Topics`, `Characters`, `Emotions`, `Summary`.
    * Datele îmbogățite sunt indexate în **Azure AI Search** (Vector + Keyword + Metadata) cu text-embedding-3-small.
2.  **RAG Standard (Baseline):**
    * Căutare hibridă simplă bazată pe similitudine.
    * Predispus la erori de context ("halucinații") când conceptele se suprapun între documente.
3.  **Semantic RAG (Metoda Propusă):**
    * Folosește un algoritm de **Consensus Filtering** (Coeziune Semantică).
    * Filtrează documentele care nu se aliniază cu contextul majoritar (Personaje/Topicuri).
    * Oferă LLM-ului un context îmbogățit cu rezumate și emoții.

## Instalare și Configurare

### Prerechizite
* Python 3.10+
* Cont Azure cu acces la: OpenAI Service, AI Search, Storage Account.

### Clonare și Dependințe
```bash
git clone <url-repo>
cd Semantic-RAG-LangExtract
pip install -r requirements.txt
python ./chat_without_metadate.py ## RAG Standard
python ./chat_with_metadate.py    ## Semantic RAG cu Metadate
```