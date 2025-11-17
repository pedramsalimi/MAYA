import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI

# Temporary shim for chromadb>=1.1 which removed IncludeEnum expected by graph retriever
from enum import Enum
import chromadb.api.types as chroma_types
if not hasattr(chroma_types, "IncludeEnum"):
    class IncludeEnum(str, Enum):
        documents = "documents"
        metadatas = "metadatas"
        embeddings = "embeddings"
        distances = "distances"
        uris = "uris"
    chroma_types.IncludeEnum = IncludeEnum

from graph_retriever.strategies import Eager          
from langchain_graph_retriever import GraphRetriever


# ---------- config ----------
# load env from repo root
load_dotenv(Path(__file__).resolve().parents[3] / ".env")
COLLECTION = "maya_health_graph_rag"

DATA_DIR = Path("/Users/p.salimi1/Documents/MAYA/maya/data/health_papers")  # put your PDFs / text there


# ---------- simple LLM-NER (no extra deps) ----------
llm_ner = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    temperature=0,
    api_version="2024-12-01-preview",
    azure_endpoint="https://mayaagent.openai.azure.com/",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)




def extract_entities(text: str) -> Dict[str, Any]:
    """Lightweight NER over your health themes using the LLM."""
    prompt = f"""
You are tagging health literature for a smart mirror for adolescent/young adult (AYA) cancer survivors.

From the text below, extract:

- clinical_topics: cardiotoxicity, cardioprotection, lifestyle, fertility, intimacy, employment, financial toxicity, resilience, peer support, digital monitoring, smart mirrors, etc.
- population: e.g. AYA survivors, childhood cancer survivors, parents, clinicians.
- domain_theme: one of
  ["Cardio-Oncology", "Psychological Wellbeing", "Personal Relationships & Life Planning",
   "Resilience & Everyday Adaptation", "Digital Monitoring & Smart Mirror"]

Return a JSON object with keys: clinical_topics (list[str]), population (list[str]), domain_theme (str).

Text:
\"\"\"{text[:2000]}\"\"\"
"""
    resp = llm_ner.invoke(prompt)
    try:
        import json

        return json.loads(resp.content)
    except Exception:
        return {
            "clinical_topics": [],
            "population": [],
            "domain_theme": "Unknown",
        }


# ---------- load + chunk ----------
def load_raw_docs() -> List[Document]:
    docs: List[Document] = []
    for p in DATA_DIR.glob("**/*"):
        if p.suffix.lower() in {".txt", ".md"}:
            text = p.read_text(encoding="utf-8")
            docs.append(Document(page_content=text, metadata={"source": str(p)}))
        elif p.suffix.lower() in {".pdf"}:
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(str(p))
            docs.extend(loader.load())
    return docs


def chunk_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )
    return splitter.split_documents(docs)


# ---------- enrich with graph metadata ----------
def tag_chunks(chunks: List[Document]) -> List[Document]:
    tagged: List[Document] = []
    for d in chunks:
        ents = extract_entities(d.page_content)
        md = dict(d.metadata)
        md.update({
            "clinical_topics": ", ".join(ents.get("clinical_topics", [])),
            "population": ", ".join(ents.get("population", [])),
            "domain_theme": ents.get("domain_theme", "Unknown"),
        })

        
        tagged.append(Document(page_content=d.page_content, metadata=md))
    return tagged


# ---------- store in Chroma ----------
def build_vector_store(tagged_docs: List[Document]) -> Chroma:
    embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-small",
            api_version="2024-12-01-preview",
            azure_endpoint="https://mayaagent.openai.azure.com/"
        )
    return Chroma.from_documents(
        documents=tagged_docs,
        embedding=embeddings,
        collection_name="health_rag_graph",
        persist_directory=".chroma-health",
    )


# ---------- create GraphRAG retriever ----------
def build_graph_retriever(store) -> GraphRetriever:
    return GraphRetriever(
        store=store,
        # Graph edges: this is where you say which metadata keys define connections
        edges=[
            ("domain_theme", "domain_theme"),
            ("clinical_topics", "clinical_topics"),
            ("population", "population"),
        ],
        # Traversal strategy: NO metadata_keys here
        strategy=Eager(
            select_k=8,   # how many nodes total to keep (k is deprecated alias)
            start_k=3,    # how many nodes from initial similarity search
            max_depth=2,  # how far to traverse
            # you can leave adjacent_k, max_traverse at defaults
        ),
    )

def main() -> None:
    raw_docs = load_raw_docs()
    chunks = chunk_docs(raw_docs)
    tagged = tag_chunks(chunks)
    store = build_vector_store(tagged)
    retriever = build_graph_retriever(store)

    # quick smoke test
    test_q = "How should we monitor cardiotoxicity and support AYA survivors' mental health?"
    docs = retriever.invoke(test_q)
    print(f"Retrieved {len(docs)} docs")
    if docs:
        print(docs[0].metadata)


if __name__ == "__main__":
    main()
