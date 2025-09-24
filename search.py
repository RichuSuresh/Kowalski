from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

BLOCKLIST_PATTERNS = [
    "enable javascript", 
    "redirecting", 
    "403 forbidden",
    "access denied", 
]

def getUrls(query):
    params = {
        'q': query,
        "format": "json",
        "language": "en",
    }
    data = requests.get(os.getenv("SEARCH_URL", "http://localhost:8080/"), params=params, timeout=10).json()

    urls = [result.get('url') for result in data['results'] if 'youtube' not in result.get('url', '').lower()]
    return urls

def getTextDocuments(query, numResults):
    urls = getUrls(query)
    print(len(urls))
    texts = []
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    for i, url in enumerate(urls):
        try:
            response = requests.get(url, headers=headers,timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            for tag in soup.find_all(['script', 'style', 'head', 'img', 'svg', 'a', 'form', 'link', 'iframe']):
                tag.extract()

            text = soup.get_text(separator=" ", strip=True)
            text_clean = " ".join(text.split()).lower()
            if any(pat in text_clean for pat in BLOCKLIST_PATTERNS) or len(text_clean) < 200:
                continue

            text = Document(
                page_content=text_clean,
                metadata={"source": url},
                id=str(i)
            )
            texts.append(text)
        except requests.exceptions.Timeout:
            pass
        except requests.exceptions.RequestException as e:
            pass
        if len(texts) == numResults:
            break

    return texts

def getRelevantTexts(query, request, numResults=2):
    embeddings = OllamaEmbeddings(
        model="embeddinggemma",
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434")
    )
    vector_store = Chroma(
        collection_name="texts",
        embedding_function=embeddings,
    )
    texts = getTextDocuments(query, numResults=numResults)
    vector_store.add_documents(texts)
    retriever = vector_store.as_retriever(search_kwargs={"k": numResults})
    documents = retriever.invoke(request)

    ## The code below was used to summarise individual web pages so they could fit within the context window of the AI
    ollamaUrl = os.getenv("OLLAMA_URL", "http://localhost:11434")

    model = ChatOllama(
        model="gemma3:1b",
        base_url=ollamaUrl,
        temperature=0,
    )

    for document in documents:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers queries or finds information about the following text: {text}"),
            ("human", "Use the information to answer the following question: {request}"),
        ])
        chain = prompt | model

        response = chain.invoke({
            "text": document.page_content,
            "request": request
        })
        document.page_content = response.content
    print(documents)
    return documents

