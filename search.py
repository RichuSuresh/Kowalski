import asyncio
import aiohttp
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

BLOCKLIST_PATTERNS = [
    "enable javascript", 
    "redirecting", 
    "403 forbidden",
    "access denied", 
]

ollamaSearchUrl = os.getenv("OLLAMA_SEARCH_URL", "http://localhost:11435")
embeddings = OllamaEmbeddings(
    model="embeddinggemma",
    base_url=ollamaSearchUrl,
    keep_alive=-1
)
vector_store = Chroma(
    collection_name="texts",
    embedding_function=embeddings,
)

async def fetchAndCleanText(url, session):
    try:
        async with session.get(url, timeout=10) as resp:
            response = await resp.text()

        soup = BeautifulSoup(response, 'html.parser')

        for tag in soup.find_all(['script', 'style', 'head', 'img', 'svg', 'a', 'form', 'link', 'iframe']):
            tag.extract()

        text = soup.get_text(separator=" ", strip=True)
        textClean = " ".join(text.split()).lower()
        if any(pat in textClean for pat in BLOCKLIST_PATTERNS) or len(textClean) < 200:
            return None
        
        text = Document(
            page_content=textClean,
            metadata={"source": url},
        )
        
        return text
    except Exception as e:
        return None

async def getTexts(query, request, numResults=2):
    params = {
        'q': query,
        "format": "json",
        "language": "en",
    }
    data = requests.get(os.getenv("SEARCH_URL", "http://localhost:8080/"), params=params, timeout=10).json()

    urls = [result.get('url') for result in data['results'] if 'youtube' not in result.get('url', '').lower()]

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    tasks = []
    async with aiohttp.ClientSession(headers=headers) as session:
        async with asyncio.TaskGroup() as tg:
            for url in urls:
                tasks.append(tg.create_task(fetchAndCleanText(url, session)))
    
    documents = [task.result() for task in tasks if task.result() is not None]
    await vector_store.aadd_documents(documents)
    retriever = vector_store.as_retriever(search_kwargs={"k": numResults})
    texts = await retriever.ainvoke(request)

    return texts
