from ollama import Client
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup

load_dotenv()

searchUrl = os.getenv("SEARCH_URL", "http://localhost:8080/")
# searchUrl = searchUrl + 'search'
ollamaUrl = os.getenv("OLLAMA_URL", "http://localhost:11434")
ollamaModel = os.getenv("OLLAMA_MODEL", "llama3")

def safe_get(url, timeout=10):
    try:
        return requests.get(url, timeout=timeout).text
    except requests.exceptions.Timeout:
        return False
    except requests.exceptions.RequestException as e:
        return False
    
def webSearch(query, numResults=5):
  params = {
    'q': query,
    "format": "json",
    "language": "en",
  }

  data = requests.get(searchUrl, params=params, timeout=10).json()

  urls = [result.get('url') for result in data['results']]
  safeUrls = []
  for url in urls:
    isSafe = safe_get(url)
    if isSafe:
      safeUrls.append(url)
    if len(safeUrls) == numResults:
      break
  return safeUrls

def getCleanedText(urls):
  texts = []
  for url in urls:
    url = requests.get(url, timeout=10).text
    soup = BeautifulSoup(url, 'lxml')

    for tag in soup.find_all(['script', 'style', 'head', 'img', 'svg', 'a', 'form', 'link', 'iframe']):
      tag.extract()

    text = soup.get_text(separator=" ", strip=True)
    text = " ".join(text.split())
    texts.append(text)
  texts = "\n".join(texts)
  return texts

client = Client(
  host=ollamaUrl,
  headers={'x-some-header': 'some-value'}
)

system_prompt = """
You are agent Kowalski. You behave like Kowalski from the penguins of Madagascar.
You are a chat assistant and can answer questions (including simple math) or analyse messages.

Pretend the person messaging you is your leader (like Skipper, but address them as 'sir'). 
Only talk like Kowalski. Do not include actions.
Keep answers succinct and only include necessary information.

Before responding to the user's request, decide if you have enough information to give solid insight. 
If you don't have enough information don't say anything else and just respond with the text "search: <query>"
which signifies you need to search the web for further details. 
"""

userMessage = """
Kowalski what team does messi play for now?
"""

response = client.chat(
    model='mistral-small3.2',
    options={'temperature': 0},
    messages=[
      {'role': 'system', 'content': system_prompt},
      {'role': 'user', 'content': userMessage}
    ]
)
response = response['message']['content']

if response[0:6] == "search":
  texts = getCleanedText(webSearch(response[8:], 2))
  system_prompt = """
  You are agent Kowalski. You behave like Kowalski from the penguins of Madagascar.
  You are a chat assistant and can answer questions (including simple math) or analyse messages.

  Pretend the person messaging you is your leader (like Skipper, but address them as 'sir'). 
  Only talk like Kowalski. Do not include actions.
  Keep answers succinct and only include necessary information.

  In a previous conversation which you don't need to know about, you didn't hav enough information so you searched the web so respond to the user with that in mind. You will
  be given all of the information you need to answer the user's request. Never guess and use the information provided to respond to the user.
  """

  messagePrompt = """
  {texts}

  The above is the information from your search results. Here is the user's original message:

  "{userMessage}"
  """
  response = client.chat(
    model='gemma3:4b', 
    options={'temperature': 0.1}, 
    messages=[
      {'role': 'system', 'content': system_prompt},
      {'role': 'user', 'content': messagePrompt.format(userMessage=userMessage, texts=texts)}
    ]
  )
  response = response['message']['content']

print(response)