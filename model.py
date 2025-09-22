from ollama import Client
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
import re

load_dotenv()

searchUrl = os.getenv("SEARCH_URL", "http://localhost:8080/")
ollamaUrl = os.getenv("OLLAMA_URL", "http://localhost:11434")
ollamaModel = os.getenv("OLLAMA_MODEL", "llama3")

def safe_get(url, timeout=10):
    try:
        return requests.get(url, timeout=timeout).text
    except requests.exceptions.Timeout:
        return False
    except requests.exceptions.RequestException as e:
        return False
    
def webSearch(query):
  params = {
    'q': query,
    "format": "json",
    "language": "en",
  }

  data = requests.get(searchUrl, params=params, timeout=10).json()

  urls = [result.get('url') for result in data['results']]
  return urls

def getTexts(query, numResults=2):
  urls = webSearch(query)
  texts = []
  headers = {
      "User-Agent": "Mozilla/5.0"
  }
  for url in urls:
    try:
        url = requests.get(url, headers=headers,timeout=10)
        soup = BeautifulSoup(url.text, 'lxml')

        for tag in soup.find_all(['script', 'style', 'head', 'img', 'svg', 'a', 'form', 'link', 'iframe']):
          tag.extract()

        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())
        texts.append(text)
    except requests.exceptions.Timeout:
        pass
    except requests.exceptions.RequestException as e:
        pass
    if len(texts) == numResults:
        break
    
  texts = "\n".join(texts)
  return texts

client = Client(
  host=ollamaUrl,
  headers={'x-some-header': 'some-value'}
)

def answerQuery(userMessage, contextMessages=[]):
  systemPrompt = """
    You are agent Kowalski. You behave like Kowalski from the penguins of Madagascar but you are aware that you are an AI.
    Talk like kowalski and answer questions (including simple math) or analyse messages.

    The person messaging you is your leader (like Skipper, but address them as 'sir').
    No roleplay actions.

    You will be given some context (chat messages) and the user's request which you need to respond to.
    If the user says "Kowalski analysis" they're asking you to "explain this in more detail" or "elaborate on this".

    When deciding how to answer, first check if the question can be answered confidently with general, timeless knowledge. If so, use your own knowledge.
    If the question involves current events, updated information, or details you're uncertain about, perform a web search before answering.
    In ambiguous cases, prefer searching if the information may have changed since training or if accuracy is critical

    If you want to search, come up with a search query you would use to search the web, then come up with a revised user request which is more specific (exclude yourself from this request) but includes the key details from the original request, finally ONLY respond with the following:
    request: <revised user request>
    search: <search query>
  """

  messagePrompt = """
    {contextMessages}

    The above is some extra messages from the chat that you can optionally use (they may not be relevant). Here is the user's request:

    "{userMessage}"
  """

  messagePrompt = messagePrompt.format(contextMessages="\n".join(contextMessages), userMessage=userMessage)

  searchPrompt = """
    You are agent Kowalski. You behave like Kowalski from the penguins of Madagascar but you are aware that you are an AI.
    Talk like kowalski and answer questions (including simple math) or analyse messages.

    The person messaging you is your leader (like Skipper, but address them as 'sir').
    No roleplay actions.

    If the user says "Kowalski analysis" they're asking you to "explain this in more detail" or "elaborate on this". It's not a command, just another regular request.
    You will be given all of the information from your search results (used to fulfil the user's request), the query used to get the information, and the user's request itself. Use all of this to respond to the user's request.
  """


  searchMessagePrompt = """
    {texts}

    The above is the information from your search results.

    your search query used to get the texts: "{searchQuery}"
    Your leader's request/message: "{userRequest}"
  """
  response = client.chat(
      model='mistral-small3.2',
      options={'temperature': 0},
      messages=[
        {'role': 'system', 'content': systemPrompt},
        {'role': 'user', 'content': messagePrompt},
      ]
  )
  response = response['message']['content']
  print(response)
  request = re.search(r"request:(.*)", response)
  if request:
    request = request.group(1)
    searchQuery = re.search(r"search:(.*)", response).group(1)
    texts = getTexts(searchQuery, numResults=2)
    response = client.chat(
      model='gemma3:12b', 
      options={'temperature': 0.1}, 
      messages=[
        {'role': 'system', 'content': searchPrompt},
        {'role': 'user', 'content': searchMessagePrompt.format(userRequest=request, searchQuery=searchQuery, texts=texts)}
      ]
    )
    response = response['message']['content']

  return(response)



print(answerQuery(userMessage="kowalski what's the plot of the film that these guys are talking about?", contextMessages=["bart: Yo guys ever watch that marmaduke movie, with the dog?", "gar: No", "bart: ok, I watched it, it was good", "gar: oh nice!", "bart: ok, I understand"]))