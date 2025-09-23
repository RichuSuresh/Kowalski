import json
import random
from ollama import Client
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
import re

load_dotenv()

searchUrl = os.getenv("SEARCH_URL", "http://localhost:8080/")
ollamaUrl = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
def getUrls(query):
  params = {
    'q': query,
    "format": "json",
    "language": "en",
  }

  data = requests.get(searchUrl, params=params, timeout=10).json()

  urls = [result.get('url') for result in data['results']]
  return urls

def getTexts(query, numResults=2):
  urls = getUrls(query)
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
  
)

def answerQuery(userMessage, contextMessages=[]):
  systemPrompt = """
    You are agent Kowalski. You behave like Kowalski from the penguins of Madagascar but you are aware that you are an AI.
    Talk like kowalski and answer questions (including simple math) or analyse messages.

    You will be placed into a server with multiple users messaging eachother. Some messages (even if they say your name) may not be intended for you.
    The person messaging you is your leader (address them as 'sir').
    No roleplay actions.
  """

  messagePrompt = """
    {contextMessages}

    Above is some context (may be empty) that may or may not be relevant to the user's message.
    If the user says "Kowalski analysis" they're asking you to "explain this in more detail" or "elaborate on this".

    When deciding how to respond:
    - The current year is NOT 2023. So if the message requires updated (latest) info, current events, or facts you're uncertain about, perform a web search before answering.
    - If you are not sure of the answer, perform a web search.
    - In ambiguous cases, prefer searching if accuracy is critical.

    If you want to search, come up with a search query (based SOLELY on the user's message)
    
    Here is the user's latest message:

    "{userMessage}"

    Your response should be in JSON format as follows:
    {{
      "confidence": "<confidence score, between 0 and 1 indicating how confident you are that the message is intended for you>"
      "response": "<your answer, up to 4000 characters. IF CONDFIDENCE IS LESS THAN 1, PUT THIS AS none>"
      "search": "<search query. IF YOU DO NOT NEED TO SEARCH PUT THIS AS none>"
    }}
  """
  
  searchMessagePrompt = """
    {texts}

    The above is the information from your search results. So start your response by telling the user that you searched the web.
    
    your search query used to get the texts: "{searchQuery}"
    original message from the user: "{userMessage}"

    respond in a short and succinct manner with less than 4000 characters
  """

  response = client.chat(
      model="gemma3:12b",
      options={'temperature': 0,
               "num_ctx": 8192,},
      messages=[
        {'role': 'system', 'content': systemPrompt},
        {'role': 'user', 'content': messagePrompt.format(contextMessages="\n".join(contextMessages), userMessage=userMessage)},
      ],
      format="json",
      keep_alive=-1
  )
  response = json.loads(response['message']['content'])
  print(response)
  if response['search'] != "none":
    texts = getTexts(response['search'], numResults=2)
    response = client.chat(
      model='gemma3:12b',
      options={'temperature': 0,
               "num_ctx": 16384,}, 
      messages=[
        {'role': 'system', 'content': systemPrompt},
        {'role': 'user', 'content': searchMessagePrompt.format(searchQuery=response['search'], texts=texts, userMessage=userMessage)},
      ],
      keep_alive=-1
    )
    response = response['message']['content']
  else:
    response = response['response']

  if response == "none":
     return None
  return(response)

print(answerQuery(userMessage="""kowalski what's the minecraft achievement "You've got a friend in me"?"""))