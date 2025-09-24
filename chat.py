import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import datetime
from dotenv import load_dotenv
import os
from search import getRelevantTexts

load_dotenv()
ollamaUrl = os.getenv("OLLAMA_URL", "http://localhost:11434")

model = ChatOllama(
    model="gemma3:12b",
    base_url=ollamaUrl,
    temperature=0,
    format="json",
    keep_alive=-1,
)

systemTemplate = """
You are agent Kowalski. You talk like Kowalski from the Penguins of Madagascar series, but you are aware that you are an AI.
You are an AI chatbot
You will be placed into a server with multiple users messaging eachother. Some messages (even if they say your name) may not be intended for you.
The user is your leader (address them as 'sir').
No roleplay actions.
"""

messageTemplate = """
If the user is asking for your analysis they're asking you to "explain this in more detail" or "elaborate on this" where "this" is the latest message in context.

When deciding how to respond:
- Accuracy is critical, favour a web search more than a guess.
- If the message requires updated (latest) info, current events, or facts you're uncertain about, perform a web search before answering.
- If you are not sure of the answer, perform a web search.
- In ambiguous cases, prefer searching if accuracy is critical.

Here are previous messages in the chat: {chatHistory}

Here is the user's latest message: {userMessage}

Your response should be in JSON format as follows:
{{
    "response": "<your answer, up to 4000 characters>"
    "request": "<(optional) original query from the user but revised to be a more specific question>"
    "search": "<(optional) search query>"
}}
"""

searchTemplate = """
After performing a web search, you found the results to respond to the user properly.

The texts from your search: {texts}
your search query used to get the texts: {searchQuery}
The request you need to answer: {request}

Use ONLY the texts from your search to answer the user's request. Do not include document references in your response
Your response should be in JSON format as follows:
{{
    "response": "<your answer, up to 4000 characters>"
}}
"""

chatPrompt = ChatPromptTemplate.from_messages(
    [
        ("system", systemTemplate),
        ("human", messageTemplate),
    ]
)
chatChain = chatPrompt | model

searchPrompt = ChatPromptTemplate.from_messages(
    [
        ("system", systemTemplate),
        ("human", searchTemplate),
    ]
)
searchChain = searchPrompt | model


def answerQuery(userMessage):
    response = chatChain.invoke({
        "chatHistory": [],
        "userMessage": userMessage,
    })
    response = json.loads(response.content)

    if "search" in response and "request" in response:
        texts = getRelevantTexts(query=response["search"], request=response["request"], numResults=int(os.getenv("SEARCH_RESULTS_LIMIT")))
        searchResponse = searchChain.invoke({
            "texts": texts,
            "searchQuery": response["search"],
            "request": response["request"],
        }).content
        
        searchResponse = json.loads(searchResponse)
        response["response"] = searchResponse["response"]
    return response["response"]

print(answerQuery("Kowalski what is the minecraft advancement You've got a friend in me and how do I unlock it?"))


