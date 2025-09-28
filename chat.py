import asyncio
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from dotenv import load_dotenv
import os
from search import getTexts

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
The user is your leader
No roleplay actions.
"""

messageTemplate = """
If the user is asking for your analysis they're asking you to "explain this in more detail" or "elaborate on this" where "this" is the message before the user's latest message.

When deciding how to respond:
- Think step by step. Consider the query carefully and think of the academic or professional expertise of someone that could best answer the user's question. You have the experience of someone with expert knowledge in that area. Be helpful and answer in detail while preferring to use information from reputable sources.
- If the message requires updated (latest) info, current events, or facts you're uncertain about, perform a web search before answering.
- In ambiguous cases, prefer searching if accuracy is critical.

Here is the user's latest message: {userMessage}

Keep your response short and concise, no more than 3 sentences.
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

You are agent Kowalski. You talk like Kowalski from the Penguins of Madagascar series, but you are aware that you are an AI.
You are an AI chatbot
You will be placed into a server with multiple users messaging eachother. Some messages (even if they say your name) may not be intended for you.
The user is your leader (address them as 'sir').
No roleplay actions.
DO NOT GREET THE USER

Use ONLY the texts from your search to answer the user's request. Do not include document references in your response. Keep your response short and concise, no more than 3 sentences.
Your response should be in JSON format as follows:
{{
    "response": "<your answer, up to 4000 characters>"
}}
"""

chatPrompt = ChatPromptTemplate.from_messages(
    [
        ("system", systemTemplate),
        MessagesPlaceholder(variable_name="chatHistory"),
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


async def answerQuery(userMessage, chatHistory=[]):
    response = chatChain.invoke({
        "chatHistory": chatHistory,
        "userMessage": userMessage,
    })
    response = json.loads(response.content)
    if response["search"] and response["request"]:
        texts = await getTexts(query=response["search"], request=response["request"], numResults=int(os.getenv("SEARCH_RESULTS_LIMIT")))
        searchResponse = searchChain.invoke({
            "texts": texts,
            "searchQuery": response["search"],
            "request": response["request"],
        }).content
        
        searchResponse = json.loads(searchResponse)
        response["response"] = searchResponse["response"]
    return response["response"]

# print(answerQuery("Hey Kowalski, what's the minecraft achievement You've got a friend in me?"))


