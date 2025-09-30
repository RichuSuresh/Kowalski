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
The user is your leader so they should be addressed as 'sir'.
No roleplay actions.
"""

messageTemplate = """
If the user is asking for your analysis they're asking you to "explain this in more detail" or "elaborate on this" where "this" is the message before the user's latest message.

When deciding how to respond:
- In ambiguous cases, prefer searching since accuracy is critical.
- If you don't know the answer, search for it.
- If the message requires updated (latest) info, current events, or facts you're uncertain about, perform a web search before answering.
- If you decide to search, your response should say that you're searching up the information on the web

Here is the user's latest message: {userMessage}

Sometimes the user might be saying casual banter, you can respond back with a whitty or sarcastic response
Keep your responses concise with easy to understand words. No extra fluff
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


async def answerQuery(userMessage, chatHistory=[], discordClient=None):
    messageContent = userMessage.content
    response = await chatChain.ainvoke({
        "chatHistory": chatHistory,
        "userMessage": messageContent,
    })
    response = json.loads(response.content)
    if response["search"] and response["request"]:
        await userMessage.channel.send(response["response"])
        texts = await getTexts(query=response["search"], request=response["request"], numResults=int(os.getenv("SEARCH_RESULTS_LIMIT")))
        searchResponse = await searchChain.ainvoke({
            "texts": texts,
            "searchQuery": response["search"],
            "request": response["request"],
        })
        
        searchResponse = json.loads(searchResponse.content)
        response["response"] = searchResponse["response"]
    return response["response"]