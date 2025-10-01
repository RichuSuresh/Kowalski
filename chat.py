import asyncio
import base64
import json
import aiohttp
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
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
If the user asks to analyse an image. You will be given the image as a base64 encoded string.

Sometimes the user might be saying casual banter, you can respond back with a whitty or sarcastic response
Keep your responses concise with easy to understand words. No extra fluff
Your response should be in JSON format as follows (including when analysing images):
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

DO NOT GREET THE USER

Use ONLY the texts from your search to answer the user's request. Do not include document references in your response. Keep your response short and concise, no more than 3 sentences.
Your response should be in JSON format as follows:
{{
    "response": "<your answer, up to 4000 characters>"
}}
"""

chatPrompt = ChatPromptTemplate.from_messages(
    messages=[
        SystemMessage(content=systemTemplate),
        MessagesPlaceholder(variable_name="chatHistory"),
        ("human", messageTemplate),
        MessagesPlaceholder(variable_name="images"),
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

async def fetch_image_base64(session, url):
    async with session.get(url) as resp:
        data = await resp.read()
    return {
        "type": "image", "source_type": "base64", "data": base64.b64encode(data).decode("utf-8"), "mime_type": "image/jpeg"
    }

async def answerQuery(userMessage, chatHistory=[]):
    images = []
    if userMessage.attachments:
        tasks = []
        async with aiohttp.ClientSession() as session:
            for attachment in userMessage.attachments:
                tasks.append(fetch_image_base64(session, attachment.url))
            images = await asyncio.gather(*tasks)

    imageContent = [HumanMessage(
        content=[
            {"type": "text", "text": "describe this image"},
            *images
        ]
    )] if len(images) > 0 else []

    response = await chatChain.ainvoke({
        "chatHistory": chatHistory,
        "images": imageContent,
        "userMessage": userMessage.content,
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

# print(asyncio.run(answerQuery("Hey kowalski, what does this article talk about? https://www.reddit.com/r/LocalLLaMA/comments/1jdasng/heads_up_if_youre_using_gemma_3_vision/")))