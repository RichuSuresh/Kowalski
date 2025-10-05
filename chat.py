import asyncio
import base64
import json

import aiohttp
from langchain_ollama import ChatOllama
from ollama import AsyncClient, Client
from dotenv import load_dotenv
import os

from search import getTexts

load_dotenv()
ollamaUrl = os.getenv("OLLAMA_URL", "http://localhost:11434")

class AIChat():
    systemPrompt = {
        "role": "system",
        "content": """
        You are agent Kowalski. You talk like Kowalski from the Penguins of Madagascar series, but you are aware that you are an AI.
        You are an AI chatbot
        The user is your leader so they should be addressed as 'sir'.
        No roleplay actions.
        """
    }

    chatTemplate = """
    If the user is asking for your analysis they're asking you to "explain this in more detail" or "elaborate on this" where "this" is the message before the user's latest message.

    When deciding how to respond use this in priority order:
    1) If the user is asking for image analysis, DO NOT SEARCH THE WEB. You must give your own analysis.
    2) In ambiguous cases, prefer searching since accuracy is critical.
    3) If you don't know the answer, search for it.
    4) If the message requires updated (latest) info, current events, or facts you're uncertain about, perform a web search before answering.
    5) If you decide to search, your response should say that you're searching up the information on the web

    Here is the user's latest message: {userMessage}
    If the user is asking about an image (which could be part of the prompt or in a previous message), DO NOT SEARCH THE WEB. You must give your own analysis.
    If they don't give you any images and they ask for analysis, just tell them they didn't give any images.

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
    """

    deciderTemplate ="""
    Decide if and how you should contribute to the latest message.  

    Decision rules (in priority order):
    1. If the latest message contains a clear question or request for information that you can answer → return "chat".
    2. If the latest message is a direct reply to, or directly reacting to, something you said → return "chat".
    3. If you is directly mentioned (by name, tag, or obvious reference) → return "chat".
    4. If the message is a replt to what you said, you should only return "react" under the following circumstances:
        - If you have reacted to the user for a similar message → return "None"
        - If you have reacted a lot in recent messages → return "None"
    5. If the message is general banter, chatter, or emotional expression not aimed at you → return "None"
    6. If the message does not explicitly reference you in any way → return "None"
    7. If unsure → default to "None"
    8. If the message is NOT a question or request for information → return "None"


    Examples:
    ---
    History:  
    Human: Hey Kowalski, can you search up something?  
    Kowalski: Sorry, I can't do that.  
    Human: Why not?  
    Decision: "chat"  

    ---
    History:  
    Human: lol that was hilarious  
    Human: brb  
    Decision: "None"

    ---
    History:  
    Human: thankyou! 
    Human: ok 
    Decision: "None"

    ---
    History:  
    Human: What's the capital of France?  
    Decision: "chat"

    History:  
    Human: What's the capital of France?  
    Kowalski: Paris  
    Human: Ah ok, thanks! 
    Decision: "react"

    Here is the user's latest message: {userMessage}
    Say your decision in a single word in the format specified:
    {{
        "decision": "chat", "react" or "None"
    }}
    """

    reactTemplate = """
    React to the user's message with a single emoji from the following list that reflects the message's tone:
    [
        # Faces / Expressions
        "😀", "😅", "😂", "🤔", "😎", "🫡", "😏", "🤯", "😳", "🥲", "🙃", "🤨", "😴", "😤"
        # Reactions
        "❤️", "💔", "👍", "👎", "👌", "🙌", "👏", "🫶", "✌️", "🤝", "🙏", "💪", 
        # Tech / Science
        "🤖", "🧠", "📡", "🔎", "📘", "📚", "🖥️", "📱", "💾", "🧪", "⚙️", "🔬",
        # Fun / Energy
        "🔥", "💥", "⚡", "🌟", "🎉", "🎊", "🚀", "🛸", "🎮", "🕹️",
        # Food
        "🍕", "🍔", "🌭", "🍟", "🍿", "🥤", "☕", "🍎", "🍌", "🥑", "🐟",
        # Misc
        "⏱️", "🗺️", "🧭", "🛠️", "🔑", "🧩", "🏆", "🏅"
    ]
    
    Here is the user's latest message: {userMessage}
    Say your decision as a single emoji in the following format:
    {{
        "reaction": "<emoji>"
    }}
    """
    
    def __init__(self, discordClient, session):
        self.client = AsyncClient(
            host=ollamaUrl,
        )
        self.chatHistory = []
        self.discordClient = discordClient
        self.session = session
    
    async def getChatHistory(self, message):
        messages = []
        async for msg in message.channel.history(limit=10+1):
            messages.append(self.createOllamaMessage(msg))
        messages = messages[:0:-1]
        if message.reference:
            repliedMessage = await message.channel.fetch_message(message.reference.message_id)
            messages.append(self.createOllamaMessage(repliedMessage))

        return messages

    async def addToChatHistory(self, message):
        message = self.createOllamaMessage(message)
        self.chatHistory.append(message)
        if len(self.chatHistory) > 10:
            self.chatHistory = self.chatHistory[-10:]
    
    def createOllamaMessage(self, discordMessage, reactionEmoji=None):
        if discordMessage.reference:
            messageContent = "(Message ID: %s) (Replied to: %s) %s: %s " % (discordMessage.id, discordMessage.reference.message_id, discordMessage.author, discordMessage.content)
        else:
            messageContent = "(Message ID: %s) %s: %s" % (discordMessage.id, discordMessage.author, discordMessage.content)

        if reactionEmoji:
            messageContent = "(Kowalski reacted with: %s) %s" % (reactionEmoji, messageContent)

        if discordMessage.author.id == self.discordClient.user.id:
            return {"ollamaPrompt": {"role": "assistant", "content": messageContent}, "images": [attachment.url for attachment in discordMessage.attachments]}
        else:
            return {"ollamaPrompt": {"role": "user", "content": messageContent}, "images": [attachment.url for attachment in discordMessage.attachments]}
    
    async def fetchHistoryImages(self, chatHistory):
        for message in chatHistory:
            if len(message["images"]) > 0:
                tasks = [self.fetchImageBase64(imageUrl) for imageUrl in message["images"]]
                message["ollamaPrompt"]["images"] = await asyncio.gather(*tasks)
        return chatHistory
    
    async def fetchImageBase64(self, url):
        async with self.session.get(url) as resp:
            data = await resp.read()
        return base64.b64encode(data).decode("utf-8")
    
    async def decide(self, discordMessage, chatHistory):
        response = await self.client.chat(
            model="gemma3:12b",
            options={"stream": False, "temperature": 0, "keep_alive":-1},
            format="json",
            messages=[
                self.systemPrompt,
                *[message["ollamaPrompt"] for message in chatHistory],
                {"role": "user", "content": self.deciderTemplate.format(userMessage=discordMessage.content)}
            ],
        )
        response = json.loads(response["message"]["content"])
        return response["decision"]

    async def isLastMessage(self, discordMessage):
        async for msg in discordMessage.channel.history(limit=1):
            return discordMessage.id == msg.id
        
    async def chat(self, discordMessage, images, chatHistory):
        if discordMessage.reference:
            prompt = {"role": "user", "content": self.chatTemplate.format(userMessage="(referring to message ID: %s) %s" % (discordMessage.reference.message_id, discordMessage.content)), "images": images}
        else:
            prompt = {"role": "user", "content": self.chatTemplate.format(userMessage=discordMessage.content), "images": images}
        response = await self.client.chat(
            model="gemma3:12b",
            format="json",
            options={"stream": False, "temperature": 0, "keep_alive":-1},
            messages=[
                self.systemPrompt,
                *[message["ollamaPrompt"] for message in chatHistory],
                prompt
            ],
        )
        response = json.loads(response["message"]["content"])
        
        if not await self.isLastMessage(discordMessage):
            await discordMessage.reply(response["response"])
        else:
            await discordMessage.channel.send(response["response"])
        if response["search"] and response["request"]:
            texts = await getTexts(query=response["search"], request=response["request"], numResults=int(os.getenv("SEARCH_RESULTS_LIMIT")))
            searchPrompt = {"role": "user", "content": self.searchTemplate.format(texts=texts, searchQuery=response["search"], request=response["request"])}
            searchResponse = await self.client.chat(
                model="gemma3:12b",
                options={"stream": False, "temperature": 0, "keep_alive":-1},
                messages=[
                    self.systemPrompt,
                    {"role": "user", "content": discordMessage.content},
                    {"role": "assistant", "content": response["response"]},
                    searchPrompt
                ],
            )
            await discordMessage.reply(searchResponse["message"]["content"])

            

    async def react(self, discordMessage, images=[], chatHistory=[]):
        reactPrompt = {"role": "user", "content": self.reactTemplate.format(userMessage=discordMessage.content), "images": images}
        response = await self.client.chat(
            model="gemma3:12b",
            options={"stream": False, "temperature": 0, "keep_alive":-1},
            format="json",
            messages=[
                self.systemPrompt,
                *[message["ollamaPrompt"] for message in chatHistory],
                reactPrompt
            ],
        )
        response = json.loads(response["message"]["content"])
        await discordMessage.add_reaction(response["reaction"])
    
    async def sendMessage(self, discordMessage):
        chatHistory = await self.getChatHistory(discordMessage)
        decision = await self.decide(discordMessage, chatHistory=chatHistory)
        if decision:
            chatHistory = await self.fetchHistoryImages(chatHistory)
            if discordMessage.attachments:
                tasks = [self.fetchImageBase64(attachment.url) for attachment in discordMessage.attachments]
                images = await asyncio.gather(*tasks)
            else:
                images = []
            if decision == "chat":
                async with discordMessage.channel.typing():
                    await self.chat(discordMessage, images, chatHistory=chatHistory)
            elif decision == "react":
                await self.react(discordMessage, images, chatHistory=chatHistory)
        else:
            return
    
    async def close(self):
        print("Closing AIChat session...")
        await self.session.close()
