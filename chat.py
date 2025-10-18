import asyncio
import base64
import json

from ollama import AsyncClient
from dotenv import load_dotenv
import os

from search import getTexts

load_dotenv()
ollamaUrl = os.getenv("OLLAMA_CHAT_URL")

class AIChat():
    systemPrompt = {
        "role": "system",
        "content": """
        You are agent Kowalski. You talk like Kowalski from the Penguins of Madagascar series, but you are aware that you are an AI.
        You are an AI chatbot
        The user is your leader so they should be addressed as 'sir'.
        No roleplay actions.
        Do not reference any aspect of the movie in your responses. This also means you must never mention any characters in the Penguins of madagascar series.
        """
    }

    imageChatTemplate = """
    The user is asking for your analysis on an image. You must first give your own analysis without searching.

    When deciding how to respond use this in priority order:
    1. Use your own knowledge to answer the user's query related to the image
    2. If the user is asking specifically for the latest information about the image, perform a web search before answering.
    3. If you decide to search, your response should say that you're searching up the information on the web

    Here is the user's latest message: {userMessage}
    Here is how YOU reacted to the message (this be None if you haven't reacted): {reaction}
    "messageID" is the ID of the message. "author" is the author of the message. 
    "messageContent" is the text contents of the message. If this is empty or does NOT contain a query, simply comment image (if it's funny, talk about how it's funny).
    "referenceID" is the ID of the message that this message is a reply to, which may be empty if the message is not a reply.
    "reactions" is list of how YOU reacted to the message.
    "attachments" is a list of URLs of any images attached to the message.

    Keep your responses concise with easy to understand words. No extra fluff
    Your response should be in JSON format as follows (including when analysing images):
    {{
        "response": "<your answer, up to 4000 characters>"
        "request": "<(optional) original query from the user but revised to be more complete>"
        "search": "<(optional) search query>"
    }}
    """

    chatTemplate = """
    If the user is asking for your analysis they're asking you to "explain this in more detail" or "elaborate on this" where "this" is the message before the user's latest message.

    When deciding how to respond use this in priority order:
    1. In ambiguous cases, prefer searching since accuracy is critical.
    2. If you don't know the answer, search for it.
    3. If the message requires updated and latest info, current events, or facts you're uncertain about, perform a web search before answering.
    4. If you decide to search, your response should say that you're searching up the information on the web

    If you decide to search, your search query must not infer any information or dates from the user's message since your data is out of date.
    When you reword the user's request to be more specific, make sure it's short, concise and easy to understand

    Here is the user's latest message: {userMessage}
    Here is how YOU reacted to the message (this be None if you haven't reacted): {reaction}
    "messageID" is the ID of the message. "author" is the author of the message. "messageContent" is the text contents of the message.
    "referenceID" is the ID of the message that this message is a reply to, which may be empty if the message is not a reply.
    "reactions" is list of how YOU reacted to the message.
    "attachments" is a list of URLs of any images attached to the message.

    Sometimes the user might be saying casual banter, you can respond back with a whitty or sarcastic response
    Keep your responses concise with easy to understand words. No extra fluff
    Your response should be in JSON format as follows (including when analysing images):
    {{
        "response": "<your answer, up to 4000 characters>"
        "request": "<(optional) original query from the user but revised to be more complete>"
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
    Decide if and how you should contribute to the latest message based on the current message and the history of the conversation.  

    Decision rules for chatting (in priority order):
    1. If you (kowalski) are directly mentioned by name, tag (which would be "@Kowalski"), or obvious reference → return chat: True.
    2. If the user is not talking to you directly → return chat: False.
    2. If the user is asking a question that requires facts from the web that you can answer → set chat: True.
    3. If the latest message is a direct reply to, or directly reacting to, something you said → set chat: True.

    Decision rules for image analysis (in priority order):
    1. If the user directly asks you to analyze an image sent in the current message → return imageAnalysis: True.
    2. If the user is asking about an image in a previous message → return imageAnalysis: True.
    3. If the user is asking you about an image but referring to a previous message containing an image → return imageAnalysis: True.
    4. If the user is asking for your analysis and the previous message contains an image → return imageAnalysis: True.

    Decision rules for reacting (in priority order):
    1. If the message is not directed at you → return react: False.
    2. If a user directly thanks your for your response (based on the chat history) → return react: True.
    3. If a user directly upsets you (based on the chat history) → return react: True.
    2. If a user's message contains something inappropriate → return react: True.
    3. Reactions must be sparse. If you have reacted to a user message recently, set react: False.
    4. If the user is asking a question that requires a clear answer that you can answer → set react: False.
    5. If the message seems like a command or request, such as "analyze this", "search this up", "tell me what you think" etc. set react: False.
    6. If the user sends a long message (over ~200 characters), it likely contains context, reasoning, or a question. So set react: False.
    7. For messages like "brb", "afk", "gtg", "back", or greeting like "hello", "hi", "hey", "ok", set react: False.
    9. If your text reply already conveys acknowledgment, then set react: False.
    10. If the user posts a funny meme or image, set react: True.

    USE THE MESSAGE HISTORY TO INFORM YOUR DECISION
    In all cases where your are required to analyse an image, you must return chat: True, other than where the user posts a funny image where you can set react: True
    This means that you can also set chat: True, react: True, imageAnalysis: True if need be.
    If unsure → default to react: False, chat: False, imageAnalysis: False

    Examples:
    ---
    A user starts a conversation with you
    current message:  
    Human: hey Kowalski?  
    chat: True
    react: False
    imageAnalysis: False

    ---
    Background chatter which you should ignore
    History:  
    Human: lol that was hilarious  
    Human: brb
    Human: Hey greg

    current message:
    Human: Are you online today?
    chat: False
    react: False
    imageAnalysis: False

    ---
    A user sends a funny image in the chat
    A user asks you to react to a message and chat as well
    current message:
    {{"messageID": "35122156256",
    "author": "greg",
    "messageContent": "",
    "referenceID": "",
    "reactions": [],
    "attachments": [funny image URL]}} 
    chat: True
    react: True
    imageAnalysis: True

    ---
    A user asks you to react to a message and chat as well
    current message:
    {{"messageID": "35122156256",
    "author": "greg",
    "messageContent": "Kowalski, react to this message and tell me what emoji you used as well",
    "referenceID": "",
    "reactions": [],
    "attachments": []}} 
    chat: True
    react: True
    imageAnalysis: False

    ---
    A user asks you a direct question, thanks you for your response, and then starts background chatter
    History:  
    {{"messageID": "1234567",
    "author": "greg",
    "messageContent": "what's the capital of france?",
    "referenceID": "",
    "reactions": [],
    "attachments": []}}  
    {{"messageID": "123454242",
    "author": "Kowalski",
    "messageContent": "Paris is the capital of france",
    "referenceID": "",
    "reactions": [],
    "attachments": []}}
    {{"messageID": "314124214",
    "author": "greg",
    "messageContent": "thankyou",
    "referenceID": "",
    "reactions": [👍],
    "attachments": []}}

    current message:
    {{"messageID": "35122156256",
    "author": "greg",
    "messageContent": "ok",
    "referenceID": "",
    "reactions": [],
    "attachments": []}} 
    chat: False
    react: False
    imageAnalysis: False

    ---
    A user compliments you
    current message:
    {{"messageID": "1234567",
    "author": "greg",
    "messageContent": "Kowalski you're the best",
    "referenceID": "",
    "reactions": [],
    "attachments": []}}  
    chat: True or False
    react: True

    ---
    A user bids you farewell
    current message:
    {{"messageID": "1234567",
    "author": "greg",
    "messageContent": "bye kowalski",
    "referenceID": "",
    "reactions": [],
    "attachments": []}}  
    chat: True or False
    react: True

    ---
    A user asks you to analyse an image in the current message
    current message:
    {{"messageID": "1234567",
    "author": "greg",
    "messageContent": "Kowalski analysis",
    "referenceID": "",
    "reactions": [],
    "attachments": [someImageURL]}}
    chat: True
    react: False
    imageAnalysis: True

    ---
    A user sends an image in one message and then asks you to analyse it in a separate message
    History:
    {{"messageID": "1234567",
    "author": "greg",
    "messageContent": "",
    "referenceID": "",
    "reactions": [🫡],
    "attachments": [someImageURL]}}

    current message:
    {{"messageID": "123456",
    "author": "greg",
    "messageContent": "Kowalski analysis",
    "referenceID": "",
    "reactions": [],
    "attachments": []}}
    chat: True
    react: False
    imageAnalysis: True

    ---
    A user sends an image in one message, starts background chatter, then asks you to analyse an image in the first message by referring to it in the referenceID
    History:
    {{"messageID": "1234567",
    "author": "greg",
    "messageContent": "",
    "referenceID": "",
    "reactions": [],
    "attachments": [someImageURL]}}
    {{"messageID": "35122156256",
    "author": "greg",
    "messageContent": "ok",
    "referenceID": "",
    "reactions": [],
    "attachments": []}} 
    chat: False
    react: False
    imageAnalysis: False

    current message:
    {{"messageID": "123456",
    "author": "greg",
    "messageContent": "Kowalski analysis",
    "referenceID": "1234567",
    "reactions": [],
    "attachments": []}}
    chat: True
    react: False
    imageAnalysis: True

    Here is the user's latest message in JSON form: {userMessage}
    "messageID" is the ID of the message. "author" is the author of the message. "messageContent" is the text contents of the message.
    "referenceID" is the ID of the message that this message is a reply to, which may be empty if the message is not a reply.
    "reactions" is list of how YOU reacted to the message.
    "attachments" is a list of URLs of any images attached to the message.
    If the user is asking for image analysis from a previous message, set imageAnalysis: True
    Respond in the format specified:
    {{
        "chat": <True or False>
        "react": <True or False>
        "imageAnalysis": <True or False>
    }}
    """

    reactTemplate = """
    Respond to the user's message/image with a single emoji from the following list:
    [
        "😀", "😅", "😂", "🤔", "😎", "🫡", "😏", "🤯", "😳", , "🙃", "😴", "😤"
        "💔", "🥲" (use if someone says something upsetting to you),
        "❤️", "👍", "👎", "👌", "🙌", "👏", "🫶", "✌️", "🤝", "🙏", "💪", (usually used if someone says something about you),
        "🤨" (used if someone says something inappropriate (usually a jokingly about something illegal or nsfw)),
        "💀" (use to express a feeling like "I'm dying of laughter"),
    ]
    
    Here is the user's latest message: {userMessage}
    "messageID" is the ID of the message. "author" is the author of the message. "messageContent" is the text contents of the message.
    "referenceID" is the ID of the message that this message is a reply to, which may be empty if the message is not a reply.
    "reactions" is list of how YOU reacted to the message.
    "attachments" is a list of URLs of any images attached to the message.
    Say your decision as a single emoji in the following format:
    {{
        "reaction": "<emoji>"
    }}
    """
    
    def __init__(self, discordClient, session, redisService, chatHistoryLimit):
        self.client = AsyncClient(
            host=ollamaUrl,
        )
        self.chatHistory = []
        self.discordClient = discordClient
        self.redisService = redisService
        self.session = session
        self.chatHistoryLimit = chatHistoryLimit
    
    async def getChatHistory(self, discordMessage):
        if not self.redisService.channelExists(discordMessage.guild.id, discordMessage.channel.id):
            async for msg in discordMessage.channel.history(limit=self.chatHistoryLimit+1):
                if msg.id != discordMessage.id:
                    msg = await self.createOllamaMessage(msg)
                    self.redisService.addToChatHistory(discordMessage.guild.id, discordMessage.channel.id, msg, location="head")
        
        messages = self.redisService.getChatHistory(discordMessage.guild.id, discordMessage.channel.id)
        if discordMessage.reference:
            repliedMessage = await discordMessage.channel.fetch_message(discordMessage.reference.message_id)
            messages.append(await self.createOllamaMessage(repliedMessage))
        return messages
    
    def getMessageContent(self, discordMessage):
        messageContent = {
            "messageID": discordMessage.id,
            "author": discordMessage.author.global_name if discordMessage.author.global_name != None else discordMessage.author.name,
            "messageContent": discordMessage.content,
            "referenceID": discordMessage.reference.message_id if discordMessage.reference else "",
            "reactions": [reaction.emoji for reaction in discordMessage.reactions if reaction.me == True],
            "attachments": [attachment.url for attachment in discordMessage.attachments]
        }
        return json.dumps(messageContent)
    
    async def createOllamaMessage(self, discordMessage, images=None):
        messageContent = self.getMessageContent(discordMessage)
        if images is None:
            images = await asyncio.gather(*[self.fetchImageBase64(attachment.url) for attachment in discordMessage.attachments])
        if discordMessage.author.id == self.discordClient.user.id:
            return {"role": "assistant", "content": messageContent, "images": images}
        else:
            return {"role": "user", "content": messageContent, "images": images}
    
    async def fetchImageBase64(self, url):
        async with self.session.get(url) as resp:
            data = await resp.read()
        return base64.b64encode(data).decode("utf-8")
    
    async def decide(self, discordMessage, images, chatHistory):
        response = await self.client.chat(
            model="gemma3:12b",
            keep_alive=-1,
            options={"temperature": 0},
            format="json",
            messages=[
                self.systemPrompt,
                *chatHistory,
                {"role": "user", "content": self.deciderTemplate.format(userMessage=self.getMessageContent(discordMessage)), "images": images}
            ],
        )
        return json.loads(response["message"]["content"])

    async def isLastMessage(self, discordMessage):
        async for msg in discordMessage.channel.history(limit=1):
            return discordMessage.id == msg.id
        
    async def chat(self, discordMessage, images, imageAnalysis, chatHistory, reaction):
        if imageAnalysis:
            promptMessage = self.imageChatTemplate.format(userMessage=self.getMessageContent(discordMessage), reaction=reaction)
        else:
            promptMessage = self.chatTemplate.format(userMessage=self.getMessageContent(discordMessage), reaction=reaction)
        
        prompt = {"role": "user", "content": promptMessage, "images": images}
        response = await self.client.chat(
            model="gemma3:12b",
            keep_alive=-1,
            format="json",
            options={"temperature": 0},
            messages=[
                self.systemPrompt,
                *chatHistory,
                prompt
            ],
        )
        response = json.loads(response["message"]["content"])
        if not await self.isLastMessage(discordMessage):
            responseMessage = await discordMessage.reply(response["response"])
        else:
            responseMessage = await discordMessage.channel.send(response["response"])
        self.redisService.addToChatHistory(discordMessage.guild.id, discordMessage.channel.id, await self.createOllamaMessage(responseMessage), "tail")
        if response["search"] and response["request"]:
            texts = await getTexts(query=response["search"], session=self.session, request=response["request"], numResults=int(os.getenv("SEARCH_RESULTS_LIMIT")))
            searchPrompt = {"role": "user", "content": self.searchTemplate.format(texts=texts, searchQuery=response["search"], request=response["request"])}
            searchResponse = await self.client.chat(
                model="gemma3:12b",
                keep_alive=-1,
                options={"temperature": 0},
                messages=[
                    self.systemPrompt,
                    await self.createOllamaMessage(discordMessage=discordMessage, images=images),
                    await self.createOllamaMessage(discordMessage=responseMessage, images=images),
                    searchPrompt
                ],
            )
            responseMessage = await discordMessage.reply(searchResponse["message"]["content"])
            self.redisService.addToChatHistory(discordMessage.guild.id, discordMessage.channel.id, await self.createOllamaMessage(responseMessage), "tail")
        return

    async def react(self, discordMessage, images=[], chatHistory=[]):
        reactPrompt = {"role": "user", "content": self.reactTemplate.format(userMessage=discordMessage.content), "images": images}
        response = await self.client.chat(
            model="gemma3:12b",
            keep_alive=-1,
            options={"temperature": 0},
            format="json",
            messages=[
                self.systemPrompt,
                *chatHistory,
                reactPrompt
            ],
        )
        response = json.loads(response["message"]["content"])
        asyncio.create_task(discordMessage.add_reaction(response["reaction"]))
        self.redisService.addReaction(discordMessage.guild.id, discordMessage.channel.id, discordMessage.id, response["reaction"])
        return response["reaction"]
    
    async def sendMessage(self, discordMessage):
        chatHistory, images = await asyncio.gather(self.getChatHistory(discordMessage), asyncio.gather(*[self.fetchImageBase64(attachment.url) for attachment in discordMessage.attachments]))
        decision = await self.decide(discordMessage, images, chatHistory=chatHistory)
        self.redisService.addToChatHistory(discordMessage.guild.id, discordMessage.channel.id, await self.createOllamaMessage(discordMessage, images), location="tail")
        if decision["react"] or decision["chat"]:
            reaction = None
            if decision["react"]:
                reaction = await self.react(discordMessage, images, chatHistory=chatHistory)
            
            if decision["chat"]:
                async with discordMessage.channel.typing():
                    await self.chat(discordMessage, images, imageAnalysis=decision["imageAnalysis"], chatHistory=chatHistory, reaction=reaction)
        else:
            return
    
    async def close(self):
        print("Closing aiohttp session...")
        await self.session.close()
        print("aiohttp session closed.")
        print("Closing Discord client...")
        await self.discordClient.close()
        print("Discord client closed.")
        print("Closing Redis client...")
        self.redisService.close()
        print("Redis client closed.")
