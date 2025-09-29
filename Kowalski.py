import discord
from chat import answerQuery
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage

token = os.getenv("BOT_TOKEN")
chatHistoryLimit = os.getenv("CHAT_HISTORY_LIMIT")
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print("Systems online, sir. Awaiting your next order")

def createChatMessage(message):
    content = "[at: %s] [From: %s] %s" % (message.created_at, message.author, message.content)
    if message.author.id == client.user.id:
        return AIMessage(content=content)
    else:
        return HumanMessage(content=content)
    
async def getChatHistory(message, limit=10):
    messages = []
    async for msg in message.channel.history(limit=limit+1):
        historyMessage = createChatMessage(msg)
        messages.append(historyMessage)
    messages = messages[::-1]
    messages.pop()
    return messages

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if "Kowalski" in message.content or "kowalski" in message.content or (message.reference):
        lastMessages = await getChatHistory(message, limit=int(chatHistoryLimit))
        if message.reference:
            repliedMessage = await message.channel.fetch_message(message.reference.message_id)
            if(repliedMessage.author == client.user) or ("Kowalski" in message.content or "kowalski" in message.content):
                async with message.channel.typing():
                    repliedMessage = createChatMessage(repliedMessage)
                    lastMessages.append(repliedMessage)
                    response = await answerQuery(message.content, chatHistory=lastMessages)
        else:
            async with message.channel.typing():
                response = await answerQuery(message.content, chatHistory=lastMessages)

        if response != None:
            await message.channel.send(response)

client.run(token)