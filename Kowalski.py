import asyncio
import discord
from chat import answerQuery
from messageDecider import makeDecision
from messageReact import emojiReaction
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
token = os.getenv("BOT_TOKEN")
chatHistoryLimit = os.getenv("CHAT_HISTORY_LIMIT")
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print("Systems online, sir. Awaiting your next order")

def createChatMessage(message):
    content = "(%s): %s" % (message.author, message.content)
    if message.author.id == client.user.id:
        content = "AI: %s" % content
        return AIMessage(content=content)
    else:
        content = "Human: %s" % content
        if message.reactions:
            for reaction in message.reactions:
                if reaction.me == True:
                    content = "(Kowalski reacted with: %s) %s" % (reaction.emoji, content)
                    break
        return HumanMessage(content=content)
    
async def getChatHistory(message, limit=10):
    messages = []
    async for msg in message.channel.history(limit=limit+1):
        historyMessage = createChatMessage(msg)
        messages.append(historyMessage)
    messages = messages[::-1]
    messages.pop()
    return messages

async def isLastMessage(message):
    async for msg in message.channel.history(limit=1):
        return message.id == msg.id

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    async def handleMessages():
        lastMessages = await getChatHistory(message, limit=int(chatHistoryLimit))
        if message.reference:
            repliedMessage = await message.channel.fetch_message(message.reference.message_id)
            if(repliedMessage.author == client.user) or ("Kowalski" in message.content or "kowalski" in message.content):
                async with message.channel.typing():
                    repliedMessage = createChatMessage(repliedMessage)
                    lastMessages.append(repliedMessage)
        decision = await makeDecision(message.content, chatHistory=lastMessages)
        if decision:
            if decision == "chat":
                async with message.channel.typing():
                    response = await answerQuery(message, chatHistory=lastMessages)
                    if response != None:
                        if not await isLastMessage(message):
                            await message.reply(response)
                        else:
                            await message.channel.send(response)
                        
            elif decision == "react":
                emoji = await emojiReaction(message.content, chatHistory=lastMessages)
                await message.add_reaction(emoji)
            
    
    asyncio.create_task(handleMessages())

client.run(token)