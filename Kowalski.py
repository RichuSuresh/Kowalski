import asyncio
import aiohttp
import discord
from chat import answerQuery, fetch_image_base64
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

async def createChatMessage(message, session):
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

        images = []
        if message.attachments:
            tasks = [fetch_image_base64(session, attachment.url) for attachment in message.attachments]
            images = await asyncio.gather(*tasks)

        return HumanMessage(
            content=[
                {"type": "text", "text": "describe this image"},
                *images
            ]
        )
    
async def getChatHistory(message, limit=10):
    messages = []
    tasks = []
    async with aiohttp.ClientSession() as session:
        async for msg in message.channel.history(limit=limit+1):
            tasks.append(createChatMessage(msg, session))
        tasks = tasks[:0:-1]
        if message.reference:
            repliedMessage = await message.channel.fetch_message(message.reference.message_id)
            tasks.append( createChatMessage(repliedMessage, session))

        messages = await asyncio.gather(*tasks)
    print(messages)
    return messages

async def isLastMessage(message):
    async for msg in message.channel.history(limit=1):
        return message.id == msg.id

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    async def handleMessages():
        chatHistory = await getChatHistory(message, limit=int(chatHistoryLimit))
        decision = await makeDecision(message.content, chatHistory=chatHistory)
        if decision:
            if decision == "chat":
                async with message.channel.typing():
                    response = await answerQuery(message, chatHistory=chatHistory)
                if response != None:
                    if not await isLastMessage(message):
                        await message.reply(response)
                    else:
                        await message.channel.send(response)
                        
            elif decision == "react":
                emoji = await emojiReaction(message.content, chatHistory=chatHistory)
                await message.add_reaction(emoji)
            
    
    asyncio.create_task(handleMessages())

client.run(token)