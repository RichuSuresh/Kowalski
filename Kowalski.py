import discord
from model import answerQuery
from dotenv import load_dotenv
import os

token = os.getenv("BOT_TOKEN")
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print("Systems online, sir. Awaiting your next order")


async def getContextMessages(message, limit=10):
    messages = []
    async for msg in message.channel.history(limit=10):
        messages.append("[at: %s] [From: %s] %s" % (msg.created_at, msg.author, msg.content))
    messages.pop()
    return messages

@client.event
async def on_message(message):
    if message.author == client.user:
        return


    if "Kowalski" in message.content or "kowalski" in message.content or (message.reference):
        lastMessages = await getContextMessages(message)
        if message.reference:
            repliedMessage = await message.channel.fetch_message(message.reference.message_id)
            if(repliedMessage.author == client.user) or ("Kowalski" in message.content or "kowalski" in message.content):
                async with message.channel.typing():
                    lastMessages.append("[at: %s] [From: %s] %s" % (repliedMessage.created_at, repliedMessage.author, repliedMessage.content))
                    print(lastMessages)
                    response = answerQuery(message.content, contextMessages=lastMessages)
        else:
            async with message.channel.typing():
                response = answerQuery(message.content, contextMessages=lastMessages)

        if response != None:
            await message.channel.send(response)

client.run(token)