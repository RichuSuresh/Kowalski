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

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.reference:
        repliedMessage = await message.channel.fetch_message(message.reference.message_id)
        if repliedMessage.author == client.user:
            await message.channel.send(answerQuery(userMessage=message.content, contextMessages=[repliedMessage.content]))

    if "Kowalski" in message.content or "kowalski" in message.content:
        await message.channel.send(answerQuery(message.content))

client.run(token)