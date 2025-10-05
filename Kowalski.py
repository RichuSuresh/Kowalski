import asyncio
import aiohttp
import discord
from chat import AIChat
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("BOT_TOKEN")
chatHistoryLimit = os.getenv("CHAT_HISTORY_LIMIT")
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

    async def handleMessages():
        await client.aiChat.sendMessage(message)
            
    asyncio.create_task(handleMessages())

async def main():
    async with aiohttp.ClientSession() as session:
        client.aiChat = AIChat(client, session)
        try:
            await client.start(token)
        finally:
            await client.aiChat.close()
            print("Clean shutdown ✅")

if __name__ == "__main__":
    asyncio.run(main())