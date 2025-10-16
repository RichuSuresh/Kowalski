import asyncio
import aiohttp
import discord
from chat import AIChat
from dotenv import load_dotenv
import os
import redis

load_dotenv()
token = os.getenv("BOT_TOKEN")
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print("Systems online, sir. Awaiting your next order")

@client.event
async def on_message(message):
    if message.author == client.user or message.author.bot:
        return
    asyncio.create_task(client.aiChat.sendMessage(message))

async def main():
    async with aiohttp.ClientSession() as session:
        client.aiChat = AIChat(client, session, chatHistoryLimit=int(os.getenv("CHAT_HISTORY_LIMIT")))
        try:
            await client.start(token)
        except KeyboardInterrupt:
            print("Keyboard interrupt detected, shutting down...")
        except asyncio.CancelledError:
            print("Tasks cancelled — shutting down cleanly...")
        finally:
            await client.aiChat.close()
            print("Clean shutdown ✅")

if __name__ == "__main__":
    asyncio.run(main())