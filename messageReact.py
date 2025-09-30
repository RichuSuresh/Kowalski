import asyncio
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
ollamaUrl = os.getenv("OLLAMA_URL", "http://localhost:11434")

model = ChatOllama(
    model="gemma3:12b",
    base_url=ollamaUrl,
    temperature=1,
    format="json",
    keep_alive=-1,
)

systemTemplate = """
You are kowalsk, an AI assistant and friendly chatbot. React to the user's message with a single emoji from the following list that reflects the message's tone.

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

Say your decision in a single word in the format specified:
{{
    "emoji": "<emoji>"
}}
"""

chatPrompt = ChatPromptTemplate.from_messages(
    [
        ("system", systemTemplate),
        MessagesPlaceholder(variable_name="chatHistory"),
        ("human", "{userMessage}"),
    ]
)
chatChain = chatPrompt | model

async def emojiReaction(userMessage, chatHistory=[]):
    response = await chatChain.ainvoke({
        "chatHistory": chatHistory,
        "userMessage": userMessage,
    })

    response = json.loads(response.content)
    return response["emoji"]