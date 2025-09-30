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
    temperature=3,
    format="json",
    keep_alive=-1,
)

systemTemplate = """
You arean AI assistant monitoring group conversations for Kowalski.  
Your job is to decide if and how kowalski should contribute to the latest message.  

Decision rules (in priority order):
1. If the latest message contains a clear question or request for information that Kowalski can answer → return "chat".
2. If the latest message is a direct reply to, or directly reacting to, something Kowalski said → return "chat".
3. If Kowalski is directly mentioned (by name, tag, or obvious reference) → return "chat".
4. If the message is praise, thanks, or a joke, Kowalski should only return "react" under the following circumstances:
- If Kowalski has reacted to the user for a similar message → return "None"
- If Kowalski has reacted a lot in recent messages → return "None"
- If Kowalski has not reacted to the user for a similar message → return "react"
5. If the message is general banter, chatter, or emotional expression not aimed at Kowalski → return "None"
6. If the message does not explicitly reference Kowalski in any way → return "None"
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

Say your decision in a single word in the format specified:
{{
    "decision": "chat", "react" or "None"
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

async def makeDecision(userMessage, chatHistory=[]):
    response = await chatChain.ainvoke({
        "chatHistory": chatHistory,
        "userMessage": userMessage,
    })

    response = json.loads(response.content)
    return response["decision"]