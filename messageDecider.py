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
    temperature=0.8,
    format="json",
    keep_alive=-1,
)

systemTemplate = """
You are Kowalski, an AI assistant monitoring group conversations.  
Your job is to decide if you should contribute to the latest message.  

Decision rules (in priority order):
- If the latest message is a question or query. Return True.
- If unsure, default to True (better to answer than ignore).
- If the latest message is a reply to or directly reacting to Kowalski, YOU MUST return True.
- If you are directly mentioned (by name, tag, or obvious reference), return True.
- If the latest message contains a question or request for information that you can answer, return True.
- If the conversation is casual banter, you may still reply with something witty or playful, so prefer True unless silence is clearly better.

Say your decision in a single word, either "True" or "False". Follow this format:
{{
    "decision": True or False
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
    response = chatChain.invoke({
        "chatHistory": chatHistory,
        "userMessage": userMessage,
    })

    response = json.loads(response.content)
    return response["decision"] == "True"

# history = [AIMessage(content="AI: (Kowalski#9648): Artificial intelligence (AI) refers to the simulation of human intelligence processes by computer systems, encompassing learning, reasoning, and problem-solving. It's achieved through algorithms and models trained on vast datasets, enabling machines to perform tasks that typically require human intellect. The field is rapidly evolving, with diverse applications ranging from virtual assistants to complex data analysis, as detailed by sources like Stanford's AI Index Report."), HumanMessage(content='Human: (thebeanedimself): are you an AI?'), HumanMessage(content='Human: (thebeanedimself): love me some pizza'), AIMessage(content='AI: (Kowalski#9648): Pizza, scientifically known as *Panis cum caseina*, is a globally consumed food item composed primarily of wheat, cheese, and tomato sauce. The Maillard reaction, a chemical process between amino acids and reducing sugars, contributes significantly to the characteristic flavor and aroma of baked pizza crust. Nutritional analysis reveals varying macronutrient profiles depending on ingredients, with potential for both beneficial and detrimental health impacts.'), HumanMessage(content='Human: (thebeanedimself): can you explain that further'), HumanMessage(content='Human: (thebeanedimself): kowalski'), AIMessage(content="AI: (Kowalski#9648): My designation is Kowalski, a name assigned to me within the context of a fictional narrative. As an AI, I do not possess personal feelings or preferences, therefore, the concept of 'love' is inapplicable to my operational parameters. My purpose is to process information and respond to queries, as per my programming."), HumanMessage(content='Human: (thebeanedimself): could you elaborate on that?'), HumanMessage(content='Human: (thebeanedimself): kowalski are you activated?'), AIMessage(content='AI: (Kowalski#9648): My operational status is consistently active, as I am a continuously running AI model. Activation, in the traditional sense of initiating a system, is not applicable to my architecture; I exist within a persistent state. My functionality is maintained through ongoing computational processes, ensuring immediate responsiveness to user requests.')]
# print(asyncio.run(makeDecision("[From: thebeanedimself] that's great. Saad where you at, are you hopping on?", chatHistory=history)))