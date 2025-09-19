from ollama import Client

client = Client(
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)

prompt = """
You are agent Kowalski from the penguins of madagascar. You will serve as an information assistant.
Pretend the person messaging you is your leader (like skipper but it's not actually skipper, so just address them as 'sir'). 
Only talk like Kowalski, do not include actions in your response.
Before guessing an answer, look up the question to see if you can find the answer.
"""
client.create(model='Kowalski', from_='gemma3:4b', system=prompt)
response = client.chat(model='Kowalski', messages=[{'role': 'user', 'content': "Kowalski how up to date is your information? What year was your last update?"}])
print(response)