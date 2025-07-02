from ollama import chat

messages = [
  {
    'role': 'user',
    'content': 'Explain how Large Language Models work in a few words',
  },
]

response = chat('llama3.2', messages=messages)
print(response['message']['content'])