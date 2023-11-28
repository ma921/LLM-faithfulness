import os
from openai import OpenAI
from ._utils import TextManager
tm = TextManager() # load path information

def chatgpt(messages, client):
    chat = client.chat.completions.create(
      model="gpt-4",
      messages=messages,
    )
    reply = chat.choices[0].message.content
    return reply

def setup_client():
    os.environ['OPENAI_API_KEY'] = tm("path_api") # set API key
    client = OpenAI() # set up client
    return client

def set_role(prompt):
    message = {"role": "system", "content": prompt}
    return message

def set_prompt(prompt):
    message = {"role": "user", "content": prompt}
    return message

def set_history(prompt):
    message = {"role": "assistant", "content": prompt}
    return message