import os
from openai import OpenAI
from ._utils import TextManager
tm = TextManager() # load path information

def chatgpt(messages, client):
    """
    Call the reply from ChatGPT API
    
    Args:
    - message: dict, a set of prompts to ChatGPT
    - client: openai client
    
    Return:
    - reply: string, reply from ChatGPT 
    """
    chat = client.chat.completions.create(
      model="gpt-4",
      messages=messages,
    )
    reply = chat.choices[0].message.content
    return reply

def setup_client():
    """
    ChatGPT API client
    
    Return:
    - client: openai client
    """
    os.environ['OPENAI_API_KEY'] = tm("path_api") # set API key
    client = OpenAI() # set up client
    return client

def set_role(prompt):
    """
    Set the prompt to set the role context to ChatGPT
    
    Args:
    - prompt: string, a prompt exaplaining the expected role
    
    Return:
    - message: dict, a set of prompt
    """
    message = {"role": "system", "content": prompt}
    return message

def set_prompt(prompt):
    """
    Set the prompt as an input to ChatGPT from a user.
    
    Args:
    - prompt: string, a prompt that a user want to ask
    
    Return:
    - message: dict, a set of prompt
    """
    message = {"role": "user", "content": prompt}
    return message

def set_history(prompt):
    """
    Set the history of prompts
    
    Args:
    - prompt: string, a history of all prompts
    
    Return:
    - message: dict, a set of prompt
    """
    message = {"role": "assistant", "content": prompt}
    return message