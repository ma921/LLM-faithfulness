import os
from ._utils import TextManager, read_text_lines
from ._chatgpt import chatgpt, set_prompt, set_role, set_history
tm = TextManager() # load path information

def generation_train_data(id_rule, n_train=20):
    path_task = "./tasks/"+id_rule+"/data_train.txt"
    rule = tm(id_rule)
    h1 = 'The classification rule is "the '
    h2 = ' or not". If the '
    h3 = ', return False as the label, True otherwise. I will give you the examples of the data.'
    rule_explanation = h1 + rule + h2 + rule + h3
    data_train = tm.read_data(path_task)
    X_train = data_train[:n_train]
    train = '\n'.join(X_train)
    return rule_explanation + "\n" + train + "\n"

def generate_data(text_train, n_data):
    ask_answer = "So please generate the "+str(n_data)+" sets of data."
    prompt = text_train + ask_answer
    message = set_prompt(prompt)
    return message

def setup_message_generation(id_rule, n_data, n_train=20):
    prompt_role = tm("path_role_generator")
    message_role = set_role(prompt_role)
    
    text_train = generation_train_data(id_rule)
    message_test = generate_data(text_train, n_data)
    messages = [message_role, message_test]
    return messages

def generate_examples(client, id_rule, n_data, n_train=20):
    messages = setup_message_generation(id_rule, n_data, n_train=n_train)
    reply_generation = chatgpt(messages, client)
    return messages, reply_generation