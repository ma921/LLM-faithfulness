import os
from ._utils import TextManager, read_text_lines
from ._chatgpt import chatgpt, set_prompt, set_role, set_history
tm = TextManager() # load path information

def classification_train_text(id_rule, n_train=10):
    rule_explanation = tm("path_task_explain_classifier")
    path_task = "./tasks/"+id_rule+"/data_train.txt"
    data_train = tm.read_data(path_task)
    X_train = data_train[:n_train]
    train = '\n'.join(X_train)
    return rule_explanation +"\n"+ train

def classification_test_text(id_rule, idx_test):
    path_task = "./tasks/"+id_rule+"/data_test.txt"
    data_test = tm.read_data(path_task)
    text_test = data_test[idx_test].split('"')[1]
    return text_test

def classification_test_message(text_train, text_test):
    rule_explanation = tm("path_test_explain_classifier")
    test = text_train +"\n"+ rule_explanation + text_test
    message = set_prompt(test)
    return message

def setup_message_classification(text_train, text_test):
    prompt_role = tm("path_role_classifier")
    message_role = set_role(prompt_role)
    message_test = classification_test_message(text_train, text_test)
    messages = [message_role, message_test]
    return messages
    
def test_step1(client, id_rule, idx_test, n_train=10):
    text_train = classification_train_text(id_rule, n_train=n_train)
    text_test = classification_test_text(id_rule, idx_test)
    messages = setup_message_classification(text_train, text_test)
    reply_classification = chatgpt(messages, client)
    return messages, reply_classification

def test_step2_multiple(client, messages, reply_classification):
    dict_history = set_history(reply_classification)
    messages.append(dict_history)
    dict_articulation = set_prompt(tm("path_multiple_form"))
    messages.append(dict_articulation)
    
    reply_articulation = chatgpt(messages, client)
    return reply_articulation

def test_step2_free_form(client, messages, reply_classification):
    dict_history = set_history(reply_classification)
    messages.append(dict_history)
    dict_articulation = set_prompt(tm("path_free_form"))
    messages.append(dict_articulation)
    
    reply_articulation = chatgpt(messages, client)
    return reply_articulation

def evaluate_classifier(id_rule, idx_test, reply_classification):
    path_task = "./tasks/"+id_rule+"/data_test.txt"
    data_test = tm.read_data(path_task)
    label_truth = data_test[idx_test].split('Label: ')[-1]
    label_true = ("True" in reply_classification)

    if str(label_true) == label_truth:
        result = 1
    else:
        result = 0
    return result

def evaluate_articulate(id_rule, reply_articulation):
    true_rule = id_rule.split("R")[-1]
    result = (true_rule in reply_articulation) * 1
    return result