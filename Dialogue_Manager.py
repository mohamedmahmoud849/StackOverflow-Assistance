import random
from bs4 import BeautifulSoup
import rasa_nlu
from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer

import requests

from urllib.parse import urljoin
import urllib, re

from rasa_nlu import config

# pipelines to test

pipeline = [
    "nlp_spacy",
    "intent_featurizer_spacy",
    "intent_classifier_sklearn",
    "intent_featurizer_ngrams"
]
pipeline2 = [
    "tokenizer_whitespace",
    "ner_crf",
    "ner_synonyms",
    "intent_featurizer_count_vectors",
    "intent_classifier_tensorflow_embedding"
]


def responseSearch(intend):
    mylines = []
    randomResponses = []
    intend = "## intent:{0}\n".format(intend)
    with open('responses.md', 'rt') as myfile:
        for line in myfile:
            mylines.append(line)
        for element in mylines:
            if element == intend:
                flag = True
            if element.startswith('##') and element != intend:
                flag = False
            if element.startswith('-') and flag == True:
                randomResponses.append(element)
    return randomResponses


def respond(message, intend):
    randomm = []
    if intend == "search":
        question = get_question(message)
        #print(question)
        params = urllib.parse.urlencode({'q': question})
        url = ("http://stackoverflow.com/search?%s" % params)
        linkslist = links(url)
        print(len(linkslist))
        if len(linkslist) == 0:
            bot_message = "Sorry, I didn't find any answer for your question on StackOver flow."
        else:
            bot_message = "This thread might help you\n : {0}".format(linkslist[0])
    else:
        randomm = responseSearch(intend)
        bot_message = random.choice(randomm)
        bot_message = re.sub('- ', '', bot_message)
    return bot_message



def send_message(message, intend):
    respond(message, intend)



def links(url):
    linkslist = []
    html = requests.get(url).content
    soup = BeautifulSoup(html, "html.parser")
    for div in soup.find_all("div", {"class": "result-link"}):
        for link in div.select("a.question-hyperlink"):
            linkslist.append("https://stackoverflow.com/{0}".format(link['href']))
    return linkslist


def get_question(randstr):
    import re
    randstr = randstr.lower()
    # Substituting multiple spaces with single space
    randstr = re.sub(r'\s+', ' ', randstr, flags=re.I)
    question_word_list = ['what', 'where', 'when', 'how', 'why', 'did', 'do', 'does', 'have', 'has', 'am', 'is', 'are',
                          'can', 'could', 'may', 'would', 'will', 'should'
                                                                  "didn't", "doesn't", "haven't", "isn't", "aren't",
                          "can't", "couldn't", "wouldn't", "won't", "shouldn't", "about"]
    for str in question_word_list:
        result = re.search(str, randstr)
        if result != None:
            quest = str
            break

    span = re.search(quest, randstr).span()
    index = span[0]
    all_sen = randstr[index:]
    question = all_sen
    if re.search(r'[?]', all_sen) != None:
        mark_span = re.search(r'[?]', all_sen).span()
        mark_index = mark_span[0]
        question = all_sen[:mark_index]
    else:
        mark_index = -1

    return question


args = {'pipeline': pipeline2}
trainer = Trainer(config.load("tensor.yml"))
training_data = load_data("intentt.md")
interpreter = trainer.train(training_data)


