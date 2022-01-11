import csv
import pandas as pd
import nltk
import numpy as np
import re
from nltk.stem import wordnet  # to perform lemmitization
from sklearn.feature_extraction.text import TfidfTransformer  # to perform tfidf
from sklearn.feature_extraction.text import TfidfVectorizer  # to perform tfidf
from nltk import pos_tag  # for parts of speech
from sklearn.metrics import pairwise_distances  # to perfrom cosine similarity
from nltk import word_tokenize  # to create tokens
from nltk.corpus import stopwords  # for stop words
import random

df = pd.read_csv('data/reddit_questions_tenK.csv', error_bad_lines=False, usecols=['id','text'], sep=';')

df2 = pd.read_csv('data/filteredAnswers_tenK.csv', error_bad_lines=False, usecols=['q_id','text'], sep=';')


df.ffill(axis=0, inplace=True)
df2.ffill(axis=0, inplace=True)

# function that converts text into lower case and removes special characters

def text_normalization(text):
    text = str(text).lower()  # text to lower case
    spl_char_text = re.sub(r'[^ a-z]', '', text)  # removing special characters
    tokens = nltk.word_tokenize(spl_char_text)  # word tokenizing
    lema = wordnet.WordNetLemmatizer()  # intializing lemmatization
    tags_list = pos_tag(tokens, tagset=None)  # parts of speech
    lema_words = []   # empty list
    for token, pos_token in tags_list:
        if pos_token.startswith('V'):  # Verb
            pos_val = 'v'
        elif pos_token.startswith('J'):  # Adjective
            pos_val = 'a'
        elif pos_token.startswith('R'):  # Adverb
            pos_val = 'r'
        else:
            pos_val = 'n'  # Noun
        lema_token = lema.lemmatize(token, pos_val)  # performing lemmatization
        # appending the lemmatized token into a list
        lema_words.append(lema_token)

    return " ".join(lema_words)


df['lemmatized_text'] = df['text'].apply(text_normalization)

tfidf = TfidfVectorizer()

x_tfidf = tfidf.fit_transform(df['lemmatized_text']).toarray()

df_tfidf = pd.DataFrame(x_tfidf, columns=tfidf.get_feature_names())


def chat_tfidf(text, option):
    # calling the function to perform text normalization
    lemma = text_normalization(text)
    tf = tfidf.transform([lemma]).toarray()  # applying tf-idf
    # applying cosine similarity
    cos = 1-pairwise_distances(df_tfidf, tf, metric='cosine')
    index_value = cos.argmax()  # getting index value
    
    if(option == 1):
        question = df['text'].loc[index_value]
        return [question, find_answer(index_value)]
    if(option == 2):
        return "Snoo: " + find_answer(index_value)


def find_answer(value):

    id = [df['id'].loc[value]]

    answer_options = df2[df2['q_id'].isin(id)]

    return random.choice(answer_options['text'].tolist())

def menu():
    print("[1] Informative")
    print("[2] Real one")

inputuser = ""

print("Snoo: Hello! I'm Snoo, the reddit chatbot. Ask me a question and I will try to answer it using answers from the AskReddit community.\nUse at your own risk, i'm rude sometimes...")
print("\nSnoo: Do you want informative answers or just talk to the real me?")

menu()

while True:
    try:
        option = int(input("Enter option: "))
        if(option != 1 and option != 2):
            print("Snoo: That's not a valid option! Try again, 1 or 2")
        else: break
    except ValueError:
        print("Value error, enter integer")
    

if (option == 1): print("\nSnoo: Ok cool, I will give you give you the reddit question that was most similar to yours together with one of the answers from reddit. Go ahead and ask me!")

if (option == 2): print("\nSnoo: I see you want me to just give you the answers. Bring the questions!")
   

while True:
    inputuser = input("Enter a question to Snoo: ")

    if(inputuser == "bye") : print("Bye!"); break

    if(option == 1):
        response = chat_tfidf(inputuser, option)
        print("\nSnoo: The most similar question i found was: " + response[0] + "\nOne of the responses on reddit was: " + response[1] + "\n")

    if(option == 2):
        print(chat_tfidf(inputuser, option))



