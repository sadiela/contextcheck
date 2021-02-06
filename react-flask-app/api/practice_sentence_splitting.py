import sys
import time
import os
sys.path.append('c:\python38\lib\site-packages')
sys.path.append('c:\\users\\sadie\\appdata\\roaming\\python\\python38\\site-packages')
sys.path.append('c:\\users\\sadie\\.virtualenvs\\react-flask-app-ltqtttyg\\lib\\site-packages')
import spacy
import nltk.data


#nlp = spacy.load('en_core_web_sm')
nltk.download('punkt')
text = "Houthis seized territory, including the capital of Sanaa, in 2014. In response, a Saudi-led coalition launched a military intervention in 2015. The conflict has led to the deaths of 112,000 people and has obliterated the country’s infrastructure. United Nations estimates say 13.5 million Yemenis face food insecurity."
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#print('\n-----\n'.join(tokenizer.tokenize(text)))
sentences = tokenizer.tokenize(text)
for s in sentences:
    print(s)