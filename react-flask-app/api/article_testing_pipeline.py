import sys
sys.path.append('c:\python38\lib\site-packages')
sys.path.append('c:\\users\\sadie\\appdata\\roaming\\python\\python38\\site-packages')
sys.path.append('..\\..\\Related_Articles')
sys.path.append('../../Related_Articles')
from flask import Flask, request, jsonify
import json
import TestSentence
import time
import string
#import newscraper
#import pymongo
#from pymongo import MongoClient
import nltk.data
#import RelatedArticles_five_calls #import getarticles

# Incorporate web scrape 

# Take list of URLS --> if web scraper working
# Assume single text file with urls separated by newlines
url_filename = "../../testing/urls.txt" # include directory
#with 

# OR
# Take directory with text files
# Return list of bias scores


def analyze_sentences(text, start_time):
    # Split into multiple sentences here
    nltk.download('punkt')
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_tokenizer.tokenize(text)
    #sentences = var.split('. ')
    #print(sentences)

    # Run through algorithm 
    results = TestSentence.output(sentences)

    print('SENTENCE RESULTS!', results['sentence_results']['article_score'])
    #results['runtime'] = str(time.time() - start_time) + " seconds\n"
    #return_res = results['sentence_results']
    return results

print("Start")

text = 'House Democrats on Wednesday announced that a floor vote will be held Thursday to remove Representative Marjorie Taylor Greene from her committee assignments after House Republicans refused to take swift action against the conspiracy-mongering Georgia congresswoman. “I spoke to Leader McCarthy this morning, and it is clear there is no alternative to holding a Floor vote on the resolution to remove Rep. Greene from her committee assignments,” House Majority Leader Steny Hoyer said in a statement. Hoyer added that the Rules Committee is scheduled to meet Wednesday afternoon, and the House will vote on the resolution on Thursday. House Minority Leader Kevin McCarthy met with Greene for several hours on Tuesday night, but House GOP leadership has appeared reticent to punish her. House Speaker Nancy Pelosi chastised Republican leadership on Thursday for failing to do more to rebuke Greene after reports that Greene previously indicated support on social media for executing prominent Democratic politicians, including Pelosi. “What I\'m concerned about is the Republican leadership in the House of Representatives, who is willing to overlook, ignore those statements,” Pelosi said at her weekly press briefing. Greene, who represents Georgia\'s 14th Congressional District, is infamous for her support of the QAnon conspiracy, which claims that a cabal of Democrats and celebrities are pedophiles who collude against conservatives and former President Trump. In a January, 2019 post, before she was elected to Congress, Greene liked a comment reading, “a bullet to the head would be quicker” to remove Pelosi, according to a CNN KFile deep dive on Greene\'s Facebook page. Greene also liked posts referring to executing FBI agents. In a statement posted to Twitter, Greene claimed that “over the years, I\'ve had teams of people manage my pages.” “Many posts have been liked. Many posts have been shared. Some did not represent my views,” Greene wrote, complaining that “CNN hasn\'t once tried to cancel a Democrat,” not even “those who called for violence while in office.” Pelosi also slammed GOP leadership for assigning Greene to the House Education Committee in light of a video of Greene confronting David Hogg, one of the student survivors of the Parkland school shooting. Social media comments recently surfaced showing Greene agreeing with people online who said the 2018 mass shooting was a planned “false flag” event. “It\'s absolutely appalling, and I think the focus has to be on the Republican leadership of this House of Representatives for the disregard they have for the death of those children,” she said. GOP Minority Whip Steve Scalise, who was shot in 2017 by a left-wing activist, has come out against Greene\'s rhetoric. “I\'ve consistently condemned the use of violent rhetoric in politics on both sides, and this is no exception. There is no place for comments like that in our political discourse,” Scalise said.'
#nltk.download('punkt')
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = sentence_tokenizer.tokenize(text)
#sentences = var.split('. ')
#print(sentences)

# Run through algorithm 
results = TestSentence.output(sentences)

# Save to file? 
print(results['article_score'])