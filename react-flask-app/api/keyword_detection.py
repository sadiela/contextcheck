import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import tokenize
import math

from operator import itemgetter
import json 

def get_top_n(dict_elem, n):
    result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)) 
    words = result.keys() 
    filtered_result = [i for i in words if i.isalpha() ]
    return filtered_result[:n]

def check_sent(word, sentences): 
    final = [all([w in x for w in word]) for x in sentences] 
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_len))

def get_keywords(body_text):
    nltk.download('stopwords')
    # read an article 
    #print(res)

    stop_words = set(stopwords.words('english'))

    total_words = body_text.split()
    total_word_length = len(total_words)

    total_sentences = tokenize.sent_tokenize(body_text)
    total_sent_len = len(total_sentences)

    tf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word not in stop_words:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1

    # Dividing by total_word_length for each dictionary element
    tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())

    idf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word not in stop_words:
            if each_word in idf_score:
                idf_score[each_word] = check_sent(each_word, total_sentences)
            else:
                idf_score[each_word] = 1

    # Performing a log and divide
    idf_score.update((x, math.log(int(total_sent_len)/y)) for x, y in idf_score.items())


    tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
    
    keywords = (get_top_n(tf_idf_score, 3))
    return keywords

    # citation: https://www.analyticsvidhya.com/blog/2020/11/words-that-matter-a-simple-guide-to-keyword-extraction-in-python/