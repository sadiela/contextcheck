import sys
sys.path.append('c:\python38\lib\site-packages')
sys.path.append('c:\\users\\sadie\\appdata\\roaming\\python\\python38\\site-packages')
from pathlib import Path
import json
import TestSentence
import time
import string
import nltk.data
from os import listdir 
from os.path import isfile, join
from copy import copy
import newscraper
import argparse


def get_free_filename(stub, directory, suffix=''):
    counter = 0
    while True:
        file_candidate = '{}/{}-{}{}'.format(
            str(directory), stub, counter, suffix)
        if Path(file_candidate).exists():
            print("file exists")
            counter += 1
        else:  # No match found
            print("no file")
            if suffix:
                Path(file_candidate).touch()
            else:
                Path(file_candidate).mkdir()
            return file_candidate

def analyze_sentences(text):
    #nltk.download('punkt')
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_tokenizer.tokenize(text)

    # Run through algorithm 
    results = TestSentence.output(sentences)
    print('ARTICLE RESULTS:', results['article_score'])

    return results

# Take list of URLS --> if web scraper working
# Assume single text file with urls separated by newlines
#url_filename = "../../testing/urls.txt" # include directory
#with 
def get_url_results(filepath, results_folder, results_filename="url"):
    url_result_path = get_free_filename(results_filename + '_results', results_folder, suffix='.json')
    full_url_result_path = get_free_filename(results_filename + '_full_results', results_folder, suffix='.json')
    url_list = []
    url_results = []
    full_url_results = []
    with open(filepath, encoding='utf-8') as file: 
        lines = file.readlines()
    for line in lines: 
        url_list.append(line.rstrip())
    print("URLS:", url_list)
    
    for url in url_list:
        try:
            article_results = {"url": url, "article_score":0}
            full_article_results = {"url": url, "results":0}
            cur_scrape = newscraper.article_parse(url)
            scrape_obj = json.loads(cur_scrape) 
            cur_results = analyze_sentences(scrape_obj['feedText'])

            article_results['article_score'] = cur_results['article_score']
            full_article_results['results'] = cur_results

            url_results.append(copy(article_results))
            full_url_results.append(copy(full_article_results))
            # save spec results and full_spec results to results folder
        except:
            print("URL parsing error:", url)

    with open(url_result_path, 'w') as fp:
        json.dump(url_results, fp, indent=4)
    with open(full_url_result_path, 'w') as fp:
        json.dump(full_url_results, fp, indent=4)
    print("DONE")


    
# OR
# Take directory with text files
# Return list of bias scores
def get_text_results(directory, results_folder, results_filename="text"):
    dir_results_path = get_free_filename(results_filename + '_results', results_folder, suffix='.json')
    full_dir_results_path = get_free_filename(results_filename + '_full_results', results_folder, suffix='.json')
    #spec_results_path = results_folder + spec + ".json"
    #full_spec_results_path = results_folder + spec + "_full.json"
    dir_results = []
    full_dir_results = []
    cur_spec_files = [f for f in listdir(directory) if isfile(join(directory, f))]
    for f in cur_spec_files: 
        article_results = {"filename": f, "article_score":0}
        full_article_results = {"filename": f, "results":0}
        # read in the text
        print(f)
        with open(directory + '/' + f, 'r', encoding='utf-8') as file: 
            article_text = file.read().replace('\n', ' ')

        testsentence_results = analyze_sentences(article_text)

        article_results['article_score'] = testsentence_results['article_score']
        full_article_results['results'] = testsentence_results

        dir_results.append(copy(article_results))
        full_dir_results.append(copy(full_article_results))
        # save spec results and full_spec results to results folder
        with open(dir_results_path, 'w') as fp:
            json.dump(dir_results, fp, indent=4)
        with open(full_dir_results_path, 'w') as fp:
            json.dump(full_dir_results, fp, indent=4)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    results_folder = "../../testing/results/"
    default_directory_name = "../../testing/Article_Txt_Files/"

    #-db DATABASE -u USERNAME -p PASSWORD -size 20000
    parser.add_argument("-t", "--type", dest = "flag_arg", default = "text", help="dir of TEXT files or file of URLs?")
    parser.add_argument('-d', "--directoryorfile", dest="path_arg", default = default_directory_name, help="path of dir/file you want to analyze")
    parser.add_argument('-r', "--result_filename", dest="results_fname", default = "pipeline_run", help="name for your results file(s)")

    args = parser.parse_args()

    if args.flag_arg =="url":
        get_url_results(args.path_arg, results_folder, args.results_fname)
    elif args.flag_arg == "text":
        get_text_results(args.path_arg, results_folder, args.results_fname)
