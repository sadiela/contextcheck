from newspaper import Article
import lxml.html
import requests, re, json
from bs4 import BeautifulSoup
#import eval
import time

def article_parse(url):
	article = Article(url)	
	try:
		article.download()	
	except:
		print("Invalid URL or article.\nNote: Paywalled/subscriber articles will not work")
		return "Error"
		
	if "cnn.com" in url:	
		article.parse()
		author = article.authors
		tite = article.title
		parseText = article.text.lower()
		parseText = parseText.replace("\n", " ")
		parseText = parseText.replace("\"", "")
		feedText = parseText.split(".")
		return title, author, feedText

	elif "foxnews.com" in url:
		article.parse()
		author = article.authors
		author = author[0]
		title = article.title
		parseText = article.text.lower()
		parseText = parseText.replace("\n", " ")
		parseText = parseText.replace("\"", "")
		feedText = parseText.split(".")
		for word in feedText:
			if word.isupper():
				feedText.remove(word)
		return title, author, feedText
	elif "huffpost.com" in url:
		article.download()
		article.parse()
		huff_soup = requests.get(url, headers = {'User-Agent': 'Mozilla/5.0'})
		huff_text = BeautifulSoup(huff_soup.text, 'html.parser')
		author = huff_text.find("div", {"class":"author-list"}).text
		title = article.title
		response = requests.get(url, headers = {'User-Agent': 'Mozilla/5.0'})
		if response.status_code != 200:
			return "FAIL TO GET URL"
		#article author does not work, utilize lxml
		
		parseText = article.text.lower()
		parseText = parseText.replace("\n", " ")
		parseText = parseText.replace("\"", "")
		feedText = parseText.split(".")
		return author, [feedText], title
	elif "twitter.com" in url:
		bearer_token = "AAAAAAAAAAAAAAAAAAAAAJFHKgEAAAAAfBSf9znUqolOrWjG%2FvOu1PEhTI0%3DHo9Km1FgrlkW4otaX8fluQdRWJc3ItZcZg80n3xFtMiWrpEhKK"
		t_id = re.split("/status/", url)[1]
		ids = "ids="+t_id
		tweet_fields = "tweet.fields=lang,author_id"
		auth = "expansions=author_id"
		url = "https://api.twitter.com/2/tweets?{}&{}&{}".format(ids, tweet_fields, auth)
		headers = {"Authorization": "Bearer {}".format(bearer_token)}
		response = requests.request("GET", url, headers = headers)
		twext = response.json()['data'][0]['text']
		atext = response.json()['includes']['users'][0]['username']
		return twext, atext
	
	
#to do: 
# date time functionality
# subscription based...?
def main():
	tic = time.perf_counter()
	url = 'https://www.huffpost.com/entry/covid-19-eviction-crisis-women_n_5fca8af3c5b626e08a29de11'
	url2 = 'https://twitter.com/Twitter/status/1339350208942125066'
	#author, feedText, title =  article_parse(url)
	text, atext = article_parse(url2)
	print(atext)
	toc = time.perf_counter()
	print(f"\nRuntime = {toc - tic:0.4f} seconds") 

if __name__ == "__main__":
	main()


	
