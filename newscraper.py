from newspaper import Article
import lxml.html
import requests, re
from bs4 import BeautifulSoup
#import eval

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
		feedText = parseText.split(" ")
		return title, author, feedText

	elif "foxnews.com" in url:
		article.parse()
		author = article.authors
		author = author[0]
		title = article.title
		parseText = article.text.lower()
		parseText = parseText.replace("\n", " ")
		parseText = parseText.replace("\"", "")
		feedText = parseText.split(" ")
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
		feedText = parseText.split(" ")
		return author, [feedText], title
	elif "twitter.com" in url:
		print("functionality not implemented/awaiting dev account API key");
		return "Error"
	
	
#to do: 
# date time functionality
# twitter API
# huffpost Author issue (403 blocked)
# subscription based...?
		
url = 'https://www.huffpost.com/entry/covid-19-eviction-crisis-women_n_5fca8af3c5b626e08a29de11'
author, feedText, title =  article_parse(url)
print(feedText)


	
