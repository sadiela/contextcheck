from newspaper import Article
import lxml.html
import requests, re, json
from bs4 import BeautifulSoup
#import eval
import time
#from Related_Articles import RelatedArticles
#import config

def article_parse(url):
	if "cnn.com" in url:	
		return cnnScrape(url)
	elif "foxnews.com" in url:
		return foxScrape(url)
	elif "huffpost.com" in url:
		return huffScrape(url)
	elif "nypost.com" in url:
		return nypScrape(url)
	else:
		return genScrape(url)
	#works for AP news, progressive.org, NYpost and maybe more
	
def cnnScrape(url): #run time ~.3 seconds
	try:	
		articleReq = requests.get(url)
		cnnArticle = lxml.html.fromstring(article.content)
		article = Article(url)	
		article.download()	
	except:
		print("Invalid URL or article.\nNote: Paywalled/subscriber articles will not work")
		return "Error"
	article.parse()
	author = article.authors
	title = article.title
	date = article.publish_date

	parseText = article.text.replace("\n", " ")
	
	cnnArticle = lxml.html.fromstring(article.content)
	sourceType = cnnArticle.cssselect('meta[name="section"]')[0].get('content')
	

	try: 
		date = date.strftime("%m/%d/%Y, %H:%M:%S")
	except:
		date = "NOT FOUND"
	data = {"title": title, "author": author, "feedText": parseText, "date": date, "source": sourceType}
	return json.dumps(data)

def foxScrape(url): #run time ~.2 seconds
	article = Article(url)	
	try:
		article.download()	
	except:
		print("Invalid URL or article.\nNote: Paywalled/subscriber articles will not work")
		return "Error"
	article.parse()
	author = article.authors
	author = author[0]
	date = article.publish_date
	title = article.title
	#parseText = article.text.lower()
	parseText = article.text.replace("\n", " ") #ads scrubbing!
	parseText2 = ""
	textList = parseText.split()
	for word in textList:
		if word.isupper():
			word = ""
		parseText2 = parseText2 + word + " "
     
	#feedText = parseText.split(".")
	#for word in feedText:
	#	if word.isupper():
	#		feedText.remove(word)

	re = requests.get(url, headers = {'User-Agent':'Mozilla/5.0'})
	fox_soup = BeautifulSoup(re.text, 'html.parser')
	meta = fox_soup.find("meta", {"name":"classification-isa"})['content']
	meta = meta.replace(',', " ")
	data = {"title": title, "author": author, "feedText": parseText2, "date": date, "meta": meta}
	return json.dumps(data)
	
def huffScrape(url): #runtime ~1.2-1.4 seconds
	article = Article(url)	
	try:
		article.download()	
	except:
		print("Invalid URL or article.\nNote: Paywalled/subscriber articles will not work")
		return "Error"
	article.parse()
	huff_soup = requests.get(url, headers = {'User-Agent': 'Mozilla/5.0'})
	huff_text = BeautifulSoup(huff_soup.text, 'html.parser')
	try:
        	author = huff_text.find("div", {"class":"author-card"}).text
        	author = author.replace("\n", "")
	except:
		author = huff_text.find("div", {"class":"wire-partner-component"}).text
		author = author.replace("\n", "")
	title = article.title
	date = article.publish_date
	response = requests.get(url, headers = {'User-Agent': 'Mozilla/5.0'})
	if response.status_code != 200:
		return "FAIL TO GET URL"
	parseText = article.text
	parseText = parseText.replace("\n", " ")
	#feedText = parseText.split(".")
	try: 
		date = date.strftime("%m/%d/%Y, %H:%M:%S")
	except:
		date = "NOT FOUND"
	data = {"title": title, "author": author, "feedText": parseText, "date": date}
	return json.dumps(data)

def nypScrape(url): #runtime ~1.2-1.4 seconds
	article = Article(url)
	try:
		article.download()
		article.parse()
	except:
		print("Invalid URL or article.\nNote: Paywalled/subscriber articles will not work")
		return "Error"
	nyp_soup = requests.get(url, headers = {'User-Agent': 'Mozilla/5.0'})
	nyp_text = BeautifulSoup(nyp_soup.text, 'html.parser')

	author = nyp_text.find("div", {"id":"author-byline"})
	author = author.find("p", {"class":"byline"}).text
	author = author.replace("\n", "")

	title = article.title
	date = article.publish_date
	response = requests.get(url, headers = {'User-Agent': 'Mozilla/5.0'})
	if response.status_code != 200:
		return "FAIL TO GET URL"
	parseText = article.text
	parseText = parseText.replace("\n", " ")
	#feedText = parseText.split(".")
	try: 
		date = date.strftime("%m/%d/%Y, %H:%M:%S")
	except:
		date = "NOT FOUND"
	data = {"title": title, "author": author, "feedText": parseText, "date": date}
	return json.dumps(data)
	
def genScrape(url):
	article = Article(url)
	try:
		article.download()
		article.parse()
	except:
		print("Invalid URL or article.\nNote: Paywalled/subscriber articles will not work")
		return "Error"
	article.parse()
	parseText = article.text.replace("\n", "")
	author = article.authors
	title = article.title
	date = article.publish_date
	try: 
		date = date.strftime("%m/%d/%Y, %H:%M:%S")
	except:
		date = "NOT FOUND"
	data = {"title": title, "author": author, "feedText": parseText, "date": date}
	return json.dumps(data)

