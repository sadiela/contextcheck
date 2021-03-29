from newspaper import Article
import lxml.html
import requests, re, json
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
	try: #requests can fail if URL is not correct, possibly unnecessary
		article_R = requests.get(url, headers = {'User-Agent':'Mozilla/5.0'})
		if article_R.status_code != 200:
			return "INVALID URL/ARTICLE (possibly unsupported)"
		else:
			cnnArticle = lxml.html.fromstring(article_R.content)
	except:
		print("Invalid URL or article.\nNote: Paywalled/subscriber articles will not work")
		return "Error"
 
	article = Article(url)
	article.download()
	try:
		article.parse()
	except:
		return "INVALID URL/ARTICLE (possibly unsupported)"
 
	author = article.authors
	title = article.title
	date = article.publish_date
	text = article.text
	sourceType = cnnArticle.cssselect('meta[name="section"]')[0].get('content')
 
	try: 
		date = date.strftime("%m/%d/%Y, %H:%M:%S")
	except:
		date = "NOT FOUND"
 
	data = {"title": title, "author": author, "feedText": text, "date": date, "sourceType": sourceType}
	return data

def foxScrape(url):
	try: #requests can fail if URL is not correct, possibly unnecessary
		article_R = requests.get(url, headers = {'User-Agent':'Mozilla/5.0'})
		if article_R.status_code != 200:
			return "INVALID URL/ARTICLE (possibly unsupported)"
		else:
			article_H = lxml.html.fromstring(article_R.content)
	except:
		return "INVALID URL/ARTICLE (possibly unsupported)"
	#download and parse article
	article = Article(url)
	article.download()
	try:
		article.parse()
	except:
		return "INVALID URL/ARTICLE (possibly unsupported)"
	#begin gathering text and title
	text = article.text
	title = article.title
	#scrub text-based advertisements
	text = re.sub(r'[.,-_\']*[A-Z]+(?![a-z])[.,-_\']*',"", text)
 
	#LXML for dates/author
	author = article.authors
	if len(author) == 0:
		try:
			author = article_H.cssselect('div[class="author-byline"]')[0].text_content()
		except:
			author = "NOT FOUND"
 
	date = article_H.cssselect('div[class="article-date"]')[0].text_content()
	try:
		sourceType = article_H.cssselect('meta[name="classification-isa"]')[0].attrib['content']
	except:
		sourceType = "not found"
 
	data = {"title": title, "author": author, "feedText": text, "date": date, "sourceType": sourceType}
	return data

def huffScrape(url):
	try: #requests can fail if URL is not correct, possibly unnecessary
		article_R = requests.get(url, headers = {'User-Agent':'Mozilla/5.0'})
		if article_R.status_code != 200:
			return "INVALID URL/ARTICLE (possibly unsupported)"
		else:
			article_H = lxml.html.fromstring(article_R.content)
	except:
		return "INVALID URL/ARTICLE (possibly unsupported)"
	#download and parse article
	article = Article(url)
	article.download()
	try:
		article.parse()
	except:
		return "INVALID URL/ARTICLE (possibly unsupported)"
	#LXML for authors
	try:
		author = article_H.cssselect('div[class="author-card__name"]')[0].text_content()
	except:
		author = "NOT FOUND"
	text = article.text
	title = article.title
	date = article.publish_date
	try: 
		date = date.strftime("%m/%d/%Y, %H:%M:%S")
	except:
		date = "NOT FOUND"
	sourceType = article_H.cssselect('meta[property="article:section"]')[0].get('content')
	data = {"title": title, "author": author, "feedText": text, "date": date, "sourceType":sourceType}
	return data

def nypScrape(url):
	try: #requests can fail if URL is not correct, possibly unnecessary
		article_R = requests.get(url, headers = {'User-Agent':'Mozilla/5.0'})
		if article_R.status_code != 200:
			return "INVALID URL/ARTICLE (possibly unsupported)"
		else:
			article_H = lxml.html.fromstring(article_R.content)
	except:
		return "INVALID URL/ARTICLE (possibly unsupported)"
	#download and parse article
	article = Article(url)
	article.download()
	try:
		article.parse()
	except:
		return "INVALID URL/ARTICLE (possibly unsupported)"
	#LXML for authors
	try:
		author = article_H.cssselect('p[class="byline"]')[0].text_content()
	except:
		author = "NOT FOUND"
	text = article.text
	title = article.title
	date = article.publish_date
	try: 
		date = date.strftime("%m/%d/%Y, %H:%M:%S")
	except:
		date = "NOT FOUND"
	sourceType = article_H.cssselect('meta[name="nypost-section"]')[0].get('content')
 
	if article_H.cssselect('meta[property="article:opinion"]')[0].get('content') == "false":
		sourceType = "Opinion " + sourceType
 
        #do nothing
 
	data = {"title": title, "author": author, "feedText": text, "date": date, "sourceType":sourceType}
	return data

def genScrape(url):
	try: #requests can fail if URL is not correct, possibly unnecessary
		article_R = requests.get(url, headers = {'User-Agent':'Mozilla/5.0'})
		if article_R.status_code != 200:
			return "INVALID URL/ARTICLE (possibly unsupported)"
		else:
			article_H = lxml.html.fromstring(article_R.content)
	except:
		return "INVALID URL/ARTICLE (possibly unsupported)"
	#download and parse article
	article = Article(url)
	article.download()
	try:
		article.parse()
	except:
		return "INVALID URL/ARTICLE (possibly unsupported)"
		author = article.authors
	if len(author) == 0:
		author = "NOT FOUND"
	text = article.text
	title = article.title
	date = article.publish_date
    
	try: 
		date = date.strftime("%m/%d/%Y, %H:%M:%S")
	except:
		date = "NOT FOUND"
    
	try:
		if article_H.cssselect('meta[property="article:opinion"]')[0].get('content') == "true":
			sourceType = "Opinion"
		else:
			sourceType = "Non-Opinion"
	except:
		sourceType = "Not found"
    
	data = {"title": title, "author": author, "feedText": text, "date": date, "sourceType": sourceType}
	return data
