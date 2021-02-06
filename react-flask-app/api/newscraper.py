from newspaper import Article
import lxml.html
import requests, re, json
from bs4 import BeautifulSoup
#import eval
import time
#import config

def article_parse(url):
	if "cnn.com" in url:	
		return cnnScrape(url)
	elif "foxnews.com" in url:
		return foxScrape(url)
	elif "huffpost.com" in url:
		return huffScrape(url)
	elif "twitter.com" in url:
		return tweetScrape(url)
	else:
		return "Invalid or unsupported URL"
	
def cnnScrape(url): #run time ~.3 seconds
	article = Article(url)	
	try:
		article.download()	
	except:
		print("Invalid URL or article.\nNote: Paywalled/subscriber articles will not work")
		return "Error"
	article.parse()
	author = article.authors
	title = article.title
	date = article.publish_date
	parseText = article.text.lower()
	parseText = parseText.replace("\n", " ")
	parseText = parseText.replace("\"", "")
	feedText = parseText.split(".")
	date = date.strftime("%m/%d/%Y, %H:%M:%S")
	data = {"title": title, "author": author, "feedText": feedText, "date": date}
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
	date = 0
	title = article.title
	parseText = article.text.lower()
	parseText = parseText.replace("\n", " ")
	parseText = parseText.replace("\"", "")
	feedText = parseText.split(".")
	for word in feedText:
		if word.isupper():
			feedText.remove(word)
	re = requests.get(url, headers = {'User-Agent':'Mozilla/5.0'})
	fox_soup = BeautifulSoup(re.text, 'html.parser')
	meta = fox_soup.find("meta", {"name":"classification-isa"})['content']
	meta = meta.split(',')
	data = {"title": title, "author": author, "feedText": feedText, "date": date, "meta": meta}
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
	author = huff_text.find("div", {"class":"author-list"}).text
	title = article.title
	date = article.publish_date
	response = requests.get(url, headers = {'User-Agent': 'Mozilla/5.0'})
	if response.status_code != 200:
		return "FAIL TO GET URL"
	#article author does not work, utilize lxml
	
	parseText = article.text.lower()
	parseText = parseText.replace("\n", " ")
	parseText = parseText.replace("\"", "")
	feedText = parseText.split(".")
	return title, author, [feedText], date
'''
def tweetScrape(url): #runtime sub .4 seconds
	bearer_token = config.bearer
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
	
'''
def main():
	tic = time.perf_counter()
	url = 'https://www.huffpost.com/entry/covid-19-eviction-crisis-women_n_5fca8af3c5b626e08a29de11'
	url2 = 'https://twitter.com/Twitter/status/1339350208942125066'
	url3 = 'https://www.foxnews.com/politics/ilhan-omar-slams-lawmakers-vaccine'
	url4 = 'https://www.cnn.com/2020/12/21/politics/bidens-coronavirus-vaccination/index.html'
	#author, feedText, title =  article_parse(url)
	author, title, text, date, meta = article_parse(url3)
	print(author)
	print(text)
	print(title)
	print(date)
	print(meta)
	
	toc = time.perf_counter()
	print(f"\nRuntime = {toc - tic:0.4f} seconds") 

if __name__ == "__main__":
	main()


	
