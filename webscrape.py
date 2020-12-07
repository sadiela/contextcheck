import lxml.html
import requests, re
import eval

cnnLink = "https://www.cnn.com/2020/11/16/politics/election-2020-donald-trump-joe-biden-transition-coronavirus/index.html"
article  = requests.get(cnnLink)
cnnArticle = lxml.html.fromstring(article.content)

#title
title = cnnArticle.xpath('//h1')
#metadata
metadata = cnnArticle.xpath('//div[@class="metadata__info js-byline-images"]')

#para1 focuses on the starting paragraph denoted by (CNN)
#para2 are the rest of the paragraphs that seem to be denoted with voice search optimization
#para3 are the rest of the paragraphs that aren't involved in voice search optimization

#articles confirmed working: 
#https://www.cnn.com/2020/11/15/politics/trump-administration-china-biden/index.html
#https://www.cnn.com/2020/11/16/politics/election-2020-donald-trump-joe-biden-transition-coronavirus/index.html

para1 = cnnArticle.xpath('//p[@class="zn-body__paragraph speakable"]')
para2 = cnnArticle.xpath('//div[@class="zn-body__paragraph speakable"]')
para3 = cnnArticle.xpath('//div[@class="zn-body__paragraph"]')
fullArticle = ''
fullArticle += para1[0].text_content()
for paragraph in para2:
    fullArticle += paragraph.text_content()
for paragraph in para3:
    fullArticle += paragraph.text_content()

parsedArticle = re.split(r'[.]',fullArticle.lower())

for sentence in parsedArticle:
	out, length = eval.test_sentence(sentence)
	print(sentence)
	for l in out['tok_probs'][0][:length]:
		avg_sum += l[1]
	print("Average bias: ", avg_sum/length)

