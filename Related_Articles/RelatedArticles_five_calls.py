from newsapi import NewsApiClient
import json
import random
# Init
newsapi = NewsApiClient(api_key='c1ff125522de4c749e615dca5cba6eb5')
import requests

#this is key word input
inp = input('\nEnter key word or begining of headline:')

#centerist source call
url1 = ('http://newsapi.org/v2/everything?'
       f'q={inp}&'
       #'from=2021-01-07&'
       'sortBy=date&'
       'sources=associated-press&'
       'domains= reuters.com,cbsnews.com,abcnews.go.com,bloomburg.com,economist.com,forbes.com,cnbc.com,thehill.com,politico.com$'
       'page = 1&'
       'apiKey=c1ff125522de4c749e615dca5cba6eb5')

response1 = requests.get(url1)
articles1 = response1.json()

#skew left source call
url2 = ('http://newsapi.org/v2/everything?'
       f'q={inp}&'
       #'from=2021-01-07&'
       'sortBy=date&'
       'sources=cnn&'
       'domains= nytimes.com,theguardian.com,msnbc.com,theatlantic.com,vox.com,washingtonpost.com,huffpost.com,thedailybeast.com&'
       'pageSize=100&'
       'page = 1&'
       'apiKey=c1ff125522de4c749e615dca5cba6eb5')

response2 = requests.get(url2)
articles2 = response2.json()

#partisan left source call
url3 = ('http://newsapi.org/v2/everything?'
       f'q={inp}&'
       #'from=2021-01-07&'
       'sortBy=date&'
       'domains= slate.com,jacobinmag.com,rawstory.com,progressive.org&'
       'pageSize=100&'
       'page = 1&'
       'apiKey=c1ff125522de4c749e615dca5cba6eb5')

response3 = requests.get(url3)
articles3 = response3.json()

#partisan right source call
url4 = ('http://newsapi.org/v2/everything?'
       f'q={inp}&'
       #'from=2021-01-07&'
       'sortBy=date&'
       'sources=fox-news&'
       'domains= dailywire.com,dailycaller.com,nationalreview.com&'
       'pageSize=100&'
       'page = 1&'
       'apiKey=c1ff125522de4c749e615dca5cba6eb5')

response4 = requests.get(url4)
articles4 = response4.json()

#skew right source call
url5 = ('http://newsapi.org/v2/everything?'
       f'q={inp}&'
       #'from=2021-01-07&'
       'sortBy=date&'
       'sources=the-wall-street-journal&'
       'domains= reason.com,quillette.com,realclearpolitics.com,nypost.com,washingtonexaminer.com,rasmussenreports.com,freebeacon.com,zerohedge.com&'
       'pageSize=100&'
       'page = 1&'
       'apiKey=c1ff125522de4c749e615dca5cba6eb5')

response5 = requests.get(url5)
articles5 = response5.json()

#politcal catagory arrays
middle = ['']
sleft = ['']
sright = ['']
pleft = ['']
pright = ['']


#all urls being sorted from json files into political catagory arrays
for x in articles1['articles']:
       middle.append(x['url'])

for x in articles2['articles']:
       sleft.append(x['url'])

for x in articles3['articles']:
       pleft.append(x['url'])

for x in articles4['articles']:
       pright.append(x['url'])

for x in articles5['articles']:
       sright.append(x['url'])

#this removes the blank space placeholder for each array
related = []
if len(pleft) > 1:
       pleft.remove('')
if len(sleft) > 1:
       sleft.remove('')
if len(middle) > 1:
       middle.remove('')
if len(sright) > 1:
       sright.remove('')
if len(pright) > 1:
       pright.remove('')

#random selection for each political catagory
related.append(random.choice(pleft))
related.append(random.choice(sleft))
related.append(random.choice(middle))
related.append(random.choice(sright))
related.append(random.choice(pright))

#print of urls and catagories
print("\n\nRelated Articles:\n")
print("Ad Fontes Media found this source politically:| URL:")
print("____________________________________________________\n")
print("Partisan Left                                 |",related[0],"\n")
print("Skews Left                                    |",related[1],"\n")
print("Centerist                                     |",related[2],"\n")
print("Skews Right                                   |",related[3],"\n")
print("Partisan Right                                |",related[4],"\n")
print("(The political leanings of each source is from Ad Fontes Media's 'Media Bias Chart 7.0 January 2021 Edition')\n\n")