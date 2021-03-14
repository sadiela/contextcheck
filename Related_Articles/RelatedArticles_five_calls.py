from newsapi import NewsApiClient
import json
import random
import requests


def getarticles(inp,date="03-02-2021"):
    api_key='c1ff125522de4c749e615dca5cba6eb5'
    # Init
    newsapi = NewsApiClient(api_key)
    
    pagesize = '50'
    sort='relavency'

    #centerist source call
    url1 = ('http://newsapi.org/v2/everything?'
        f'q={inp}&'
        f'from={date}&'
        f'sortBy={sort}&'
        'sources=associated-press&'
        'domains= reuters.com,cbsnews.com,abcnews.go.com,bloomburg.com,economist.com,forbes.com,cnbc.com,thehill.com,politico.com$'
        f'pageSize={pagesize}&'
        'page = 1&'
        f'apiKey={api_key}')

    response1 = requests.get(url1)
    articles1 = response1.json()

    #skew left source call
    url2 = ('http://newsapi.org/v2/everything?'
        f'q={inp}&'
        f'from={date}&'
        f'sortBy={sort}&'
        'sources=cnn&'
        'domains= nytimes.com,theguardian.com,msnbc.com,theatlantic.com,vox.com,washingtonpost.com,huffpost.com,thedailybeast.com&'
        f'pageSize={pagesize}&'
        'page = 1&'
        f'apiKey={api_key}')

    response2 = requests.get(url2)
    articles2 = response2.json()

    #partisan left source call
    url3 = ('http://newsapi.org/v2/everything?'
        f'q={inp}&'
        f'from={date}&'
        f'sortBy={sort}&'
        'domains= slate.com,jacobinmag.com,rawstory.com,progressive.org&'
        f'pageSize={pagesize}&'
        'page = 1&'
        f'apiKey={api_key}')

    response3 = requests.get(url3)
    articles3 = response3.json()

    #partisan right source call
    url4 = ('http://newsapi.org/v2/everything?'
        f'q={inp}&'
        f'from={date}&'
        f'sortBy={sort}&'
        'sources=fox-news&'
        'domains= dailywire.com,dailycaller.com,nationalreview.com&'
        f'pageSize={pagesize}&'
        'page = 1&'
        f'apiKey={api_key}')

    response4 = requests.get(url4)
    articles4 = response4.json()

    #skew right source call
    url5 = ('http://newsapi.org/v2/everything?'
        f'q={inp}&'
        f'from={date}&'
        f'sortBy={sort}&'
        'sources=the-wall-street-journal&'
        'domains= reason.com,quillette.com,realclearpolitics.com,nypost.com,washingtonexaminer.com,rasmussenreports.com,freebeacon.com,zerohedge.com&'
        f'pageSize={pagesize}&'
        'page = 1&'
        f'apiKey={api_key}')

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

    # print("pleft:", pleft)
    # print("sleft:", sleft)
    # print("middle:", middle)
    # print("sright:", sright)
    # print("pright:", pright)
    # #this removes the blank space placeholder for each array
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

    #returns array of articles, elements 0-4 is partisan left to partisan right respectively
    return related

#print(getarticles('immigration','02-01-2021'))
