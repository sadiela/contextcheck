from newsapi import NewsApiClient
import json
import random
import requests

# Init
newsapi = NewsApiClient(api_key='c1ff125522de4c749e615dca5cba6eb5')

def getarticles(inp):
       

       url = ('http://newsapi.org/v2/everything?'
              f'q={inp}&'
              #'from=2021-01-02&'
              'sortBy=date&'
              'sources=fox-news, associated-press, cnn, the-wall-street-journal&'
              'domains= slate.com,dailywire.com,reuters.com,cbsnews.com,abcnews.go.com,bloomburg.com,economist.com,forbes.com,cnbc.com,thehill.com,politico.com,nytimes.com,theguardian.com,msnbc.com,theatlantic.com,vox.com,washingtonpost.com,huffpost.com,thedailybeast.com,reason.com,quillette.com,realclearpolitics.com,nypost.com,washingtonexaminer.com,rasmussenreports.com,freebeacon.com,zerohedge.com,jacobinmag.com,rawstory.com,progressive.org,slate.com,dailycaller.com,nationalreview.com&'
              'pageSize=100&'
              'apiKey=c1ff125522de4c749e615dca5cba6eb5')

       response = requests.get(url)
       articles = response.json()
       #print(articles)

       news_urls = ['']
       middle = ['']
       sleft = ['']
       sright = ['']
       pleft = ['']
       pright = ['']

       for x in articles['articles']:
              news_urls.append(x['url'])

       #print(news_urls)

       #centristsources
       ap =  'apnews.com'
       re = 'reuters.com'
       cb = 'cbsnews.com'
       ab = 'abcnews.go.com'
       bl = 'bloomburg.com'
       ec = 'economist.com'
       fo = 'forbes.com'
       cc = 'cnbc.com'
       hi = 'thehill.com'
       po = 'politico.com'

       #skew left
       cn = 'cnn.com'
       nt = 'nytimes.com'
       gu = 'theguardian.com'
       ms = 'msnbc.com'
       al = 'theatlantic.com'
       vo = 'vox.com'
       wp = 'washingtonpost.com'
       hp = 'huffpost.com'
       db = 'thedailybeast.com'

       #skew right
       ws = 'wsj.com'
       rs = 'reason.com'
       qu = 'quillette.com'
       rc = 'realclearpolitics.com'
       np = 'nypost.com'
       we = 'washingtonexaminer.com'
       rr = 'rasmussenreports.com'
       fb = 'freebeacon.com'
       zh = 'zerohedge.com'

       #partisan left
       jb = 'jacobinmag.com'
       ra = 'rawstory.com'
       pr = 'progressive.org'
       sl = 'slate.com'

       #partisan right
       dw ='dailywire.com'
       fn = 'foxnews.com'
       dc = 'dailycaller.com'
       nr = 'nationalreview.com'


       for i in range(len(news_urls)):
       #partisan right
              if dw in news_urls[i]:
                     pright.append(news_urls[i])
              if fn in news_urls[i]:
                     pright.append(news_urls[i])
              if dc in news_urls[i]:
                     pright.append(news_urls[i])
              if nr in news_urls[i]:
                     pright.append(news_urls[i])
       #skew left
              if cn in news_urls[i]:
                     sleft.append(news_urls[i])
              if nt in news_urls[i]:
                     sleft.append(news_urls[i])
              if gu in news_urls[i]:
                     sleft.append(news_urls[i])
              if ms in news_urls[i]:
                     sleft.append(news_urls[i])
              if al in news_urls[i]:
                     sleft.append(news_urls[i])
              if vo in news_urls[i]:
                     sleft.append(news_urls[i])
              if wp in news_urls[i]:
                     sleft.append(news_urls[i])
              if hp in news_urls[i]:
                     sleft.append(news_urls[i])
              if hp in news_urls[i]:
                     sleft.append(news_urls[i])
       #skew right
              if ws in news_urls[i]:
                     sright.append(news_urls[i])
              if rs in news_urls[i]:
                     sright.append(news_urls[i])
              if qu in news_urls[i]:
                     sright.append(news_urls[i])
              if rc in news_urls[i]:
                     sright.append(news_urls[i])
              if np in news_urls[i]:
                     sright.append(news_urls[i])
              if we in news_urls[i]:
                     sright.append(news_urls[i])
              if rr in news_urls[i]:
                     sright.append(news_urls[i])
              if fb in news_urls[i]:
                     sright.append(news_urls[i])
              if zh in news_urls[i]:
                     sright.append(news_urls[i])
       #partisan left
              if jb in news_urls[i]:
                     pleft.append(news_urls[i])
              if ra in news_urls[i]:
                     pleft.append(news_urls[i])
              if pr in news_urls[i]:
                     pleft.append(news_urls[i])
              if sl in news_urls[i]:
                     pleft.append(news_urls[i])
       #centrist
              if ap in news_urls[i]:
                     middle.append(news_urls[i])
              if re in news_urls[i]:
                     middle.append(news_urls[i])
              if cb in news_urls[i]:
                     middle.append(news_urls[i])
              if ab in news_urls[i]:
                     middle.append(news_urls[i])
              if bl in news_urls[i]:
                     middle.append(news_urls[i])
              if ec in news_urls[i]:
                     middle.append(news_urls[i])
              if fo in news_urls[i]:
                     middle.append(news_urls[i])
              if cc in news_urls[i]:
                     middle.append(news_urls[i])
              if hi in news_urls[i]:
                     middle.append(news_urls[i])
              if po in news_urls[i]:
                     middle.append(news_urls[i])

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

       related.append(random.choice(pleft))
       related.append(random.choice(sleft))
       related.append(random.choice(middle))
       related.append(random.choice(sright))
       related.append(random.choice(pright))
   
       return related

