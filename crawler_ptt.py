import urllib.request as req
import bs4
import pandas as pd
import concurrent.futures
import time

list = []
author = 'sp89005'


def scrape(urls):
    request = req.Request(urls, headers={
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.54'
    })

    with req.urlopen(request) as response:
        data = response.read().decode('utf-8')

    root = bs4.BeautifulSoup(data, 'html.parser')
    comments = root.find_all('span', class_ ='yellow--text text--darken-2' )


    for comment in comments:
        list.append(comment.string[1:])


urls = [f"https://www.pttweb.cc/user/{author}/cvs?t=message&page={page}" for page in range(1,50)]

start_time = time.time() 
# with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#     executor.map(scrape, urls)
for url in urls:
    scrape(url)
end_time = time.time()
    
print(f"{end_time - start_time} 秒爬取 {len(urls)} 頁的文章")

dict = {'comment': list}
df = pd.DataFrame(dict)
df.to_csv('/home/huiyu8794/pp_final/data/crawler_result.csv')