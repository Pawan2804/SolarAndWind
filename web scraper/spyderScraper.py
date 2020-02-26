import asyncio
import json
import aiohttp

originalurl = "https://api.weather.com/v1/location/VOBL:9:IN/observations/historical.json?apiKey=6532d6454b8aa370768e63d6ba5a832e&units=e&startDate=20100101&endDate=20100131"
baseurl = "https://api.weather.com/v1/location/VOBL:9:IN/observations/historical.json?apiKey=6532d6454b8aa370768e63d6ba5a832e&units=e&startDate="

import datetime

start = datetime.datetime.strptime("01-01-2010", "%d-%m-%Y")
end = datetime.datetime.strptime("01-01-2020", "%d-%m-%Y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]

urlList = []
for date in date_generated:
    start = date.strftime("%Y%m%d")
    url = baseurl+start
    urlList.append(url)

data = []

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        texts = await asyncio.gather(*[
            fetch(session, url)
            for url in urls
        ])
        data.append(texts)
        return texts
async def aim():
  result = await fetch_all(urlList) 

aim()
print(data[0])