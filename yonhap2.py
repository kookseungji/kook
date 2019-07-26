import scrapy
import re
from datetime import timedelta, date
from urllib import parse
import time
import random
from time import sleep
# https://search.naver.com/search.naver?
# &where=news
# &query=%EA%B8%88%EB%A6%AC
# &nso=so:r,p:from20050101to20050102,a:all
# &start=1

start_date = date(2010, 1, 1)
end_date = date(2014, 12, 28)
cnt_per_page = 10
keyword = "금리"
url_format = 'https://search.naver.com/search.naver?&where=news&query={0}&start={1}&nso=so:r,p:from{2}to{2},a:all'



item = {}
class NewscrawlSpider(scrapy.Spider):
    def date_range(start_date, end_date):
        for i in range(int ((end_date - start_date).days)+1):
            yield start_date + timedelta(i)

    name = 'yonhap2'
    start_urls = []

    for single_date in date_range(start_date, end_date):
        start_urls.append(url_format.format(keyword,1,single_date.strftime('%Y%m%d')))
        

    # allowed_domains = ['naver.com']
    def parse(self, response):
        for news in response.css('dl'):  
            title = news.css('dd span._sp_each_source::text').get()

            if title == '연합뉴스':
                link = news.css('dd a._sp_each_url::attr(href)').get()

                yield response.follow(link, self.parse_detail)
            else:
                pass

            ############################
            ##### 관련뉴스 수집하기 #####
            ############################
            title2 = news.css('span.press::text').getall()

            if len(title2) > 0 :
                for idx, val in enumerate(title2):
                    if val == '연합뉴스': 
                        link = news.css('ul.relation_lst li a::attr(href)').getall()
                        # print('***************************', link)
                        link = link[2*idx]
                        yield response.follow(link, self.parse_detail) 
                    else:
                        pass
            else:
                pass

        total_cnt = int(re.sub('건', '', response.css('div.title_desc.all_my span::text').get().split('/')[1])) 
        query_str = parse.parse_qs(parse.urlsplit(response.url).query)   # url에서 query 분해해서 저장하는 변수
        currpage = int(query_str['start'][0])
    

        startdate = query_str['nso'][0]
        print("=================== [" + startdate + '] ' + str(currpage) + '/' + str(total_cnt) + "===================") 
        ################################################
        ###########    TIME ############################
        ################################################
        sleep(0.5)
        ################################################
        ################################################
        if currpage  < total_cnt : 
            yield response.follow(url_format.format(keyword, currpage+10, startdate) , self.parse)
            

    def parse_detail(self, response):
        table = response.css('div.content')
        
        press = table.css('div.press_logo a img::attr(title)').get()
        title = table.css('h3::text').get()
        date = table.css('span.t11::text').get().split(' ')[0].replace('.','')
        link = response.url
        content = str(table.xpath('//div[@id="articleBodyContents"]/text()').getall())
        content = re.sub(' +', ' ', str(re.sub(re.compile('<.*?>'), ' ', content.replace('"','')).replace('\r\n','').replace('\n','').replace('\t','').replace('\u200b','').replace('\\n\\t','').replace('\\n\\n','').replace('\\n','').replace('\\n\\t\\n\\t','').replace('\\n\\n\\t','').replace(' ','').strip()))
        

        item['press'] = press
        item['date'] = date
        item['title'] = title 
        item['link'] = link
        item['content'] = content

        yield item

