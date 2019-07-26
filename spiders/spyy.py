# -*- coding: utf-8 -*-
import scrapy
import re
from datetime import timedelta, date
start_date = date(2017, 1, 1)
end_date = date(2017, 1, 2)
cnt_per_page = 10
item={}
keyword = "마라탕"
url_format = "https://search.naver.com/search.naver?date_from={0}&date_option=8&date_to={0}&dup_remove=1&nso=so%3Add%2Cp%3Afrom{0}to{0}&post_blogurl=&post_blogurl_without=&query={1}&sm=tab_pge&srchby=all&st=date&where=post&start={2}"

class SpyySpider(scrapy.Spider):
    def daterange(start_date, end_date):
        for n in range(int ((end_date - start_date).days)):
            yield start_date + timedelta(n)
    name = 'spyy'
    allowed_domains = ['naver.com']
    start_urls = []
       
    for single_date in daterange(start_date, end_date):
        start_urls.append(url_format.format(single_date.strftime("%Y%m%d"), keyword, 1))

    def parse(self, response):
             
        for qu in response.css('li.sh_blog_top'):
            link=qu.css('a::attr(href)').get()
            next_page = link
            if next_page is not None:
                yield response.follow(next_page, callback=self.pse)         
    def pse(self, response):   
        iframe=response.css('iframe::attr(src)').get()
        yield response.follow(iframe, callback=self.pe)
    def pe(self, response):   
        yield {
               'content': response.css('span.se-fs-.se-ff-::text').getall(),
               'date': response.css('span.se_publishDate.pcol2::text').get(),
               'id': response.css('span.nick a::text').get()
                }
                 
        if 'blog.naver.com' in response.url :
            #title = str(response.xpath("//div[@class='se-module se-module-text se-title-text']/p/span/text()").get())
            #item['date'] = response.xpath("//span[contains(@class, 'se_publishDate')]/text()").get()
            content = str(response.xpath("//div[@class='se-main-container']").get())

            if content == 'None' :
                title = str(response.xpath("//div[contains(@class,'se_title')]//h3").get())
                item['date'] = response.xpath("//span[contains(@class, 'se_publishDate')]/text()").get()
                content = str(response.xpath("//div[contains(@class, 'sect_dsc')]").get())

            if content == 'None' :
                title = str(response.xpath("//div[@class='htitle']/span/text()").get())
                item['date'] = response.xpath("//p[contains(@class,'_postAddDate')]/text()").get()
                content = str(response.xpath("//div[@id='postViewArea']/div").get())

        title = re.sub(' +', ' ', str(re.sub(re.compile('<.*?>'), ' ', title.replace('"','')).replace('\r\n','').replace('\n','').replace('\t','').replace('\u200b','').strip()))
        content = re.sub(' +', ' ', str(re.sub(re.compile('<.*?>'), ' ', content.replace('"','')).replace('\r\n','').replace('\n','').replace('\t','').replace('\u200b','').replace('\xa0','').strip()))
        item['title'] = title 
        item['content'] = content  

        yield item