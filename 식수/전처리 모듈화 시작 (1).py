#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def z_score_normalize(lst):
    normalized = []
    for value in lst:
        normalized_num = (value - np.mean(lst)) / np.std(lst)
        normalized.append(normalized_num)
    return normalized


# In[619]:


import pandas as pd
prev=pd.read_csv('20170101-20200615_original.csv')
target=pd.DataFrame({'timestamp':['2020-06-10','2020-06-11','2020-06-12','2020-06-13','2020-06-14','2020-06-15','2020-06-16'],'target_value':[100,200,300,400,500,600,700]})


# In[620]:


prev.tail()


# In[621]:


#prev=06/15 까지 채운 데이터프레임 
##오늘(6월 16일)부터 10일 간의 데이터 채워지는 함수입니다
def make_df(prev,n,target):
    prev['timestamp']=pd.to_datetime(prev['timestamp'])
    from bs4 import BeautifulSoup
    import requests
    bam=requests.get('https://m.weather.naver.com/')  
    soup=BeautifulSoup(bam.text,'html.parser')
    content=soup.find_all('div',{'class':'weather'})
    prob=[]
    for i in range(n):
        prob.append(content[i].find_all('span',{'class':'percent'})[0].text[:-1]) 

    bam=requests.get('https://m.weather.naver.com/')
    soup=BeautifulSoup(bam.text,'html.parser')
    content=soup.find_all('div',{'class':'weekly_item_temperature'})
    temp=[]
    for i in range(n):
        temp.append(content[i].find_all('span')[3].text[6:8]) # 06/17~06/27

    import numpy as np
    from datetime import timedelta
    num=[]
    for i in range(1,n):
        num.append(int(prev.index[-1]+i))

    for i in range(n-1):    
        prev.loc[num[i]]=np.nan
        prev.loc[num[i],'timestamp']=prev.loc[num[i]-1,'timestamp']+timedelta(days=1)
        prev.loc[num[i],'day_cos']=prev.loc[num[i]-7,'day_cos']
        prev.loc[num[i],'day_sin']=prev.loc[num[i]-7,'day_sin']
        prev.loc[num[i],'weekday']=prev.loc[num[i]-7,'weekday']
        prev.loc[num[i],'dayofweek_median']=prev.loc[num[i]-7,'dayofweek_median']
        prev.loc[num[i],'dayofweek_mean']=prev.loc[num[i]-7,'dayofweek_mean']
          
    from datetime import datetime
    for i in range(n):
        prev.loc[prev['timestamp']==datetime.strptime(datetime.today().strftime("%Y-%m-%d"),"%Y-%m-%d")+timedelta(days=i),'temperature']=temp[i] 
        prev.loc[prev['timestamp']==datetime.strptime(datetime.today().strftime("%Y-%m-%d"),"%Y-%m-%d")+timedelta(days=i),'rain_prob']=int(prob[i])*0.1
    
    for i in range(len(target)):
        prev.loc[prev['timestamp']==datetime.strptime(target['timestamp'][i],"%Y-%m-%d"),'target_value'] =target['target_value'][i]
    
    url='http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo?solYear={year}&solMonth={date}&ServiceKey=J%2FYE3sN528TkttbFXsmjXDfVrpZyKNjduV1r%2FcUxAfJ7wHnrjO2F5zzkJx2JvBHsmuNstAVe%2B5n6fS2dNgcKvQ%3D%3D'.format(year=str((prev.loc[prev.index[-1],'timestamp']).year),date=str(prev.loc[prev.index[-1],'timestamp'].month))
    resp=requests.get(url)
    soup=BeautifulSoup(resp.content)
    Text=soup.find_all('locdate')
    holiday=[]
    for i in range(len(Text)):
        holiday.append(pd.to_datetime(Text[i].text))  
    def holiday1(prev):
        if prev['timestamp'] in holiday:
            return 1
        else:
            return 0
    def before_holiday(prev):
        if prev['holiday']==1:
            return 0
        elif prev['timestamp']+timedelta(days=1) in holiday:
            return 1
        else:
            return 0    

    def after_holiday(prev):
        if prev['holiday']==1:
            return 0
        elif prev['timestamp']+timedelta(days=-1) in holiday:
            return 1
        else:
            return 0 
    def abnormal(prev):
        try: 
            if (prev['timestamp'].month in [6,7,8]) & (int(prev['temperature'])>=28):
                return 1
            elif (prev['timestamp'].month in [9,10,11]) & (int(prev['temperature'])>=5):
                return 1
            elif (prev['timestamp'].month in [12,1,2]) & (int(prev['temperature'])<=-5):
                return 1
            elif (prev['timestamp'].month in [3,4,5]) & (int(prev['temperature'])<=5):
                return 1
            else:
                return 0
        except:
            prev['temperature']==np.nan
            
    prev['abnormal']=prev.apply(abnormal,axis=1)         
    prev['holiday']=prev.apply(holiday1,axis=1)
    prev['before_holiday']=prev.apply(before_holiday,axis=1)
    prev['after_holiday']=prev.apply(after_holiday,axis=1)   
    
    bam=requests.get('https://www.weather.go.kr/weather/climate/past_table.jsp?stn=108&yy=2020&obs=21&x=18&y=7')  # 과거 rainfall rain_prob
    soup=BeautifulSoup(bam.text,'html.parser')
    content=soup.find_all('td')
    month=[i.month for i in prev[prev['rainfall'].isnull()]['timestamp']]
    day=[i.day for i in prev[prev['rainfall'].isnull()]['timestamp']]
    prev.loc[prev['rainfall'].isnull(),'rainfall']=[i.replace("\xa0", "0") if i =='\xa0' else i for i in [content[3+(day[u]-1)*13+month[u]].text for u in range(len(month))]]   
    prev.loc[(prev['rain_prob'].isnull())&(prev['rainfall']=='0'),'rain_prob']=0
    
    bam=requests.get('https://www.weather.go.kr/weather/climate/past_table.jsp?stn=108&yy=2020&obs=08&x=25&y=4')  # 과거 temperature
    soup=BeautifulSoup(bam.text,'html.parser')
    content=soup.find_all('td')
    month=[i.month for i in prev[prev['temperature'].isnull()]['timestamp']]
    day=[i.day for i in prev[prev['temperature'].isnull()]['timestamp']]
    prev.loc[prev['temperature'].isnull(),'temperature']=[i for i in [content[3+(day[u]-1)*13+month[u]].text for u in range(len(month))]]
   
    if float(prev.loc[prev['abnormal'].isnull(),'temperature'])>=28:
        prev.loc[prev['abnormal'].isnull(),'abnormal']=1
    else:
        prev.loc[prev['abnormal'].isnull(),'abnormal']=0
    
    return prev    


# In[622]:


make_df(prev,10,target).tail(20)

