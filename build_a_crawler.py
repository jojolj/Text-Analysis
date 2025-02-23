# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 23:02:55 2023
WebCrawler

@author: justd
"""
import requests #用来爬取网页
from bs4 import BeautifulSoup #用来解析网页
seeds = ['https://www.sis.uta.fi/~tojape/', 'https://www.tuni.fi/en']

end_sum = 0

def do_save_action(text=None):
    pass
print("len(seeds):", len(seeds))
print("seeds[end_sum]:", seeds[end_sum])

print("seeds:",seeds)
#%%
while end_sum < 3:
    print("len(seeds):", len(seeds))
    print("seeds[end_sum]:", seeds[end_sum])
    if end_sum < len(seeds):
        r = requests.get(seeds[end_sum])
        print("end_sum:", end_sum)
        end_sum = end_sum + 1
        do_save_action(r.text)
        soup = BeautifulSoup(r.content)
        urls=soup.find_all('a')
        for url in urls:
            seeds.append(url)
            print("seeds:",seeds)
                        
    else:
        break
       
#%%以上爬虫缺点
1. 网页一个个爬速度慢，需开启多个线程，或用分布式架构去并发地爬取网页？？？
2. seed url和解析到的url都存放在一个列表里， 
    应该设计一个更合理的数据结构来存放这些待爬取的url,如队列或优先队列
3. 为避免每次对同一个网站（含多个网页的网站其ip地址都一样）都发起DNS请求费时， 可以考虑将ip地址进行缓存。
4. 对解析到的url进行去重 ？？？

##如何并行爬取
分布式爬虫架构，使用一致性Hash算法








