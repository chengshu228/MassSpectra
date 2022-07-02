import pandas as pd
import numpy as np
import os

location = os.path.dirname(os.path.realpath(__file__))

hmdb = pd.read_csv(location+r'\HMDB_data.csv', delimiter=",", usecols=[0,1])

# print(hmdb.head(5))

import os
import numpy as np
import re
import requests
import urllib
from urllib import parse,error
import chardet
from selenium import webdriver
 
location = os.path.dirname(os.path.realpath(__file__))
mass_spectra_location = os.path.join(location, "mass_spectra")
if not os.path.exists(mass_spectra_location):
    os.mkdir(mass_spectra_location)
smiles_intensity_location = os.path.join(location, "smiles_intensity")
if not os.path.exists(smiles_intensity_location):
    os.mkdir(smiles_intensity_location)
html_intensity_location = os.path.join(location, "html_intensity")
if not os.path.exists(html_intensity_location):
    os.mkdir(html_intensity_location)
for _,_,filename in os.walk(mass_spectra_location): 
    # 路径名, 当前路径下的文件夹名字, 当前路径下的文件名字
    continue

# #打开谷歌浏览器
# driver = webdriver.Edge()
# #打开百度搜索主页
# driver.get('https://hmdb.ca/metabolites')
# '''
# 调用selenium库中的find_element_by_xpath()方法定位搜索框，
# 同时使用send_keys()方法在其中输入信息
# '''
# driver.find_element_by_xpath('//*[@id="kw"]').send_keys('HMDB0030820')
# '''
# 调用selenium库中的find_element_by_xpath()方法定位搜索按钮，
# 同时使用click()方法对按钮进行点击
# '''
# driver.find_element_by_xpath('//*[@id="su"]').click()



def get_redirect_url(base_url, query, filename):
    # key=query
    # key={"wd":key}

    # data=urllib.parse.urlencode(key)
    # url = base_url + data


    # # 请求头，这里我设置了浏览器代理
    # param = urllib.parse.urlencode(param)
    # print(param)

    url = base_url + query
    print(url)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36 Edg/100.0.1185.29'
    }
    req=urllib.request.Request(url,headers=headers) #创建请求对象
    res=urllib.request.urlopen(req) #对网页发起请求并获取响应
    html=res.read().decode("utf-8")
    with open("/html_intensity/{}.html".format(filename),"w",encoding="utf-8") as f:
        f.write(html)
        f.close()

    return None
 
 
# if __name__ == '__main__':
#     redirect_url = get_redirect_url(base_url, query, file_name)
 
base_url = "https://hmdb.ca/metabolites/"

# proxy = {"http": "47.101.44.122"}
# proxy_handler = urllib.request.ProxyHandler(proxy)
# opener = urllib.request.build_opener(proxy_handler)
# urllib.request.install_opener(opener)



try :
    with open("get_name.txt", "w", encoding="utf-8") as f_out:

        for i in np.arange(len(filename)):
            file_no = int(filename[i][2:-4])
            # print("file_no: ", file_no)
            # file_no = int(filename[i][2:-4])
            with open(mass_spectra_location+r"/{}".format(filename[i]), "r", encoding="utf-8") as f_in:
                f_context = f_in.read()
                # result = re.findall(r"CAS (.+?)\n", f_context, re.S)
                result = re.findall(r"RECORD_TITLE: (.+?); ", f_context, re.S)
                print(filename[i], result)
                for item in result:
                    f_out.write(item+'\n')
                f_in.close()
        f_out.close()
    

except error.HTTPError as e:
    print("HTTPError:{}".format(e.reason))
    print("HTTPError:{}".format(e))    
except error.URLError as e:
    print("URLError:{}".format(e.reason))
    print("URLError:{}".format(e))
except Exception as e:
    print(e)