from urllib.request import urlopen
import ssl
from urllib.request import urlretrieve
import urllib
import os
import shutil

while True:
    url='https://passport.99fund.com/cif/login/loginVerifyCode.htm?time=1528702116568'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    data = urllib.request.urlopen(req).read()

    with open('./temp.jpg','wb') as f:
        f.write(data)

    str=""
    while len(str)!=4:
        str = input("code=")
        if str=='':
            break

    if len(str)==4:
        shutil.move('./temp.jpg','./code/{}.jpg'.format(str))