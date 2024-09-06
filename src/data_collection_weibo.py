import requests
import json

# 填写您的 APP_KEY 和 APP_SECRET
APP_KEY = 'your_app_key'
APP_SECRET = 'your_app_secret'

# 获取 access_token
def get_access_token():
    url = f'https://api.weibo.com/oauth2/access_token?client_id={APP_KEY}&client_secret={APP_SECRET}&grant_type=client_credentials'
    response = requests.post(url)
    data = json.loads(response.text)
    return data['access_token']

# 根据关键词搜索微博
def search_weibo(keyword, access_token, page=1):
    url = f'https://api.weibo.com/2/search/topics.json?access_token={access_token}&q={keyword}&page={page}'
    response = requests.get(url)
    data = json.loads(response.text)
    return data

access_token = get_access_token()
keyword = '大模型'  # 您要搜索的关键词
data = search_weibo(keyword, access_token)
print(data)