import tweepy

# 填写您的 Twitter API 认证信息
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

# 认证
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# 创建 API 对象
api = tweepy.API(auth)

# 定义关键词
keyword = '大模型'

# 搜索推文
tweets = tweepy.Cursor(api.search_tweets, q=keyword).items(100)  # 采集 100 条推文

# 打印推文内容
for tweet in tweets:
    print(tweet.text)