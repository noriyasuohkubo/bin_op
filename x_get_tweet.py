import tweepy
import time

#コンシューマーキー
#vCuYAICcquo09ywAnfmVlLz6m

#Secret Key
#6zmE2KC8rcEL6spdOSrRcRal1O5Co6EPDHEN87ySZK1FcZCKL9

#ベアラートークン
#AAAAAAAAAAAAAAAAAAAAAI6B9QEAAAAACFGyREW%2F48Vh0UCtoBhI9aKWwLw%3DrKMbzVzEquhYMkqTW936qkx5Hs1PLtUtyCRyQIQYphtMW6zcr3
#
#
# --- 設定情報 (取得したキーを入力) ---
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAI6B9QEAAAAACFGyREW%2F48Vh0UCtoBhI9aKWwLw%3DrKMbzVzEquhYMkqTW936qkx5Hs1PLtUtyCRyQIQYphtMW6zcr3'
API_KEY = 'YOUR_API_KEY'
API_SECRET = '6zmE2KC8rcEL6spdOSrRcRal1O5Co6EPDHEN87ySZK1FcZCKL9'
ACCESS_TOKEN = 'YOUR_ACCESS_TOKEN'
ACCESS_TOKEN_SECRET = 'YOUR_ACCESS_TOKEN_SECRET'

# --- 認証 ---
client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    #consumer_key=API_KEY,
    #consumer_secret=API_SECRET,
    #access_token=ACCESS_TOKEN,
    #access_token_secret=ACCESS_TOKEN_SECRET
)

# --- 通知対象のユーザーID ---
TARGET_USERNAME = 'realDonaldTrump'  # @を抜いたユーザー名
check_name = 'Donald J. Trump'
user_id = 25073877

def check_new_posts():
    # ユーザーIDを取得
    #user = client.get_user(username=TARGET_USERNAME)
    #print(user.data.name,user.data.id)
    #user_id = user.data.id

    #if user.data.name != check_name:
    #    raise Exception("check_name wrond!!!")
    #user_id = user.data.id

    # 最新の投稿を取得
    # see:https://docs.tweepy.org/en/stable/client.html#tweepy.Client.get_users_tweets
    tweets = client.get_users_tweets(id=user_id, max_results=5, start_time='2026-03-01T00:00:00Z')

    if tweets.data != None:
        print(len(tweets.data), tweets.data, )

        for d in tweets.data:
            print(d.id, d.text, d)
            #print(f"最新の投稿: {latest_tweet.text}")
            # ここでメール送信やSlack通知などの処理を入れる


time.perf_counter()

# --- 定期実行 ---
if __name__ == "__main__":
    start = time.perf_counter()
    check_new_posts()
    print("process time:", time.perf_counter() - start )
    """
    while True:
        try:
            check_new_posts()
            # 60秒待機（API制限に注意）
            time.sleep(60)
        except Exception as e:
            print(f"エラー: {e}")
            time.sleep(300)  #
    """