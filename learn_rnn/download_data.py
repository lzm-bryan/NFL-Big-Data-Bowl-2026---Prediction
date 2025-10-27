import requests

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv"
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
r = requests.get(url, timeout=30)
r.raise_for_status()  # 如果失败会抛异常
with open("shampoo.csv", "wb") as f:
    f.write(r.content)

print("下载完成 -> shampoo.csv")
