import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("temperature.csv")   # 换成你的文件名或路径

# 展示前 5 行（默认）
print(df.head())


# 想看后几行用：
print(df.tail(5))
