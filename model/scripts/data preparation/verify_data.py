import pandas as pd

df = pd.read_csv("D:\Python\Credit_UnderWriting_Model\data\processed\district_features.csv")
print(df.shape)
print(df.columns)
print(df.isnull().sum())
df.head()
