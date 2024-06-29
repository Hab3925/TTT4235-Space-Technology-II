import pandas as pd

df = pd.read_csv("Ballon_NTNU.csv",skiprows=0,header=0, sep=";")
df = pd.DataFrame({'time': df['t'], 'pressure': df['pressure'], 'height': df['height'], 'temperature': df['temp']})
df = df.replace(',', '.', regex=True)
df = df.astype(float)

df['height'] = df['height'] *1000
df['pressure'] = df['pressure'] * 1000

df.dropna(inplace=True)
print(df.head())

df.to_csv("yep.csv", index=False)