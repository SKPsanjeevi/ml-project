import pandas as pd
import quandl

quandl.ApiConfig.api_key =  "-mEZpeg3QHmxA4fn8pKY"
df = quandl.get("WIKI/GOOGL")

print(df.head())
