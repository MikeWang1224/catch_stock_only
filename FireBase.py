import firebase_admin
from firebase_admin import credentials, firestore
import yfinance as yf
import pandas as pd
from datetime import datetime
import json
import os
import re
import time
import requests
from datetime import datetime
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
# 讀取 Firebase 服務帳戶金鑰
#cred = credentials.Certificate(
    #"stockgpt-150d0-firebase-adminsdk-fbsvc-874413114f.json")
key_dict=json.loads(os.environ["FIREBASE"])
cred = credentials.Certificate(key_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()
# 初始化 Firebase
#firebase_admin.initialize_app(cred)

# 取得 Firestore 資料庫
#db = firestore.client()

# 取得台積電 (TSMC) 股價
tsmc = yf.Ticker("2330.TW")
df_tsmc = tsmc.history(period="6mo")

# 取得聯電 (UMC) 股價
umc = yf.Ticker("2303.TW")
df_umc = umc.history(period="6mo")

# 取得鴻海股價
foxconn = yf.Ticker("2317.TW")
df_foxconn = foxconn.history(period="6mo")

# 取得美股台積電 ADR (TSM) 股價
tsm_adr = yf.Ticker("TSM")
df_tsm_adr = tsm_adr.history(period="6mo")

# 取得美股聯電 ADR (UMC) 股價
umc_adr = yf.Ticker("UMC")
df_umc_adr = umc_adr.history(period="6mo")

# 取得美股鴻海 ADR (HNHPF) 股價
foxconn_adr = yf.Ticker("HNHPF")
df_foxconn_adr = foxconn_adr.history(period="6mo")

# 計算技術指標函數
def calculate_indicators(df):
    # SMA 指標
    df['SMA_5'] = df['Close'].rolling(window=5).mean().round(5)
    df['SMA_10'] = df['Close'].rolling(window=10).mean().round(5)
    df['SMA_50'] = df['Close'].rolling(window=50).mean().round(5)

    # RSI 指標
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=20).mean()
    avg_loss = loss.rolling(window=20).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = (100 - (100 / (1 + rs))).round(5)

    # KD 指標
    df['Lowest_14'] = df['Low'].rolling(window=14).min()
    df['Highest_14'] = df['High'].rolling(window=14).max()
    df['K'] = (100 * (df['Close'] - df['Lowest_14']) / (df['Highest_14'] - df['Lowest_14'])).round(5)
    df['D'] = df['K'].rolling(window=3).mean().round(5)

    # MACD 指標
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = (df['EMA_12'] - df['EMA_26']).round(5)
    df['SignalLine'] = df['MACD'].ewm(span=9, adjust=False).mean().round(5)

    return df

# 為所有股票計算指標
df_tsmc = calculate_indicators(df_tsmc)
df_umc = calculate_indicators(df_umc)
df_foxconn = calculate_indicators(df_foxconn)

# 選擇要顯示的欄位
selected_columns = ['Close', 'MACD', 'RSI', 'K', 'D', 'Volume']

# Firebase Collection 名稱
collection_name = "NEW_stock_data"

# 將所有股票的數據整理成符合預期結構的格式
def save_data_by_date():
    stock_data = {}

    # 處理台股數據
    for df, stock in [(df_tsmc, "2330.TW"), (df_umc, "2303.TW"), (df_foxconn, "2317.TW")]:
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d")
            if date_str not in stock_data:
                stock_data[date_str] = {}
            stock_data[date_str][stock] = {col: round(float(row[col]), 5) for col in selected_columns if not pd.isna(row[col])}

    # 處理 ADR 數據
    for df, stock in [(df_tsm_adr, "TSM"), (df_umc_adr, "UMC"), (df_foxconn_adr, "HNHPF")]:
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d")
            if date_str not in stock_data:
                stock_data[date_str] = {}
            if stock not in stock_data[date_str]:
                stock_data[date_str][stock] = {}
            stock_data[date_str][stock]["Close"] = round(float(row["Close"]), 5)
            stock_data[date_str][stock]["Volume"] = round(float(row["Volume"]), 5)

    # 將數據寫入 Firestore
    batch = db.batch()
    count = 0
    for date, data in stock_data.items():
        doc_ref = db.collection(collection_name).document(date)
        batch.set(doc_ref, data)
        count += 1
        if count >= 450:
            batch.commit()
            print(f"批次寫入了 {count} 筆資料")
            batch = db.batch()
            count = 0

    if count > 0:
        batch.commit()
        print(f"批次寫入了剩餘的 {count} 筆資料")

    print("所有股票數據已成功寫入 Firestore！")

# 執行保存
save_data_by_date()
