# -*- coding: utf-8 -*-
"""
FireBase_LSTM_v2.py
"""

import os, json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import firebase_admin
from firebase_admin import credentials, firestore

# ================= Firebase 初始化 =================
key_dict = json.loads(os.environ.get("FIREBASE", "{}"))
db = None

if key_dict:
    cred = credentials.Certificate(key_dict)
    try:
        firebase_admin.get_app()
    except Exception:
        firebase_admin.initialize_app(cred)
    db = firestore.client()

# ================= Firestore 讀取 =================
def load_df_from_firestore(ticker, collection="NEW_stock_data_liteon", days=400):
    rows = []
    if db:
        for doc in db.collection(collection).stream():
            p = doc.to_dict().get(ticker)
            if p:
                rows.append({"date": doc.id, **p})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").tail(days).set_index("date")

# ================= 假日補今天（只給預測用） =================
def ensure_today_row(df):
    today = pd.Timestamp(datetime.now().date())
    last_trade = df.index.max()
    if last_trade < today:
        df.loc[today] = df.loc[last_trade]
        print(f"⚠️ 今日無資料，使用 {last_trade.date()} 補今日")
    return df.sort_index()

# ================= Sequence =================
def create_sequences(df, features, steps=10, window=60):
    X, y = [], []
    data = df[features].values
    logret = np.log(df["Close"] / df["Close"].shift())

    for i in range(window, len(df) - steps):
        X.append(data[i-window:i])
        y.append(logret.iloc[i:i+steps].values)

    return np.array(X), np.array(y)

# ================= LSTM =================
def build_lstm(input_shape, steps):
    m = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.1),
        Dense(steps)
    ])
    m.compile(optimizer="adam", loss="huber")
    return m

# ================= 原預測圖（不動） =================
def plot_and_save(df_hist, future_df):
    hist = df_hist.tail(10)

    hist_dates = hist.index.strftime("%Y-%m-%d").tolist()
    future_dates = future_df["date"].dt.strftime("%Y-%m-%d").tolist()

    all_dates = hist_dates + future_dates
    x_hist = np.arange(len(hist_dates))
    x_future = np.arange(len(hist_dates), len(all_dates))

    plt.figure(figsize=(18,8))
    ax = plt.gca()

    ax.plot(x_hist, hist["Close"], label="Close")
    ax.plot(x_hist, hist["SMA5"], label="SMA5")
    ax.plot(x_hist, hist["SMA10"], label="SMA10")

    ax.plot(
        np.concatenate([[x_hist[-1]], x_future]),
        [hist["Close"].iloc[-1]] + future_df["Pred_Close"].tolist(),
        "r:o", label="Pred Close"
    )

    ax.set_xticks(np.arange(len(all_dates)))
    ax.set_xticklabels(all_dates, rotation=45, ha="right")

    ax.legend()
    ax.set_title("2301.TW LSTM 預測（Return-based 穩定版）")

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{datetime.now():%Y-%m-%d}_pred.png", dpi=300)
    plt.close()

# ================= 回測誤差圖（✔ 完全對齊交易日） =================
def plot_backtest_error(df_real, X_te_s, y_te, model, steps, split, lookback):
    anchor_idx = split + lookback - 1

    X_last = X_te_s[-1:]
    y_true = y_te[-1]
    pred_ret = model.predict(X_last, verbose=0)[0]

    # 🔥 使用「純交易日 df」
    dates = df_real.index[anchor_idx + 1 : anchor_idx + 1 + steps]
    start_price = df_real["Close"].iloc[anchor_idx]

    p_t, p_p = start_price, start_price
    true_prices, pred_prices = [], []

    for rt, rp in zip(y_true, pred_ret):
        p_t *= np.exp(rt)
        p_p *= np.exp(rp)
        true_prices.append(p_t)
        pred_prices.append(p_p)

    true_prices = np.array(true_prices)
    pred_prices = np.array(pred_prices)

    mae = np.mean(np.abs(true_prices - pred_prices))
    rmse = np.sqrt(np.mean((true_prices - pred_prices) ** 2))

    plt.figure(figsize=(12,6))
    plt.plot(dates, true_prices, label="Actual Close")
    plt.plot(dates, pred_prices, "--o", label="Pred Close")
    plt.xticks(dates, [d.strftime("%Y-%m-%d") for d in dates], rotation=45)
    plt.title(f"Backtest Prediction | MAE={mae:.2f}, RMSE={rmse:.2f}")
    plt.legend()
    plt.grid(True)

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{datetime.now():%Y-%m-%d}_backtest.png", dpi=300)
    plt.close()

# ================= Main =================
if __name__ == "__main__":
    TICKER = "2301.TW"
    LOOKBACK = 60
    STEPS = 10

    # 🔥 關鍵：分兩份 df
    df_real = load_df_from_firestore(TICKER)
    df_pred = ensure_today_row(df_real.copy())

    FEATURES = ["Close","Volume","RSI","MACD","K","D","ATR_14"]

    for df in (df_real, df_pred):
        df["SMA5"] = df["Close"].rolling(5).mean()
        df["SMA10"] = df["Close"].rolling(10).mean()

    df_real = df_real.dropna()
    df_pred = df_pred.dropna()

    X, y = create_sequences(df_real, FEATURES, STEPS, LOOKBACK)
    split = int(len(X) * 0.85)

    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    sx = MinMaxScaler()
    sx.fit(df_real[FEATURES].iloc[:split + LOOKBACK])

    def scale_X(X):
        n,t,f = X.shape
        return sx.transform(X.reshape(-1,f)).reshape(n,t,f)

    X_tr_s = scale_X(X_tr)
    X_te_s = scale_X(X_te)

    model = build_lstm((LOOKBACK,len(FEATURES)), STEPS)
    model.fit(
        X_tr_s, y_tr,
        epochs=50,
        batch_size=32,
        verbose=2,
        callbacks=[EarlyStopping(patience=6, restore_best_weights=True)]
    )

    # ===== 未來預測（用 df_pred）=====
    raw_returns = model.predict(X_te_s)[-1]

    today = pd.Timestamp(datetime.now().date())
    last_trade_date = df_real.index[df_real.index < today][-1]
    last_close = df_real.loc[last_trade_date, "Close"]

    prices, price = [], last_close
    for r in raw_returns:
        price *= np.exp(r)
        prices.append(price)

    future_df = pd.DataFrame({"Pred_Close": prices})
    future_df["date"] = pd.bdate_range(last_trade_date + BDay(1), periods=STEPS)

    plot_and_save(df_pred, future_df)
    plot_backtest_error(df_real, X_te_s, y_te, model, STEPS, split, LOOKBACK)
