# -*- coding: utf-8 -*-
"""
Refactored main script (抓股票資料 / 技術指標 / Firestore 寫入 已移除)
假設以下工作已在外部完成並提供 df：
 - 抓取 2301.TW 歷史資料
 - 計算所有技術指標
 - 將歷史資料寫回 Firestore

本檔案只負責：
 - 建立 LSTM 訓練資料
 - 訓練 / 預測 multi-step Close
 - 計算 Pred MA5 / MA10
 - Baseline 評估
 - 繪圖 +（可選）上傳 Storage
 - 將預測結果寫回 Firestore
"""

import os
import math
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Firebase（僅用於預測寫回與圖片上傳；不再負責歷史資料）
import json
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import storage

# =============================================================
# Firebase init（保留，用於 preds / image）
# =============================================================
key_dict = json.loads(os.environ.get("FIREBASE", "{}"))
db = None
bucket = None

if key_dict:
    cred = credentials.Certificate(key_dict)
    try:
        firebase_admin.get_app()
    except Exception:
        firebase_admin.initialize_app(
            cred, {"storageBucket": f"{key_dict.get('project_id')}.appspot.com"}
        )
    db = firestore.client()
    try:
        storage_client = storage.Client.from_service_account_info(key_dict)
        bucket = storage_client.bucket(f"{key_dict.get('project_id')}.appspot.com")
    except Exception:
        bucket = None

# =============================================================
# Dataset helpers（保留）
# =============================================================

def create_sequences(df, features, target_steps=10, window=60):
    X, y = [], []
    closes = df['Close'].values
    data = df[features].values
    for i in range(window, len(df) - target_steps + 1):
        X.append(data[i-window:i])
        y.append(closes[i:i+target_steps])
    return np.array(X), np.array(y)


def time_series_split(X, y, test_ratio=0.15):
    n = len(X)
    test_n = int(n * test_ratio)
    split_idx = n - test_n
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

# =============================================================
# Model
# =============================================================

def build_lstm_multi_step(input_shape, output_steps=10):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(output_steps))
    model.compile(optimizer='adam', loss='mae')
    return model

# =============================================================
# MA / metrics helpers
# =============================================================

def compute_pred_ma_from_pred_closes(last_known_closes, pred_closes):
    closes_seq = list(last_known_closes)
    results = []
    for pc in pred_closes:
        closes_seq.append(pc)
        ma5 = np.mean(closes_seq[-5:]) if len(closes_seq) >= 5 else np.mean(closes_seq)
        ma10 = np.mean(closes_seq[-10:]) if len(closes_seq) >= 10 else np.mean(closes_seq)
        results.append((pc, ma5, ma10))
    return results


def compute_metrics(y_true, y_pred):
    maes, rmses = [], []
    for step in range(y_true.shape[1]):
        maes.append(mean_absolute_error(y_true[:, step], y_pred[:, step]))
        rmses.append(math.sqrt(mean_squared_error(y_true[:, step], y_pred[:, step])))
    return np.array(maes), np.array(rmses)


def compute_ma_from_predictions(last_known_window_closes, y_pred_matrix, ma_period=5):
    n_samples, _ = last_known_window_closes.shape
    steps = y_pred_matrix.shape[1]
    preds_ma = np.zeros((n_samples, steps))
    for i in range(n_samples):
        seq = list(last_known_window_closes[i])
        for t in range(steps):
            seq.append(y_pred_matrix[i, t])
            look = seq[-ma_period:] if len(seq) >= ma_period else seq
            preds_ma[i, t] = np.mean(look)
    return preds_ma


def compute_true_ma(last_window, y_true, ma_period=5):
    n_samples, _ = last_window.shape
    steps = y_true.shape[1]
    true_ma = np.zeros((n_samples, steps))
    for i in range(n_samples):
        seq = list(last_window[i])
        for t in range(steps):
            seq.append(y_true[i, t])
            look = seq[-ma_period:] if len(seq) >= ma_period else seq
            true_ma[i, t] = np.mean(look)
    return true_ma

# =============================================================
# Plot + upload（保留）
# =============================================================

def plot_and_upload_to_storage(df_real, df_future, bucket_obj=None):
    df_real_plot = df_real.tail(10)
    if df_real_plot.empty:
        return None

    last_hist_date = df_real_plot.index[-1]
    start_row = {
        'date': last_hist_date,
        'Pred_Close': df_real_plot['Close'].iloc[-1],
        'Pred_MA5': df_real_plot['SMA_5'].iloc[-1],
        'Pred_MA10': df_real_plot['SMA_10'].iloc[-1],
    }

    df_future_plot = pd.concat([pd.DataFrame([start_row]), df_future], ignore_index=True)

    plt.figure(figsize=(16, 8))

    x_real = range(len(df_real_plot))
    plt.plot(x_real, df_real_plot['Close'], label='Close')
    plt.plot(x_real, df_real_plot['SMA_5'], label='SMA5')
    plt.plot(x_real, df_real_plot['SMA_10'], label='SMA10')

    offset = len(df_real_plot) - 1
    x_future = [offset + i for i in range(len(df_future_plot))]
    plt.plot(x_future, df_future_plot['Pred_Close'], 'r:o', label='Pred Close')

    for xf, val in zip(x_future, df_future_plot['Pred_Close']):
        plt.annotate(f"{val:.2f}", (xf, val), xytext=(6, 6), textcoords='offset points',
                     fontsize=8, bbox=dict(fc='white', alpha=0.7))

    plt.plot(x_future, df_future_plot['Pred_MA5'], '--', label='Pred MA5')
    plt.plot(x_future, df_future_plot['Pred_MA10'], '--', label='Pred MA10')

    labels = [d.strftime('%m-%d') for d in df_real_plot.index[:-1]] + \
             [d.strftime('%m-%d') for d in df_future_plot['date']]
    plt.xticks(range(len(labels)), labels, rotation=45)

    plt.legend()
    plt.title('2301.TW 預測')

    os.makedirs('results', exist_ok=True)
    fname = f"{datetime.now().strftime('%Y-%m-%d')}_pred.png"
    fpath = os.path.join('results', fname)
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close()

    if bucket_obj is not None:
        blob = bucket_obj.blob(f"LSTM_Pred_Images/{fname}")
        blob.upload_from_filename(fpath)
        try:
            blob.make_public()
            return blob.public_url
        except Exception:
            return blob.public_url

    return None

# =============================================================
# Main（df 由外部提供）
# =============================================================
if __name__ == '__main__':
    TICKER = '2301.TW'
    LOOKBACK = 60
    PRED_STEPS = 10
    TEST_RATIO = 0.15

    # === df 必須已包含所有特徵與技術指標 ===
    # 例如：from data_prepare import load_prepared_df
    # df = load_prepared_df()
    raise RuntimeError('請在此匯入你已完成「抓資料 + 指標 + Firestore」的 df')
