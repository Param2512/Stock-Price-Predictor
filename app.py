import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
import io

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    color: #ffffff;
    margin: 0;
    letter-spacing: -1px;
}
.hero p {
    color: #a0c4d8;
    font-size: 1rem;
    margin-top: 0.5rem;
}

.metric-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.metric-card .label {
    color: #7c8db5;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem;
    color: #00d4aa;
    font-weight: 700;
}

.stButton > button {
    background: linear-gradient(90deg, #00d4aa, #0099cc);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    width: 100%;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📈 Stock Price Predictor</h1>
    <p>LSTM Deep Learning Model — Technical Indicator Feature Engineering</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    ticker     = st.text_input("Ticker Symbol", value="AAPL").upper()
    start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
    end_date   = st.date_input("End Date",   value=pd.to_datetime("2024-01-01"))
    window     = st.slider("Lookback Window (days)", 30, 90, 60)
    epochs     = st.slider("Epochs", 10, 60, 50)
    run_btn    = st.button("🚀 Run Prediction")

    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown("- 3-layer LSTM\n- 13 Technical Features\n- EarlyStopping\n- MinMax Scaling")

# ── Main ──────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(f"Downloading {ticker} data..."):
        df = yf.download(ticker, start=str(start_date), end=str(end_date), auto_adjust=True)
        df.dropna(inplace=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    if df.empty:
        st.error("No data found. Check the ticker symbol.")
        st.stop()

    # Feature engineering
    with st.spinner("Engineering features..."):
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["EMA_12"] = df["Close"].ewm(span=12).mean()
        delta = df["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        df["RSI"]         = 100 - (100 / (1 + gain / loss))
        df["MACD"]        = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
        df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
        sma20 = df["Close"].rolling(20).mean()
        std20 = df["Close"].rolling(20).std()
        df["BB_upper"]   = sma20 + 2 * std20
        df["BB_lower"]   = sma20 - 2 * std20
        df["Return_1d"]  = df["Close"].pct_change(1)
        df["Return_5d"]  = df["Close"].pct_change(5)
        df["Volatility"] = df["Return_1d"].rolling(20).std()
        df.dropna(inplace=True)

    FEATURES = ["Close","Volume","SMA_20","SMA_50","EMA_12","RSI",
                "MACD","MACD_signal","BB_upper","BB_lower",
                "Return_1d","Return_5d","Volatility"]

    data      = df[FEATURES].values
    split_idx = int(len(data) * 0.85)
    scaler    = MinMaxScaler()
    train_sc  = scaler.fit_transform(data[:split_idx])
    test_sc   = scaler.transform(data[split_idx:])

    def make_seq(arr, w):
        X, y = [], []
        for i in range(w, len(arr)):
            X.append(arr[i-w:i]); y.append(arr[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = make_seq(train_sc, window)
    X_test,  y_test  = make_seq(np.vstack([train_sc[-window:], test_sc]), window)

    # Build & train
    with st.spinner("Training LSTM model... (this takes a few minutes)"):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(window, len(FEATURES))),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        ]
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                            validation_split=0.1, callbacks=callbacks, verbose=0)

    # Predict
    def inv(vals, sc, n):
        d = np.zeros((len(vals), n)); d[:, 0] = vals.flatten()
        return sc.inverse_transform(d)[:, 0]

    y_pred = inv(model.predict(X_test), scaler, len(FEATURES))
    y_true = inv(y_test.reshape(-1,1),  scaler, len(FEATURES))

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2   = 1 - np.sum((y_true-y_pred)**2) / np.sum((y_true-np.mean(y_true))**2)

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.markdown("### 📊 Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE",  f"${rmse:.2f}")
    c2.metric("MAE",   f"${mae:.2f}")
    c3.metric("MAPE",  f"{mape:.2f}%")
    c4.metric("R²",    f"{r2:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    st.markdown("### 📈 Results")
    fig = plt.figure(figsize=(14, 10), facecolor="#0e1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

    def dark_ax(ax):
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#aaaaaa")
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        ax.title.set_color("#ffffff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(y_true, label="Actual Price",    color="#4fc3f7", linewidth=1.8)
    ax1.plot(y_pred, label="Predicted Price", color="#ff7043", linewidth=1.8, linestyle="--")
    ax1.fill_between(range(len(y_true)), y_true, y_pred, alpha=0.07, color="#ff7043")
    ax1.set_title(f"{ticker} — Actual vs Predicted Close Price", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Price (USD)"); ax1.set_xlabel("Trading Days")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white")
    ax1.grid(alpha=0.15); dark_ax(ax1)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(history.history["loss"],     label="Train", color="#4fc3f7", linewidth=1.5)
    ax2.plot(history.history["val_loss"], label="Val",   color="#ff7043", linewidth=1.5)
    ax2.set_title("Training & Validation Loss", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Loss"); ax2.set_xlabel("Epoch")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white")
    ax2.grid(alpha=0.15); dark_ax(ax2)

    residuals = y_true - y_pred
    colors = ["#4fc3f7" if r >= 0 else "#ff7043" for r in residuals]
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(range(len(residuals)), residuals, color=colors, alpha=0.7, width=1.0)
    ax3.axhline(0, color="white", linewidth=0.8)
    ax3.set_title("Prediction Residuals", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Residual (USD)"); ax3.set_xlabel("Trading Days")
    ax3.grid(alpha=0.15); dark_ax(ax3)

    st.pyplot(fig)

    # Download button
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#0e1117")
    st.download_button("⬇️ Download Chart", buf.getvalue(), f"{ticker}_prediction.png", "image/png")

else:
    st.info("👈 Configure settings in the sidebar and click **Run Prediction** to start.")
    st.markdown("""
    #### How it works
    1. Enter any stock ticker (AAPL, TSLA, GOOGL, etc.)
    2. Set date range and model parameters
    3. Click Run — model trains live on fetched data
    4. View predictions, metrics, and download charts
    """)
