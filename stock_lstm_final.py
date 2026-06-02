import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ─────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────
TICKER     = "AAPL"
START_DATE = "2018-01-01"
END_DATE   = "2024-01-01"
WINDOW     = 60
EPOCHS     = 50
BATCH_SIZE = 32
TEST_SPLIT = 0.15
VAL_SPLIT  = 0.1

# ─────────────────────────────────────────
# 2. DOWNLOAD DATA
# ─────────────────────────────────────────
print(f"Downloading {TICKER} data ...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)
df.dropna(inplace=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────
df["SMA_20"] = df["Close"].rolling(20).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()
df["EMA_12"] = df["Close"].ewm(span=12).mean()

delta = df["Close"].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
df["RSI"] = 100 - (100 / (1 + gain / loss))

df["MACD"]        = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

sma20          = df["Close"].rolling(20).mean()
std20          = df["Close"].rolling(20).std()
df["BB_upper"] = sma20 + 2 * std20
df["BB_lower"] = sma20 - 2 * std20

df["Return_1d"]  = df["Close"].pct_change(1)
df["Return_5d"]  = df["Close"].pct_change(5)
df["Volatility"] = df["Return_1d"].rolling(20).std()
df.dropna(inplace=True)

FEATURES = [
    "Close", "Volume",
    "SMA_20", "SMA_50", "EMA_12",
    "RSI", "MACD", "MACD_signal",
    "BB_upper", "BB_lower",
    "Return_1d", "Return_5d", "Volatility",
]

# ─────────────────────────────────────────
# 4. SCALE & SPLIT
# ─────────────────────────────────────────
data      = df[FEATURES].values
split_idx = int(len(data) * (1 - TEST_SPLIT))

scaler       = MinMaxScaler()
train_scaled = scaler.fit_transform(data[:split_idx])
test_scaled  = scaler.transform(data[split_idx:])

def make_sequences(arr, window):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i - window:i])
        y.append(arr[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = make_sequences(train_scaled, WINDOW)
X_test,  y_test  = make_sequences(
    np.vstack([train_scaled[-WINDOW:], test_scaled]), WINDOW
)
print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")

# ─────────────────────────────────────────
# 5. MODEL
# ─────────────────────────────────────────
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(WINDOW, len(FEATURES))),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# ─────────────────────────────────────────
# 6. TRAIN
# ─────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
]
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    callbacks=callbacks,
    verbose=1,
)

# ─────────────────────────────────────────
# 7. PREDICT & INVERSE-SCALE
# ─────────────────────────────────────────
def inverse_close(vals, scaler, n):
    dummy = np.zeros((len(vals), n))
    dummy[:, 0] = vals.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

y_pred = inverse_close(model.predict(X_test),      scaler, len(FEATURES))
y_true = inverse_close(y_test.reshape(-1,1),        scaler, len(FEATURES))

# ─────────────────────────────────────────
# 8. METRICS
# ─────────────────────────────────────────
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
r2   = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

print(f"\n{'='*45}")
print(f"  TICKER : {TICKER}  |  Test period: last {TEST_SPLIT*100:.0f}%")
print(f"{'='*45}")
print(f"  RMSE     : ${rmse:.2f}")
print(f"  MAE      : ${mae:.2f}")
print(f"  MAPE     : {mape:.2f}%")
print(f"  R²       : {r2:.4f}")
print(f"{'='*45}")

# ─────────────────────────────────────────
# 9. PLOTS  (3-panel, clean)
# ─────────────────────────────────────────
fig = plt.figure(figsize=(14, 11))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

# Panel A — Actual vs Predicted
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(y_true, label="Actual Price",    color="#185FA5", linewidth=1.8)
ax1.plot(y_pred, label="Predicted Price", color="#D85A30", linewidth=1.8, linestyle="--")
ax1.fill_between(range(len(y_true)), y_true, y_pred, alpha=0.08, color="#D85A30")
ax1.set_title(f"{TICKER} — Actual vs Predicted Close Price (Test Set)", fontsize=13, fontweight="bold")
ax1.set_ylabel("Price (USD)")
ax1.set_xlabel("Trading Days")
# Metrics annotation box
textstr = f"RMSE: ${rmse:.2f}  |  MAE: ${mae:.2f}  |  MAPE: {mape:.2f}%  |  R²: {r2:.4f}"
ax1.text(0.01, 0.97, textstr, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax1.legend(loc="lower right")
ax1.grid(alpha=0.25)

# Panel B — Training loss
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(history.history["loss"],     label="Train loss", color="#185FA5", linewidth=1.5)
ax2.plot(history.history["val_loss"], label="Val loss",   color="#D85A30", linewidth=1.5)
ax2.set_title("Training & Validation Loss (MSE)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Loss"); ax2.set_xlabel("Epoch")
ax2.legend(); ax2.grid(alpha=0.25)

# Panel C — Residuals
residuals = y_true - y_pred
ax3 = fig.add_subplot(gs[1, 1])
ax3.bar(range(len(residuals)), residuals, color=["#185FA5" if r >= 0 else "#D85A30" for r in residuals], alpha=0.6, width=1.0)
ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_title("Prediction Residuals (Actual − Predicted)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Residual (USD)"); ax3.set_xlabel("Trading Days")
ax3.grid(alpha=0.25)

plt.suptitle(f"{TICKER} LSTM Stock Price Prediction", fontsize=15, fontweight="bold", y=1.01)
plt.savefig("stock_lstm_final_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved → stock_lstm_final_results.png")

model.save("stock_lstm_final.keras")
print("Model saved → stock_lstm_final.keras")
