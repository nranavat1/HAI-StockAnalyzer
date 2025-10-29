import os, glob, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#  Settings 
MAX_TICKERS   = 1000   # how many tickers to read at most
TEST_TRIALS   = 200    # exactly 200 test trials
WIN           = 10     # days of history (window length)
EPOCHS        = 15
BATCH         = 64
LR            = 1e-3
SEED          = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

#  Model (TCN) 
# Make sure TCN.py is in the same folder with class TCN(in_ch, widths=(...), k=..., drop=...)
from TCN import TCN  # your fixed TCN implementation

#  Load Kaggle data 
import kagglehub
root = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")
stocks_dir = os.path.join(root, "stocks")
csvs = sorted(glob.glob(os.path.join(stocks_dir, "*.csv")))[:MAX_TICKERS]
print(f"Reading {len(csvs)} CSV files")

#  Read the data, build the training and testing data
# normalize the data so we don't have large values during training
X_list, y_list, tickers_list = [], [], []
open_mu_list, open_sigma_list = [], []

for fp in csvs:
    try:
        df = pd.read_csv(fp)
    except Exception:
        continue

    required = {"Date","Open","High","Low","Close","Volume"}
    if not required.issubset(df.columns):
        continue

    df = df[["Open","High","Low","Close","Volume"]].dropna().reset_index(drop=True)
    if len(df) <= WIN:
        continue

    # Per-ticker normalization and remember stats (for OPEN only)
    mu = {}; sigma = {}
    for col in ["Open","High","Low","Close","Volume"]:
        col_vals = pd.to_numeric(df[col], errors="coerce")
        m = float(col_vals.mean())
        s = float(col_vals.std(ddof=0))  # population-style std for stability
        mu[col], sigma[col] = m, s
        if s <= 1e-8 or np.isnan(s):
            df[col] = 0.0
        else:
            df[col] = (col_vals - m) / s

    ticker_symbol = os.path.splitext(os.path.basename(fp))[0].upper()

    # Past WIN days (normalized) -> predict next day's OPEN (normalized)
    X = df.head(WIN).values.T                   # (5, WIN)
    y = float(df.iloc[WIN]["Open"]) if len(df) > WIN else None
    if y is None or np.isnan(y):
        continue

    X_list.append(X)
    y_list.append(y)
    tickers_list.append(ticker_symbol)
    open_mu_list.append(mu["Open"])
    open_sigma_list.append(sigma["Open"])

    if len(X_list) >= 1200:   # safety cap so it runs fast
        break

total = len(X_list)
print("Total usable samples:", total)

#  Stack and split (random 200 for test) 
X = np.stack(X_list).astype(np.float32)            # (N, 5, WIN)
y = np.array(y_list, dtype=np.float32)             # (N,)
idx = np.arange(len(X))
np.random.shuffle(idx)

test_idx  = idx[:TEST_TRIALS]
train_idx = idx[TEST_TRIALS:]

X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

tickers_test     = [tickers_list[i]    for i in test_idx]
open_mu_test     = np.array([open_mu_list[i]    for i in test_idx], dtype=np.float32)
open_sigma_test  = np.array([open_sigma_list[i] for i in test_idx], dtype=np.float32)

print("Training and testing data shape: ", X_train.shape, y_train.shape, "|", X_test.shape, y_test.shape)

#  Dataset / DataLoader 
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  
        self.y = torch.tensor(y, dtype=torch.float32) 
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=BATCH, shuffle=True)
test_loader  = DataLoader(StockDataset(X_test,  y_test ), batch_size=BATCH, shuffle=False)

#  Train 
model = TCN(in_ch=1, num_channels=32, k=3, num_layers=8, device = device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        preds = model(xb)                       # (B,)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch:02d}: Train MSE = {total_loss / len(train_loader.dataset):.6f}")

#  Evaluate 
model.eval()
preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device, non_blocking=True)
        preds.append(model(xb).cpu().numpy())
preds = np.concatenate(preds, axis=0).astype(np.float32) 

#  Denormalize per sample back to dollars 
# previous normalized price = last OPEN in window
prev_norm = X_test[:, 0, -1]           
pred_norm = preds                        
true_norm = y_test                       

safe_sigma = np.where(open_sigma_test <= 1e-8, 1.0, open_sigma_test)

prev_real = prev_norm * safe_sigma + open_mu_test
pred_real = pred_norm * safe_sigma + open_mu_test
true_real = true_norm * safe_sigma + open_mu_test

#  metrics in normalized space (comparable to training) 
mse_norm = float(np.mean((pred_norm - true_norm)**2))
print(f"Test MSE: {mse_norm:}")

#  Create AI advice for the human and save the data
ai_advice = np.where(pred_real > prev_real, "BUY", "HOLD")

study_df = pd.DataFrame({
    "Ticker": tickers_test,
    "Previous_Open": prev_real,
    "Predicted_Next_Open": pred_real,
    "True_Next_Open": true_real,
    "AI_Advice": ai_advice
})
study_df.to_csv("study_trials_with_ai.csv", index=False)
print(f" Saved study_trials_with_ai.csv ({len(study_df)} trials)")

PATH = "model_weights.pth"
model_state_dict = model.state_dict()
torch.save(model_state_dict, PATH)
