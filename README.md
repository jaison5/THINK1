# 論文一：Yahoo 匯率日線的前視洩漏，與無前視的量子序列模型基準

**暫定題目**：Look-Ahead Leakage and Simulator Artifacts in Quantum Time-Series Forecasting: An Audit and a Leak-Free Walk-Forward Benchmark

這個資料夾放論文一用到的所有程式、資料、結果和紀錄，由 `B:\qfin-main\sync_papers.py` 從原本的實驗資料夾複製過來。資料夾結構和原本相同（`qfin-main`、`A`、`B` 並排），程式不用改路徑就能直接執行。

## 研究目的

1. **稽核**：證明 Yahoo 匯率日線的「當日最高價、最低價」會洩漏隔天收盤的方向，讓已發表的量子序列模型（QLSTM、QFWP）看起來有約 70% 的方向準確率。
2. **修正**：建立無前視、walk-forward、多個匯率、有古典基準與統計檢定的評測流程，回答兩個問題：去掉前視後，QFWP 是否仍優於 QLSTM？量子模型是否優於同參數量的古典模型？
3. **系統面**：證明論文一（arXiv 2512.21820）「批次加速約 2 倍就到頂、反向傳播是瓶頸」是模擬器實作造成的；並指出在本 repo 的電路配置下，一半的量子參數不影響輸出，「參數量相同」的比較並不公平。

## 資料夾對照

| 位置 | 內容 | 對應論文章節 |
|---|---|---|
| `qfin-main/` | 原論文實驗一的重現：PennyLane 程式（`pl_models.py`、`pl_exp1.py`）、Mac（`results/`）與 Windows（`results_win/`）的結果、補充計時（`pl_exp1_timing.py`）、重現報告 `REPRODUCTION_WIN.md` | 重現原論文 |
| `A/probe_leakage.py`、`A/data/probe/`、`A/results/probe/` | 跨資產前視探針：22 個 Yahoo 匯率、6 檔股票／ETF（對照）、4 個加密貨幣 | 前視稽核 |
| `A/`（`wf.py`、`run_a1.py`、`run_a2.py`、`gpu_batch.py`、`report_a.py`、`data/`、`results/a1`、`results/a2`） | A1：無前視 walk-forward 基準（3 個匯率 × 有／無前視 × 10 個 fold × 5 個模型 × 20 個種子）；A2：batch 的影響是 batch 本身還是更新次數 | 無前視基準 |
| `B/` | `fastsim.py` 批次模擬器、`check_fastsim.py` 等價性檢查、B1 每步計時、B2 完整訓練計時、零梯度參數分析；報告 `B/README.md` | 系統面分析 |
| `A/REFERENCES.md`、`B/REFERENCES.md` | 參考文獻與重疊程度判斷 | 相關研究 |

## 目前狀態（2026-10-19）

| 項目 | 狀態 |
|---|---|
| 原論文實驗一的重現（Mac、Windows） | 完成 |
| 跨資產前視探針 | 完成 |
| B：模擬器效能剖析 | 完成（`B/README.md`） |
| A1：walk-forward 基準 | 進行中（GPU／CPU 批次執行器 `A/gpu_batch.py`） |
| A2：batch vs. 更新次數 | 進行中（和 A1 一起排程） |
| A 的彙整報告 | A 跑完後執行 `A/report_a.py`，產生 `A/results/summary_a.md` |

## 目前的主要發現

- **前視存在於所有 Yahoo 匯率**：22 個匯率用「當天高低價中點」猜隔天方向，平均 71.2%（61.7%～74.5%）；改用前一天的高低價只剩 52.0%。這些匯率的 K 棒 Close 幾乎等於 Open（中位數 |Close − Open| 只有一天高低價差的 0～17%），代表 Close 記錄的是 K 棒開始時的價格。
- **對照組沒有前視**：股票／ETF 兩種規則都約 51%，加密貨幣都約 53%；它們的 Close 是真正的收盤價。
- **前視足以解釋論文的準確率**：在論文的 EUR/USD 測試集上，零參數規則 RMSE 0.00323、DA 66.8%，接近論文的 QFWP（0.0029、72.3%），而且比論文的 QLSTM 在 batch 32、64 的結果還好。
- **批次加速的上限是模擬器造成的**：PennyLane lightning.qubit 把一個批次拆成每個樣本一條電路；改用真正批次化的模擬器後，完整訓練快 6～61 倍、批次加速比 25～64 倍（原本 2.5 倍），結果和 PennyLane 相同（QLSTM 60／60 次逐位相同）。
- **參數量相同的比較不公平**：QLSTM 32 個參數中有 16 個、QFWP 33 個中有 8 個，梯度永遠是 0。
- **結果會隨硬體改變**：同一份程式、資料與種子，Mac 和 Windows 有 18 次 QFWP 訓練結果不同，使 batch 4、64 的顯著性不穩定。
- A1、A2 的結果待實驗完成後補上。

## 重跑方式

環境（Windows，在本資料夾用 Git Bash）：

```bash
py -3.11 -m venv qfin-main/.venv-pl
qfin-main/.venv-pl/Scripts/python.exe -m pip install -r qfin-main/requirements-pl.txt yfinance
```

GPU 環境（選用，給 `A/gpu_batch.py` 與 `B` 的 GPU 測試）：`py -3.11 -m venv B/.venv-gpu`，再安裝 `torch numpy pandas scipy`（torch 用 `--index-url https://download.pytorch.org/whl/cu128`）。

```bash
cd A
../qfin-main/.venv-pl/Scripts/python.exe probe_leakage.py      # 前視探針（幾分鐘）
../qfin-main/.venv-pl/Scripts/python.exe fetch_data.py         # A 用的 3 個匯率
../B/.venv-gpu/Scripts/python.exe gpu_batch.py --procs 6       # A1＋A2（也可改用 run_a1.py、run_a2.py 只用 CPU）
../qfin-main/.venv-pl/Scripts/python.exe report_a.py           # 彙整
```

原論文的重現步驟見 `qfin-main/README.md`、`qfin-main/REPRODUCTION_WIN.md`；B 的步驟見 `B/README.md`。

## 相關文件

- [新論文的突破策略與投稿設計](https://claude.ai/code/artifact/cbcf5076-6d03-4d70-9ebd-d0170d1c2da1)
- [兩篇論文的痛點與新論文方向](https://claude.ai/code/artifact/d17dcfc8-2d23-4782-a022-1cdd99179598)
- [論文重現報告講稿](https://claude.ai/code/artifact/00809585-dfaf-4ce4-9f9e-5843fceddfcc)
- 原論文：Batched Training for QLSTM vs. QFWP（[arXiv 2512.21820](https://arxiv.org/abs/2512.21820)）
