# ==========================================================
# S&P 500 Intelligence Terminal V7.3 (FINAL STABLE)
# ==========================================================
import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import io

# ==========================================================
# CONFIG
# ==========================================================
FMP_API_KEY = "YOUR_API_KEY"
LIMIT_TICKERS = 50
MAX_WORKERS = 4
BATCH_SIZE = 10

st.set_page_config(layout="wide", page_title="S&P 500 V7.3")

# ==========================================================
# HELPERS
# ==========================================================
def safe_request(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

def fmt(x): return f"{x:,.2f}" if pd.notna(x) else "N/A"
def fmt_pct(x): return f"{x:.1f}%" if pd.notna(x) else "N/A"

# ==========================================================
# LOAD TICKERS (403 SAFE)
# ==========================================================
@st.cache_data(ttl=86400)
def load_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    df = pd.read_html(io.StringIO(html))[0]

    df = df.rename(columns={
        "Symbol":"Ticker",
        "Security":"Name",
        "GICS Sector":"Sector"
    })
    df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)

    return df.head(LIMIT_TICKERS)

# ==========================================================
# FETCH DATA (BATCHED + SAFE)
# ==========================================================
def get_data(t):
    d = {"Ticker": t}

    q = safe_request(f"https://financialmodelingprep.com/api/v3/quote/{t}?apikey={FMP_API_KEY}")
    m = safe_request(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{t}?apikey={FMP_API_KEY}")

    if q and isinstance(q, list) and len(q) > 0:
        q = q[0]
        d["Price"] = q.get("price")
        d["PE"] = q.get("pe")
        d["MarketCap"] = q.get("marketCap")
        d["EPS"] = q.get("eps")

    if m and isinstance(m, list) and len(m) > 0:
        m = m[0]
        d["PB"] = m.get("pbRatioTTM")
        d["ROE"] = m.get("roeTTM")
        d["PEG"] = m.get("pegRatioTTM")
        d["FCF_Yield"] = m.get("freeCashFlowYieldTTM")
        d["Debt_Equity"] = m.get("debtToEquityTTM")
        d["BVPS"] = m.get("bookValuePerShareTTM")

    return d

@st.cache_data(ttl=3600)
def fetch_all(df):
    tickers = df["Ticker"].tolist()
    results = []

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i+BATCH_SIZE]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(get_data, t) for t in batch]
            for f in as_completed(futures):
                results.append(f.result())

        time.sleep(0.5)

    return pd.DataFrame(results)

# ==========================================================
# SCORING ENGINE (ROBUST)
# ==========================================================
def add_scores(df):
    required = ["PE", "PB", "ROE", "FCF_Yield"]

    for col in required:
        if col not in df.columns:
            df[col] = None

    df = df[df["Sector"].notna()]

    sec = df.groupby("Sector").agg({
        col: "mean" for col in required
    }).rename(columns={
        "PE": "PE_sec",
        "PB": "PB_sec",
        "ROE": "ROE_sec",
        "FCF_Yield": "FCF_sec"
    }).reset_index()

    df = df.merge(sec, on="Sector", how="left")

    df["PE_rel"] = df["PE"] / df["PE_sec"]
    df["PB_rel"] = df["PB"] / df["PB_sec"]

    df["PE_score"] = df["PE_rel"].rank(pct=True, ascending=False)
    df["PB_score"] = df["PB_rel"].rank(pct=True, ascending=False)
    df["ROE_score"] = df["ROE"].rank(pct=True)
    df["FCF_score"] = df["FCF_Yield"].rank(pct=True)

    df[["PE_score","PB_score","ROE_score","FCF_score"]] = df[
        ["PE_score","PB_score","ROE_score","FCF_score"]
    ].fillna(0)

    df["Value"] = df["PE_score"] * 0.5 + df["PB_score"] * 0.5
    df["Quality"] = df["ROE_score"] * 0.6 + df["FCF_score"] * 0.4

    df["Total_Score"] = (df["Value"] * 0.4 + df["Quality"] * 0.6) * 100

    return df

# ==========================================================
# VALUATION
# ==========================================================
def graham(eps, bvps):
    if eps and bvps and eps > 0 and bvps > 0:
        return (22.5 * eps * bvps) ** 0.5
    return None

def dcf(fcf_yield, mcap, price, g, r, tg):
    if not fcf_yield or not mcap or not price or price <= 0:
        return None

    fcf = fcf_yield * mcap
    shares = mcap / price

    g, r, tg = g/100, r/100, tg/100

    pv = sum(fcf*(1+g)**yr/(1+r)**yr for yr in range(1, 11))
    tv = (fcf*(1+g)**10*(1+tg))/(r-tg)
    pv += tv/(1+r)**10

    return pv/shares

# ==========================================================
# MAIN
# ==========================================================
st.title("📊 S&P 500 Intelligence Terminal V7.3")

sp = load_sp500()
df_raw = fetch_all(sp)

df = pd.merge(df_raw, sp, on="Ticker")

# Ensure required columns exist BEFORE scoring
for col in ["PE","PB","ROE","FCF_Yield"]:
    if col not in df.columns:
        df[col] = None

df = add_scores(df)

st.caption(f"Data completeness: {df['Price'].notna().mean()*100:.1f}%")

# ==========================================================
# FILTERS
# ==========================================================
sector = st.sidebar.selectbox("Sector", ["All"] + sorted(df["Sector"].dropna().unique()))
pe_max = st.sidebar.slider("Max PE", 0, 100, 50)

fdf = df.copy()
if sector != "All":
    fdf = fdf[fdf["Sector"] == sector]

fdf = fdf[(fdf["PE"].isna()) | (fdf["PE"] <= pe_max)]

# ==========================================================
# TABS
# ==========================================================
tab1, tab2, tab3, tab4 = st.tabs(["📋 Screener","🗺️ Map","📈 Charts","🏦 Deep Dive"])

# TAB 1
with tab1:
    st.dataframe(
        fdf.sort_values("Total_Score", ascending=False),
        use_container_width=True
    )

# TAB 2
with tab2:
    mdf = fdf.dropna(subset=["PE","ROE"])
    fig = px.scatter(mdf, x="PE", y="ROE", color="Sector", size="MarketCap")
    st.plotly_chart(fig, use_container_width=True)

# TAB 3
with tab3:
    t = st.selectbox("Stock", fdf["Ticker"])
    hist = yf.download(t, period="1y", progress=False)
    st.line_chart(hist["Close"])

# TAB 4
with tab4:
    t = st.selectbox("Deep Dive", fdf["Ticker"], key="deep")
    row = fdf[fdf["Ticker"] == t].iloc[0]

    g_val = graham(row.get("EPS"), row.get("BVPS"))

    c1, c2 = st.columns(2)
    growth = c1.slider("Growth %", 0, 30, 10)
    discount = c2.slider("Discount %", 5, 20, 10)
    terminal = st.slider("Terminal %", 1, 5, 3)

    d_val = dcf(
        row.get("FCF_Yield"),
        row.get("MarketCap"),
        row.get("Price"),
        growth, discount, terminal
    )

    st.metric("Price", fmt(row.get("Price")))
    st.metric("Graham", fmt(g_val))
    st.metric("DCF", fmt(d_val))
