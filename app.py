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
# CONFIG & API
# ==========================================================
# ⚠️ PASTE YOUR KEY HERE
FMP_API_KEY = "1eA8Evcu6mUTfbv2CuV4fdlBBzAQ599j"

LIMIT_TICKERS = 50  # Change to 500 for full production launch
MAX_WORKERS = 4
BATCH_SIZE = 10
US_RISK_FREE_RATE = 4.3

st.set_page_config(layout="wide", page_title="S&P 500 Intelligence Terminal V7.4", page_icon="🏦")

# ==========================================================
# HELPERS
# ==========================================================
def safe_request(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200: return r.json()
    except: return None
    return None

def fmt(x): return f"{x:,.2f}" if pd.notna(x) else "N/A"
def fmt_pct(x): return f"{x:.1f}%" if pd.notna(x) else "N/A"

# ==========================================================
# 1. LOAD TICKERS (Wikipedia + 403 Safety)
# ==========================================================
@st.cache_data(ttl=86400)
def load_sp500():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers, timeout=10).text
        df = pd.read_html(io.StringIO(html))[0]
        df = df.rename(columns={"Symbol":"Ticker","Security":"Name","GICS Sector":"Sector"})
        df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)
        return df.head(LIMIT_TICKERS)
    except Exception as e:
        st.error(f"List Fetch Failed: {e}")
        return pd.DataFrame()

# ==========================================================
# 2. FETCH DATA (Balanced Batching Engine)
# ==========================================================
def get_data(t):
    d = {"Ticker": t}
    q = safe_request(f"https://financialmodelingprep.com/api/v3/quote/{t}?apikey={FMP_API_KEY}")
    m = safe_request(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{t}?apikey={FMP_API_KEY}")

    if q and isinstance(q, list) and len(q) > 0:
        q_data = q[0]
        d.update({"Price": q_data.get("price"), "PE": q_data.get("pe"), "MarketCap": q_data.get("marketCap"), "EPS": q_data.get("eps")})

    if m and isinstance(m, list) and len(m) > 0:
        m_data = m[0]
        d.update({"PB": m_data.get("pbRatioTTM"), "ROE": m_data.get("roeTTM"), "PEG": m_data.get("pegRatioTTM"),
                  "FCF_Yield": m_data.get("freeCashFlowYieldTTM"), "BVPS": m_data.get("bookValuePerShareTTM")})
    return d

@st.cache_data(ttl=3600)
def fetch_all(df_list):
    tickers = df_list["Ticker"].tolist()
    results = []
    progress = st.progress(0)
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i+BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(get_data, t) for t in batch]
            for f in as_completed(futures): results.append(f.result())
        progress.progress(min((i + BATCH_SIZE) / len(tickers), 1.0))
        time.sleep(0.5)
    return pd.DataFrame(results)

# ==========================================================
# 3. SCORING ENGINE (V7.4 Robust Version)
# ==========================================================
def add_scores(df):
    required = ["PE", "PB", "ROE", "FCF_Yield"]
    for col in required:
        if col not in df.columns: df[col] = None
    
    df = df[df["Sector"].notna()]
    
    # Sector Averages
    sec = df.groupby("Sector").agg({col: "mean" for col in required}).rename(columns={c: f"{c}_sec" for c in required}).reset_index()
    df = df.merge(sec, on="Sector", how="left")

    # Relative Ranking
    df["PE_score"] = (df["PE"] / df["PE_sec"]).rank(pct=True, ascending=False)
    df["PB_score"] = (df["PB"] / df["PB_sec"]).rank(pct=True, ascending=False)
    df["ROE_score"] = df["ROE"].rank(pct=True)
    df["FCF_score"] = df["FCF_Yield"].rank(pct=True)

    # Fill NaNs with 0 for logic
    score_cols = ["PE_score","PB_score","ROE_score","FCF_score"]
    df[score_cols] = df[score_cols].fillna(0)

    # Weights: Value(40%) / Quality(60%)
    df["Value"] = (df["PE_score"] * 0.5 + df["PB_score"] * 0.5) * 100
    df["Quality"] = (df["ROE_score"] * 0.6 + df["FCF_score"] * 0.4) * 100
    df["Total_Score"] = (df["Value"] * 0.4 + df["Quality"] * 0.6)
    return df

# ==========================================================
# 4. VALUATION MATH
# ==========================================================
def graham(eps, bvps):
    if eps and bvps and eps > 0 and bvps > 0: return (22.5 * eps * bvps) ** 0.5
    return None

def dcf(fcf_yield, mcap, price, g, r, tg):
    if not fcf_yield or not mcap or not price or price <= 0: return None
    fcf, shares = fcf_yield * mcap, mcap / price
    g_r, r_r, tg_r = g/100, r/100, tg/100
    pv = sum(fcf*(1+g_r)**yr/(1+r_r)**yr for yr in range(1, 11))
    tv = (fcf*(1+g_r)**10*(1+tg_r))/(r_r-tg_r)
    pv += tv/(1+r_r)**10
    return pv/shares

# ==========================================================
# MAIN UI
# ==========================================================
st.title("📊 S&P 500 Intelligence Terminal V7.4")

sp = load_sp500()
df_raw = fetch_all(sp)

if df_raw.empty:
    st.error("No data fetched. Check API key.")
    st.stop()

df = pd.merge(df_raw, sp, on="Ticker")

# Column Enforcement
required_cols = ["Price","PE","PB","ROE","FCF_Yield","MarketCap","EPS","BVPS"]
for col in required_cols:
    if col not in df.columns: df[col] = None

df = add_scores(df)
completeness = df["Price"].notna().mean()*100
st.caption(f"Universe: {len(df)} stocks • Data Health: {completeness:.0f}%")

# Filters
sector_list = ["All"] + sorted(df["Sector"].dropna().unique().tolist())
sector = st.sidebar.selectbox("Sector", sector_list)
pe_max = st.sidebar.slider("Max PE", 0, 100, 100)

fdf = df.copy()
if sector != "All": fdf = fdf[fdf["Sector"] == sector]
fdf = fdf[(fdf["PE"].isna()) | (fdf["PE"] <= pe_max)]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Screener","🗺️ Quality Map","📈 Price Chart","🏦 Deep Dive"])

with tab1:
    st.subheader(" Ranked Opportunities")
    st.dataframe(fdf.sort_values("Total_Score", ascending=False)[["Ticker","Name","Total_Score","Value","Quality","PE","ROE"]], 
                 use_container_width=True, hide_index=True)

with tab2:
    st.subheader("🗺️ Valuation vs Quality")
    fig = px.scatter(fdf.dropna(subset=["PE","ROE"]), x="PE", y="ROE", color="Sector", size="MarketCap", hover_name="Ticker", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("📈 Live Historical Chart")
    t_chart = st.selectbox("Select Ticker", fdf["Ticker"].unique())
    if t_chart:
        hist = yf.download(t_chart, period="1y", progress=False)
        fig_p = go.Figure(data=[go.Scatter(x=hist.index, y=hist['Close'], name="Price")])
        fig_p.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_p, use_container_width=True)

with tab4:
    st.subheader("🕵️ Fundamental Deep Dive")
    t_dive = st.selectbox("Analyze Ticker", fdf["Ticker"].unique(), key="deep")
    row = fdf[fdf["Ticker"] == t_dive].iloc[0]
    g_val = graham(row.get("EPS"), row.get("BVPS"))

    c1, c2, c3 = st.columns(3)
    growth = c1.slider("FCF Growth %", 0, 30, 10)
    discount = c2.slider("Discount %", 5, 20, 10)
    terminal = c3.slider("Terminal %", 1, 5, 3)

    d_val = dcf(row.get("FCF_Yield"), row.get("MarketCap"), row.get("Price"), growth, discount, terminal)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Market Price", f"${row.get('Price', 0):.2f}")
    m2.metric("Graham Fair Value", f"${g_val:.2f}" if g_val else "N/A")
    m3.metric("DCF Intrinsic Value", f"${d_val:.2f}" if d_val else "N/A")
