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
FMP_API_KEY = "1eA8Evcu6mUTfbv2CuV4fdlBBzAQ599j"
LIMIT_TICKERS = 50  # Change to 500 when you are ready to launch!
MAX_WORKERS = 4
BATCH_SIZE = 10
US_RISK_FREE_RATE = 4.3

st.set_page_config(layout="wide", page_title="S&P 500 Terminal V7.5", page_icon="🏦")

# ==========================================================
# HELPERS (The Safety Shield)
# ==========================================
def safe_fmt(v, prefix="", suffix=""):
    """Prevents TypeError by checking for None/NaN before formatting."""
    if pd.isna(v) or v is None:
        return "N/A"
    try:
        return f"{prefix}{float(v):,.2f}{suffix}"
    except:
        return "N/A"

def safe_request(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200: return r.json()
    except: return None
    return None

# ==========================================================
# 1. LOAD TICKERS (Wikipedia + 403 Safety)
# ==========================================================
@st.cache_data(ttl=86400)
def load_sp500():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        html_data = requests.get(url, headers=headers, timeout=10).text
        df = pd.read_html(io.StringIO(html_data))[0]
        df = df.rename(columns={"Symbol":"Ticker","Security":"Name","GICS Sector":"Sector"})
        df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)
        return df.head(LIMIT_TICKERS)
    except Exception as e:
        st.error(f"Wikipedia Fetch Failed: {e}")
        return pd.DataFrame()

# ==========================================================
# 2. FETCH DATA (Parallel FMP Engine)
# ==========================================================
def get_data(t):
    d = {"Ticker": t}
    q = safe_request(f"https://financialmodelingprep.com/api/v3/quote/{t}?apikey={FMP_API_KEY}")
    m = safe_request(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{t}?apikey={FMP_API_KEY}")

    if q and isinstance(q, list) and len(q) > 0:
        val = q[0]
        d.update({"Price": val.get("price"), "PE": val.get("pe"), "MarketCap": val.get("marketCap"), "EPS": val.get("eps")})

    if m and isinstance(m, list) and len(m) > 0:
        val = m[0]
        d.update({"PB": val.get("pbRatioTTM"), "ROE": val.get("roeTTM"), "PEG": val.get("pegRatioTTM"),
                  "FCF_Yield": val.get("freeCashFlowYieldTTM"), "BVPS": val.get("bookValuePerShareTTM")})
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
        time.sleep(0.4)
    return pd.DataFrame(results)

# ==========================================================
# 3. SCORING ENGINE (Sector-Relative)
# ==========================================================
def add_scores(df):
    required = ["PE", "PB", "ROE", "FCF_Yield"]
    for col in required:
        if col not in df.columns: df[col] = None
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sector Averages
    df = df[df["Sector"].notna()]
    sec = df.groupby("Sector")[required].mean().add_suffix('_sec').reset_index()
    df = df.merge(sec, on="Sector", how="left")

    # Ranking logic
    df["PE_score"] = (df["PE_sec"] / df["PE"]).rank(pct=True) if "PE_sec" in df.columns else 0
    df["PB_score"] = (df["PB_sec"] / df["PB"]).rank(pct=True) if "PB_sec" in df.columns else 0
    df["ROE_score"] = df["ROE"].rank(pct=True)
    df["FCF_score"] = df["FCF_Yield"].rank(pct=True)

    df["Value"] = (df["PE_score"].fillna(0) * 0.5 + df["PB_score"].fillna(0) * 0.5) * 100
    df["Quality"] = (df["ROE_score"].fillna(0) * 0.6 + df["FCF_score"].fillna(0) * 0.4) * 100
    df["Total_Score"] = (df["Value"] * 0.4 + df["Quality"] * 0.6)
    return df

# ==========================================================
# 4. VALUATION MATH
# ==========================================================
def graham_calc(eps, bvps):
    if pd.notna(eps) and pd.notna(bvps) and eps > 0 and bvps > 0:
        return (22.5 * eps * bvps) ** 0.5
    return None

def dcf_calc(fcf_yield, mcap, price, g, r, tg):
    if not fcf_yield or not mcap or not price or price <= 0: return None
    fcf, shares = fcf_yield * mcap, mcap / price
    gr, dr, tgr = g/100, r/100, tg/100
    pv = sum(fcf*(1+gr)**yr/(1+dr)**yr for yr in range(1, 11))
    tv = (fcf*(1+gr)**10*(1+tgr))/(dr-tgr)
    pv += tv/(1+dr)**10
    return pv/shares

# ==========================================================
# MAIN UI
# ==========================================================
st.title("📊 S&P 500 Intelligence Terminal V7.5")

sp_data = load_sp500()
df_raw = fetch_all(sp_data)

if df_raw.empty:
    st.error("API Limit Reached or Network Error.")
    st.stop()

df = pd.merge(df_raw, sp_data, on="Ticker")
df = add_scores(df)

# Filters
with st.sidebar:
    st.header("⚙️ Settings")
    sector_list = ["All"] + sorted(df["Sector"].dropna().unique().tolist())
    sel_sector = st.selectbox("Sector", sector_list)
    max_pe = st.slider("Max P/E", 0, 100, 100)

fdf = df.copy()
if sel_sector != "All": fdf = fdf[fdf["Sector"] == sel_sector]
fdf = fdf[(fdf["PE"].isna()) | (fdf["PE"] <= max_pe)]

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["📋 Screener","🗺️ Quality Map","📈 Price Chart","🏦 Deep Dive"])

with tab1:
    st.subheader(" Ranked Opportunities")
    st.dataframe(fdf.sort_values("Total_Score", ascending=False)[["Ticker","Name","Total_Score","PE","ROE","MarketCap"]], 
                 use_container_width=True, hide_index=True)

with tab2:
    st.subheader("🗺️ Valuation vs Profitability")
    fig = px.scatter(fdf.dropna(subset=["PE","ROE"]), x="PE", y="ROE", color="Sector", size="MarketCap", hover_name="Ticker", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("📈 Live Historical Chart (Yahoo)")
    t_chart = st.selectbox("Select Symbol", fdf["Ticker"].unique())
    if t_chart:
        hist = yf.download(t_chart, period="1y", progress=False)
        fig_p = go.Figure(data=[go.Scatter(x=hist.index, y=hist['Close'], line=dict(color="#1f77b4"))])
        fig_p.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_p, use_container_width=True)

with tab4:
    st.subheader("🕵️ Fundamental Terminal")
    t_dive = st.selectbox("Ticker for Deep Dive", fdf["Ticker"].unique(), key="deep_dive")
    row = fdf[fdf["Ticker"] == t_dive].iloc[0]

    st.markdown("### 🔢 DCF Assumptions")
    c1, c2, c3 = st.columns(3)
    g_rate = c1.slider("Growth %", 0, 30, 10)
    d_rate = c2.slider("Discount %", 5, 20, 10)
    t_rate = c3.slider("Terminal %", 1, 5, 3)

    # MATH
    g_val = graham_calc(row.get("EPS"), row.get("BVPS"))
    d_val = dcf_calc(row.get("FCF_Yield"), row.get("MarketCap"), row.get("Price"), g_rate, d_rate, t_rate)

    st.divider()
    m1, m2, m3 = st.columns(3)
    
    # ⚠️ FIXED: Using safe_fmt to prevent crash
    m1.metric("Market Price", safe_fmt(row.get('Price'), prefix="$"))
    m2.metric("Graham Value", safe_fmt(g_val, prefix="$"))
    m3.metric("DCF Value", safe_fmt(d_val, prefix="$"))

    st.caption("The Graham Number and DCF help estimate if a stock is undervalued compared to its actual earnings and cash flow.")
