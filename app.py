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
# ⚠️ PASTE YOUR KEY HERE
FMP_API_KEY = "1eA8Evcu6mUTfbv2CuV4fdlBBzAQ599j"
LIMIT_TICKERS = 50 
MAX_WORKERS = 4
BATCH_SIZE = 10
US_RISK_FREE_RATE = 4.3

st.set_page_config(layout="wide", page_title="S&P 500 V7.2", page_icon="🏦")

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
# LOAD TICKERS (Wikipedia + User-Agent Fix)
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
# FETCH DATA (Balanced Parallel Engine)
# ==========================================================
def get_data(t):
    d = {"Ticker": t}
    q = safe_request(f"https://financialmodelingprep.com/api/v3/quote/{t}?apikey={FMP_API_KEY}")
    m = safe_request(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{t}?apikey={FMP_API_KEY}")

    if q and isinstance(q, list) and len(q) > 0:
        q_data = q[0]
        d.update({
            "Price": q_data.get("price"),
            "PE": q_data.get("pe"),
            "MarketCap": q_data.get("marketCap"),
            "EPS": q_data.get("eps")
        })

    if m and isinstance(m, list) and len(m) > 0:
        m_data = m[0]
        d.update({
            "PB": m_data.get("pbRatioTTM"),
            "ROE": m_data.get("roeTTM"),
            "PEG": m_data.get("pegRatioTTM"),
            "FCF_Yield": m_data.get("freeCashFlowYieldTTM"),
            "Debt_Equity": m_data.get("debtToEquityTTM"),
            "BVPS": m_data.get("bookValuePerShareTTM")
        })
    return d

@st.cache_data(ttl=3600)
def fetch_all(df_tickers):
    tickers = df_tickers["Ticker"].tolist()
    results = []
    progress = st.progress(0)
    
    # Process in smaller batches to respect FMP rate limits
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i+BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(get_data, t) for t in batch]
            for f in as_completed(futures):
                results.append(f.result())
        
        progress.progress(min((i + BATCH_SIZE) / len(tickers), 1.0))
        time.sleep(0.5)  # Throttle to prevent API lockout
        
    return pd.DataFrame(results)

# ==========================================================
# SCORING ENGINE (V7 PRO - Relative Sector Ranking)
# ==========================================================
def add_scores(df):
    if df.empty: return df
    
    # Calculate group averages
    sec = df.groupby("Sector").agg({
        "PE":"mean","PB":"mean","ROE":"mean","FCF_Yield":"mean"
    }).rename(columns={
        "PE":"PE_sec","PB":"PB_sec","ROE":"ROE_sec","FCF_Yield":"FCF_sec"
    }).reset_index()

    df = df.merge(sec, on="Sector", how="left")

    # Higher score for being "Cheaper" than industry average
    df["PE_score"] = (df["PE"] / df["PE_sec"]).rank(pct=True, ascending=False)
    df["PB_score"] = (df["PB"] / df["PB_sec"]).rank(pct=True, ascending=False)
    # Higher score for better Profitability than industry average
    df["ROE_score"] = df["ROE"].rank(pct=True)
    df["FCF_score"] = df["FCF_Yield"].rank(pct=True)

    df["Value"] = (df["PE_score"] * 0.5 + df["PB_score"] * 0.5) * 100
    df["Quality"] = (df["ROE_score"] * 0.6 + df["FCF_score"] * 0.4) * 100

    df["Total_Score"] = (df["Value"] * 0.4 + df["Quality"] * 0.6)
    return df

# ==========================================================
# VALUATION MATH
# ==========================================================
def calc_graham(eps, bvps):
    if eps and bvps and eps > 0 and bvps > 0:
        return (22.5 * eps * bvps) ** 0.5
    return None

def calc_dcf(fcf_yield, mcap, price, g, r, tg):
    if not fcf_yield or not mcap or not price or price <= 0: return None
    fcf = fcf_yield * mcap
    shares = mcap / price
    g_r, r_r, tg_r = g/100, r/100, tg/100
    pv = sum(fcf*(1+g_r)**yr/(1+r_r)**yr for yr in range(1, 11))
    tv = (fcf*(1+g_r)**10*(1+tg_r))/(r_r-tg_r)
    pv += tv/(1+r_r)**10
    return pv/shares

# ==========================================================
# MAIN APP
# ==========================================================
st.title("📊 S&P 500 Intelligence Terminal V7.2")

sp_list = load_sp500()
df_raw = fetch_all(sp_list)
df = pd.merge(df_raw, sp_list, on="Ticker")
df = add_scores(df)

st.caption(f"Universe: {len(df)} stocks • Data Health: {df['Price'].notna().mean()*100:.0f}%")

# ==========================================================
# FILTERS
# ==========================================================
with st.sidebar:
    st.header("⚙️ Dashboard Filters")
    sector_choice = st.selectbox("Industry Sector", ["All"] + sorted(df["Sector"].dropna().unique().tolist()))
    pe_filter = st.slider("Max P/E Ratio", 0, 100, 100)

fdf = df.copy()
if sector_choice != "All":
    fdf = fdf[fdf["Sector"] == sector_choice]
fdf = fdf[(fdf["PE"].isna()) | (fdf["PE"] <= pe_filter)]

# ==========================================================
# TABS
# ==========================================================
tab1, tab2, tab3, tab4 = st.tabs(["📋 Screener","🗺️ Quality Map","📈 Price Chart","🏦 Deep Dive"])

with tab1:
    st.subheader("💰 Ranked Investment Opportunities")
    display_cols = ["Ticker","Name","Total_Score","Value","Quality","PE","ROE"]
    st.dataframe(fdf.sort_values("Total_Score", ascending=False)[display_cols], 
                 use_container_width=True, hide_index=True)

with tab2:
    st.subheader("🗺️ Relative Valuation (P/E vs ROE)")
    fig = px.scatter(fdf.dropna(subset=["PE","ROE"]), x="PE", y="ROE", 
                     color="Sector", size="MarketCap", hover_name="Ticker", 
                     template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("📈 On-Demand Technical Chart")
    sel_ticker = st.selectbox("Select Symbol", fdf["Ticker"].unique())
    if sel_ticker:
        hist = yf.download(sel_ticker, period="1y", progress=False)
        fig_c = go.Figure(data=[go.Scatter(x=hist.index, y=hist['Close'], name="Price", line=dict(color="#1f77b4"))])
        fig_c.update_layout(template="plotly_white", height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_c, use_container_width=True)

with tab4:
    st.subheader("🕵️ Intrinsic Value Terminal")
    t_dive = st.selectbox("Ticker for Deep Dive", fdf["Ticker"].unique(), key="deep")
    row = fdf[fdf["Ticker"] == t_dive].iloc[0]

    g_val = calc_graham(row.get("EPS"), row.get("BVPS"))

    st.markdown("### 🔢 DCF Assumptions")
    c1, c2, c3 = st.columns(3)
    g_rate = c1.slider("Growth %", 0, 30, 10)
    d_rate = c2.slider("Discount %", 5, 20, 10)
    t_rate = c3.slider("Terminal %", 1, 5, 3)

    d_val = calc_dcf(row.get("FCF_Yield"), row.get("MarketCap"), row.get("Price"), g_rate, d_rate, t_rate)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Market Price", f"${row.get('Price', 0):.2f}")
    m2.metric("Graham Fair Value", f"${g_val:.2f}" if g_val else "N/A")
    m3.metric("DCF Intrinsic Value", f"${d_val:.2f}" if d_val else "N/A")
