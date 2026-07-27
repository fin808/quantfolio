import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from quantfolio.config import TICKERS, UNIVERSE
from quantfolio.data.prices import fetch_prices, to_returns
from quantfolio.data.universe import align
from quantfolio.estimation import annualised_mean, sample_cov, shrunk_cov
from quantfolio.optimise import (Constraints, efficient_frontier,
                                 min_variance, max_sharpe, summarise)
from quantfolio.backtest import backtest, performance

st.set_page_config(page_title="Quantfolio", layout="wide")


@st.cache_data
def load(tickers):
    return align(fetch_prices(list(tickers)), mode="common")


st.sidebar.header("Universe")
chosen = st.sidebar.multiselect("Assets", TICKERS, default=TICKERS)

st.sidebar.header("Constraints")
w_max = st.sidebar.slider("Max position", 0.05, 1.0, 0.25, 0.05)
use_sectors = st.sidebar.checkbox("Sector limits")
cov_method = st.sidebar.radio("Covariance", ["sample", "ledoit-wolf"])
rf = st.sidebar.number_input("Risk-free rate", 0.0, 0.10, 0.02, 0.005)

if len(chosen) < 2:
    st.warning("Pick at least two assets.")
    st.stop()

prices = load(tuple(chosen))
rets = to_returns(prices)
mu = annualised_mean(rets)
cov = sample_cov(rets) if cov_method == "sample" else shrunk_cov(rets)[0]

limits = {}
if use_sectors:
    st.sidebar.caption("Group bounds")
    for label in UNIVERSE:
        if any(t in chosen for t in UNIVERSE[label]):
            limits[label] = st.sidebar.slider(label, 0.0, 1.0, (0.0, 0.5), 0.05)

cons = Constraints(w_max=w_max, groups=UNIVERSE, group_limits=limits)

st.title("Portfolio construction")
st.caption(f"{prices.index[0].date()} to {prices.index[-1].date()} · "
           f"{len(prices):,} observations · {len(chosen)} assets")

tab1, tab2, tab3 = st.tabs(["Frontier", "Weights", "Backtest"])

with tab1:
    try:
        ef = efficient_frontier(mu, cov, cons)
        mv, ms = min_variance(cov, cons), max_sharpe(mu, cov, cons, rf=rf)
    except RuntimeError as e:
        st.error(f"Infeasible: {e}")
        st.stop()

    mv_s, ms_s = summarise(mv, mu, cov, rf), summarise(ms, mu, cov, rf)
    c1, c2, c3 = st.columns(3)
    c1.metric("Max-Sharpe ratio", ms_s["sharpe"])
    c2.metric("Return / vol", f"{ms_s['ret']:.1%} / {ms_s['vol']:.1%}")
    c3.metric("Effective N", ms_s["effective_n"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ef["vol"], ef["ret"], lw=2)
    vols = rets.std() * (252 ** 0.5)
    ax.scatter(vols, mu, s=20, c="grey")
    for t in mu.index:
        ax.annotate(t, (vols[t], mu[t]), fontsize=7)
    ax.scatter(mv_s["vol"], mv_s["ret"], marker="*", s=180, c="tab:blue")
    ax.scatter(ms_s["vol"], ms_s["ret"], marker="*", s=180, c="tab:red")
    ax.set_xlabel("annualised volatility")
    ax.set_ylabel("annualised return")
    st.pyplot(fig)

with tab2:
    w = pd.DataFrame({"min variance": mv, "max sharpe": ms})
    st.bar_chart(w[w.sum(axis=1) > 1e-4])
    st.dataframe(w.round(4).style.format("{:.2%}"))
    st.json({"min variance": mv_s, "max sharpe": ms_s})

with tab3:
    st.caption("Walk-forward, quarterly rebalancing, 10bps costs.")
    lookback = st.select_slider("Lookback (days)", [252, 504, 756, 1260], 756)
    if st.button("Run backtest"):
        def eq(win):
            return pd.Series(1 / win.shape[1], index=win.columns)

        strategies = {
            "1/N": eq,
            "min vol": lambda win: min_variance(sample_cov(win), cons),
            "max sharpe": lambda win: max_sharpe(
                annualised_mean(win), sample_cov(win), cons, rf=rf),
        }
        curves, table = {}, {}
        bar = st.progress(0.0)
        for i, (name, fn) in enumerate(strategies.items(), 1):
            r, _, t = backtest(prices, fn, lookback=lookback)
            curves[name] = (1 + r).cumprod()
            table[name] = {**performance(r, rf), "turnover": round(t.mean(), 3)}
            bar.progress(i / len(strategies))
        st.line_chart(pd.DataFrame(curves))
        st.dataframe(pd.DataFrame(table).T)