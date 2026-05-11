import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# --- 1. 設定網頁標題與 Session State ---
st.set_page_config(page_title="智能投資組合優化器", layout="wide")
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# ==========================================
# 🔐 登入邏輯
# ==========================================
if not st.session_state.authenticated:
    st.title('🔒 系統登入')
    st.markdown("請輸入授權碼以存取高階回測功能。")
    password = st.text_input("🔑 請輸入系統密碼 (Access Code)", type="password")
    if password:
        if password == "5428":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("⛔ 密碼錯誤")
    st.stop()

# ==========================================
# 🚀 主程式
# ==========================================
st.title('📈 智能投資組合優化器 (勝率視覺化版)')
st.markdown("""
本系統結合 **「數學優化」** 與 **「蒙地卡羅模擬」**，並透過 **「持有期間勝率分析」** 驗證長期投資價值。
""")

# --- 2. 參數設定 ---
st.sidebar.header('1. 標的選擇')
tickers_input = st.sidebar.text_input('股票/基金代號 (請用空白隔開)', 'VFIAX VBTLX TSLA NVDA')
user_tickers = tickers_input.upper().split()

st.sidebar.header('2. 基準指數')
bench_input = st.sidebar.text_input('基準代號', 'SPY:60 AGG:40')
years = st.sidebar.slider('回測/預測年數', 1, 20, 10)
risk_free_rate = 0.02

# --- 融資設定 ---
st.sidebar.markdown("---")
st.sidebar.header("3. 融資設定")
use_margin = st.sidebar.checkbox("開啟融資回測模式")
if use_margin:
    loan_ratio = st.sidebar.slider("融資成數", 0.0, 0.9, 0.6, 0.1)
    margin_rate = st.sidebar.number_input("融資年利率 (%)", 2.0, 15.0, 6.0, 0.1) / 100
    self_fund_ratio = 1 - loan_ratio
    if self_fund_ratio <= 0.01: self_fund_ratio = 0.01
    leverage = 1 / self_fund_ratio
    st.sidebar.info(f"槓桿倍數：**{leverage:.1f} 倍**")
else:
    loan_ratio, margin_rate, leverage = 0.0, 0.0, 1.0

# --- 4. 策略混合器 ---
st.sidebar.markdown("---")
st.sidebar.header("4. 策略混合權重 (Strategy Mix)")
st.sidebar.caption("調整兩種演算法在最終投組中的佔比")
mc_weight_ratio = st.sidebar.slider("蒙地卡羅 (MC) 佔比", 0.0, 1.0, 0.4, 0.1)
sharpe_weight_ratio = 1.0 - mc_weight_ratio
st.sidebar.text(f"配置：MC {mc_weight_ratio:.0%} + MaxSharpe {sharpe_weight_ratio:.0%}")

# --- 5. 投資金額 ---
st.sidebar.markdown("---")
st.sidebar.header("5. 投資金額")
initial_investment = st.sidebar.number_input("初始本金 ($)", value=100000, step=10000)

# --- 3. 核心邏輯 ---
if st.sidebar.button('開始計算'):
    if len(user_tickers) < 2:
        st.error("請至少輸入兩檔標的。")
    else:
        with st.spinner('正在進行雙軌運算 (數學優化 + 3000次隨機模擬)...'):
            try:
                # ==========================
                # A. 數據準備
                # ==========================
                end_date = datetime.today()
                start_date = end_date - timedelta(days=365*years + 365)

                data = yf.download(user_tickers, start=start_date, end=end_date, auto_adjust=True)
                if 'Close' in data.columns: df_close = data['Close']
                else: df_close = data
                df_close.dropna(inplace=True)

                if df_close.empty: st.stop()
                if df_close.index.tz is not None: df_close.index = df_close.index.tz_localize(None)
                tickers = df_close.columns.tolist()

                # Benchmark
                bench_config = []
                try:
                    items = bench_input.strip().split()
                    for item in items:
                        if ':' in item: parts = item.split(':'); ticker = parts[0].upper(); weight = float(parts[1])
                        else: ticker = item.upper(); weight = 100.0
                        bench_config.append({'ticker': ticker, 'weight': weight})
                    total_bw = sum([x['weight'] for x in bench_config]) or 1
                    for x in bench_config: x['weight'] /= total_bw
                    bench_tickers = [x['ticker'] for x in bench_config]
                    bench_weights = [x['weight'] for x in bench_config]
                except: st.stop()

                bench_data = yf.download(bench_tickers, start=start_date, end=end_date, auto_adjust=True)
                if 'Close' in bench_data.columns: df_bench = bench_data['Close']
                else: df_bench = bench_data
                if isinstance(df_bench, pd.Series): df_bench = df_bench.to_frame(name=bench_tickers[0])
                if df_bench.index.tz is not None: df_bench.index = df_bench.index.tz_localize(None)

                common_index = df_close.index.intersection(df_bench.index)
                df_close = df_close.loc[common_index]
                df_bench = df_bench.loc[common_index]

                if df_bench.empty: normalized_bench = None
                else:
                    b_ret = df_bench.pct_change().fillna(0)
                    try: comp_b_ret = b_ret[bench_tickers].dot(bench_weights)
                    except: comp_b_ret = b_ret.mean(axis=1)
                    normalized_bench = (1 + comp_b_ret).cumprod()
                    normalized_bench.name = "基準指數"

                # 統計數據
                daily_ret = df_close.pct_change().dropna()
                cov_matrix = daily_ret.cov() * 252
                mean_returns = daily_ret.mean() * 252
                normalized_prices = df_close / df_close.iloc[0]
                num_assets = len(tickers)

                # 函數定義
                def calculate_margin_equity(raw_portfolio_value, leverage, loan_ratio, annual_rate):
                    if leverage == 1: return raw_portfolio_value
                    debt = leverage - 1
                    daily_rate = annual_rate / 365
                    position_value = raw_portfolio_value * leverage
                    interest_cost = pd.Series(np.arange(len(raw_portfolio_value)) * debt * daily_rate, index=raw_portfolio_value.index)
                    margin_equity = position_value - debt - interest_cost
                    return margin_equity

                # ==========================
                # B1. 策略一：最大夏普 (Solver)
                # ==========================
                def neg_sharpe(w, m_ret, cov, rf):
                    ret = np.sum(m_ret * w)
                    vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
                    return -(ret - rf) / vol

                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0, 1) for _ in range(num_assets))
                init_guess = [1/num_assets] * num_assets

                res_sharpe = minimize(neg_sharpe, init_guess, args=(mean_returns, cov_matrix, risk_free_rate),
                                      method='SLSQP', bounds=bounds, constraints=constraints)
                w_sharpe = res_sharpe.x

                ret_sharpe = np.sum(mean_returns * w_sharpe)
                vol_sharpe = np.sqrt(np.dot(w_sharpe.T, np.dot(cov_matrix, w_sharpe)))

                # ==========================
                # B2. 策略二：蒙地卡羅 (MC)
                # ==========================
                num_sims = 3000
                rand_w = np.random.random((num_sims, num_assets))
                rand_w = rand_w / rand_w.sum(axis=1)[:, None]
                port_ret = np.dot(rand_w, mean_returns)
                port_vol = np.zeros(num_sims)
                for i in range(num_sims):
                    port_vol[i] = np.sqrt(np.dot(rand_w[i].T, np.dot(cov_matrix, rand_w[i])))

                port_sharpe = (port_ret - risk_free_rate) / port_vol
                best_mc_idx = port_sharpe.argmax()
                w_mc = rand_w[best_mc_idx]
                ret_mc = port_ret[best_mc_idx]
                vol_mc = port_vol[best_mc_idx]

                # ==========================
                # B3. 混合策略 (Blending)
                # ==========================
                w_final = (w_mc * mc_weight_ratio) + (w_sharpe * sharpe_weight_ratio)
                ret_final = np.sum(mean_returns * w_final)
                vol_final = np.sqrt(np.dot(w_final.T, np.dot(cov_matrix, w_final)))

                st.success(f"混合運算完成！(MC: {mc_weight_ratio:.0%} / Solver: {sharpe_weight_ratio:.0%})")

                # ==========================
                # C. 顯示區塊 (版面重構)
                # ==========================
                st.subheader("📊 策略分析結果")

                col_top1, col_top2 = st.columns(2)
                with col_top1:
                    st.markdown("#### 1. 策略權重比較")
                    df_comp = pd.DataFrame({
                        '標的': tickers,
                        '🎲 MC最佳解': [f"{x:.1%}" for x in w_mc],
                        '🚀 最大夏普': [f"{x:.1%}" for x in w_sharpe],
                        '🏆 最終混合': [f"{x:.1%}" for x in w_final]
                    })
                    st.dataframe(df_comp, hide_index=True, use_container_width=True)
                with col_top2:
                    st.markdown("#### 2. 預期數據比較 (再平衡模式)")
                    st.info(f"""
                    **🏆 最終混合投組 (數學預期)**
                    * 預期年化報酬：**{ret_final:.2%}**
                    * 預期年化波動：**{vol_final:.2%}**
                    * *註：此為固定權重再平衡之數學期望值*
                    """)
                    st.markdown("---")
                    col_in1, col_in2 = st.columns(2)
                    col_in1.write(f"**🎲 MC 最佳解**")
                    col_in1.caption(f"報酬: {ret_mc:.1%} | 波動: {vol_mc:.1%}")
                    col_in2.write(f"**🚀 最大夏普解**")
                    col_in2.caption(f"報酬: {ret_sharpe:.1%} | 波動: {vol_sharpe:.1%}")

                st.markdown("---")
                st.subheader("☁️ 效率前緣與策略落點")
                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(x=port_vol, y=port_ret, mode='markers', marker=dict(color=port_sharpe, colorscale='Viridis', size=5), name='隨機投組'))
                fig_ef.add_trace(go.Scatter(x=[vol_mc], y=[ret_mc], mode='markers+text', marker=dict(color='orange', size=15, symbol='star'), name='MC 最佳解', text=['MC'], textposition="top center"))
                fig_ef.add_trace(go.Scatter(x=[vol_sharpe], y=[ret_sharpe], mode='markers+text', marker=dict(color='red', size=15, symbol='diamond'), name='最大夏普', text=['Sharpe'], textposition="bottom center"))
                fig_ef.add_trace(go.Scatter(x=[vol_final], y=[ret_final], mode='markers+text', marker=dict(color='blue', size=18, symbol='circle'), name='最終混合', text=['Final'], textposition="middle right"))
                fig_ef.update_layout(xaxis_title="年化波動度 (Risk)", yaxis_title="年化報酬率 (Return)", height=500)
                st.plotly_chart(fig_ef, use_container_width=True)

                # ==========================
                # D. 回測 (Buy & Hold)
                # ==========================
                raw_port_val = (normalized_prices * w_final).sum(axis=1)
                margin_port_val = calculate_margin_equity(raw_port_val, leverage, loan_ratio, margin_rate)
                margin_port_val.name = "🏆 混合策略投組"

                st.markdown("---")
                st.subheader("📈 資產成長回測 (買入持有模式)")
                st.caption("註：買入持有模式下，強勢股權重會隨時間增加，故歷史報酬通常高於固定權重的數學預期。")

                fig_bt = px.line(margin_port_val, title='混合策略 vs Benchmark')
                fig_bt.update_traces(line=dict(color='blue', width=3))
                if normalized_bench is not None:
                    aligned_bench = normalized_bench.reindex(margin_port_val.index).ffill()
                    if aligned_bench.iloc[0] > 0: aligned_bench = aligned_bench / aligned_bench.iloc[0]
                    fig_bt.add_trace(go.Scatter(x=aligned_bench.index, y=aligned_bench, mode='lines', name=f'基準 ({bench_input})', line=dict(color='gray', width=2, dash='dash')))
                st.plotly_chart(fig_bt, use_container_width=True)

                def calculate_avg_annual_ret(series):
                    temp = series.copy()
                    if temp.index.tz is not None: temp.index = temp.index.tz_localize(None)
                    ann = temp.resample('YE').last().pct_change().dropna()
                    curr_yr = datetime.now().year
                    if curr_yr in ann.index.year: ann = ann[ann.index.year != curr_yr]
                    return ann.mean()

                def calculate_mdd(series):
                    roll_max = series.cummax()
                    dd = (series - roll_max) / roll_max
                    return dd.min()

                total_ret = margin_port_val.iloc[-1] - 1
                avg_ret_hist = calculate_avg_annual_ret(margin_port_val)
                vol_hist = margin_port_val.pct_change().dropna().std() * np.sqrt(252)
                mdd = calculate_mdd(margin_port_val)

                # ★ 新增：Sharpe Ratio 計算
                sharpe_hist = (avg_ret_hist - risk_free_rate) / vol_hist

                # ★ 修改：5個 metric 卡片（新增 Sharpe Ratio）
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("總報酬率", f"{total_ret:.2%}")
                c2.metric("平均年報酬 (歷史)", f"{avg_ret_hist:.2%}")
                c3.metric("年化波動 (歷史)", f"{vol_hist:.2%}")
                c4.metric("最大回撤 (MDD)", f"{mdd:.2%}")
                c5.metric("Sharpe Ratio", f"{sharpe_hist:.2f}")

                # ★ 新增：MDD 視覺化走勢圖
                st.markdown("#### 📉 最大回撤走勢圖")
                roll_max = margin_port_val.cummax()
                drawdown = (margin_port_val - roll_max) / roll_max

                fig_mdd = go.Figure()
                fig_mdd.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    fill='tozeroy',
                    mode='lines',
                    line=dict(color='#d62728', width=1),
                    fillcolor='rgba(214, 39, 40, 0.2)',
                    name='回撤幅度'
                ))
                fig_mdd.update_layout(
                    yaxis=dict(tickformat=".0%"),
                    yaxis_title="回撤幅度",
                    xaxis_title="日期",
                    height=300,
                    hovermode="x unified"
                )
                st.plotly_chart(fig_mdd, use_container_width=True)

                if use_margin:
                    st.markdown("---")
                    st.subheader(f"💰 融資效益視覺化 (本金 ${initial_investment:,.0f})")
                    v1, v2 = st.columns(2)
                    own = initial_investment
                    loan = own * (leverage - 1)
                    end_no_marg = own * raw_port_val.iloc[-1]
                    end_marg = own * margin_port_val.iloc[-1]
                    with v1:
                        fg = go.Figure()
                        fg.add_trace(go.Bar(name='自有', x=['無融資'], y=[own], marker_color='#2ca02c'))
                        fg.add_trace(go.Bar(name='自有', x=['有融資'], y=[own], marker_color='#2ca02c', showlegend=False))
                        fg.add_trace(go.Bar(name='借款', x=['有融資'], y=[loan], marker_color='#d62728'))
                        fg.update_layout(barmode='stack', title=f'初始購買力 ({leverage:.1f}x)', height=300)
                        st.plotly_chart(fg, use_container_width=True)
                    with v2:
                        fg2 = go.Figure()
                        fg2.add_trace(go.Bar(x=['無融資', '有融資'], y=[end_no_marg, end_marg], marker_color=['#1f77b4', '#ff7f0e']))
                        fg2.update_layout(title='期末淨值比較', height=300)
                        st.plotly_chart(fg2, use_container_width=True)

                # ==========================================
                # 持有期間 vs 正報酬機率圖
                # ==========================================
                st.markdown("---")
                st.subheader("⏳ 長期持有勝率分析 (Holding Period vs Win Rate)")
                st.caption("此圖顯示：隨著持有時間拉長，獲得正報酬的機率變化。")

                win_rates = []
                years_range = range(1, 11)

                for y in years_range:
                    window = int(y * 252)
                    if len(margin_port_val) > window:
                        roll_ret = margin_port_val.pct_change(window).dropna()
                        win_rate = (roll_ret > 0).mean()
                        win_rates.append(win_rate)
                    else:
                        win_rates.append(0)

                fig_win = go.Figure()
                fig_win.add_trace(go.Bar(
                    x=[f"{y}年" for y in years_range],
                    y=win_rates,
                    text=[f"{w:.1%}" for w in win_rates],
                    textposition='auto',
                    marker_color='#2ca02c'
                ))
                fig_win.update_layout(
                    title="持有年數 vs 正報酬機率",
                    xaxis_title="持有期間",
                    yaxis_title="正報酬機率 (%)",
                    yaxis=dict(tickformat=".0%"),
                    height=400
                )
                st.plotly_chart(fig_win, use_container_width=True)

                with st.expander("查看詳細滾動數據表"):
                    rolling_periods = {'3個月': 63, '6個月': 126, '1年': 252, '3年': 756, '5年': 1260, '10年': 2520}
                    r_rows = []
                    def get_r_stats(s, n):
                        r = {'標的': n}
                        for k, v in rolling_periods.items():
                            if len(s)>v: r[k] = (s.pct_change(v).dropna()>0).mean()
                            else: r[k] = np.nan
                        return r
                    r_rows.append(get_r_stats(margin_port_val, "🏆 混合投組"))
                    for t in tickers: r_rows.append(get_r_stats(df_close[t], t))
                    st.dataframe(pd.DataFrame(r_rows).style.format({k:'{:.0%}' for k in rolling_periods}).background_gradient(cmap='RdYlGn', vmin=0, vmax=1))

                # 未來預測 (喇叭圖)
                st.markdown("---")
                with st.expander("🔮 未來情境模擬：蒙地卡羅壓力測試 (Trumpet Chart)", expanded=True):
                    sim_years = years
                    num_sims_fut = 1000
                    mu_fut = avg_ret_hist
                    sigma_fut = vol_hist

                    st.info(f"模擬參數：年化報酬 **{mu_fut:.2%}**, 波動率 **{sigma_fut:.2%}**, 模擬 **{sim_years}** 年。")
                    dt = 1/252
                    days = int(sim_years * 252)
                    drift = (mu_fut - 0.5 * sigma_fut**2) * dt
                    diffusion = sigma_fut * np.sqrt(dt) * np.random.normal(0, 1, (days, num_sims_fut))
                    daily_log_ret = drift + diffusion
                    cum_log_ret = np.cumsum(daily_log_ret, axis=0)
                    price_paths = initial_investment * np.exp(cum_log_ret)
                    start_row = np.full((1, num_sims_fut), initial_investment)
                    price_paths = np.vstack([start_row, price_paths])
                    dates_fut = [datetime.today() + timedelta(days=x*(365/252)) for x in range(days + 1)]

                    p05 = np.percentile(price_paths, 5, axis=1)
                    p50 = np.percentile(price_paths, 50, axis=1)
                    p95 = np.percentile(price_paths, 95, axis=1)

                    fig_mc = go.Figure()
                    for i in range(min(30, num_sims_fut)):
                        fig_mc.add_trace(go.Scatter(x=dates_fut, y=price_paths[:, i], mode='lines', line=dict(color='lightgrey', width=0.5), opacity=0.3, showlegend=False, hoverinfo='skip'))

                    fig_mc.add_trace(go.Scatter(x=dates_fut, y=p05, mode='lines', name='悲觀 (5%)', line=dict(color='#d62728', width=1)))
                    fig_mc.add_trace(go.Scatter(x=dates_fut, y=p50, mode='lines', name='中性 (Base)', line=dict(color='#1f77b4', width=2), fill='tonexty', fillcolor='rgba(214, 39, 40, 0.1)'))
                    fig_mc.add_trace(go.Scatter(x=dates_fut, y=p95, mode='lines', name='樂觀 (95%)', line=dict(color='#2ca02c', width=1), fill='tonexty', fillcolor='rgba(44, 160, 44, 0.1)'))

                    fig_mc.update_layout(title='未來資產情境模擬', yaxis_title='資產價值 ($)', height=450, hovermode="x unified")
                    st.plotly_chart(fig_mc, use_container_width=True)

                    e95, e50, e05 = p95[-1], p50[-1], p05[-1]
                    c95 = (e95/initial_investment)**(1/sim_years)-1
                    c50 = (e50/initial_investment)**(1/sim_years)-1
                    c05 = (e05/initial_investment)**(1/sim_years)-1

                    st.markdown(f"""
                    **統計摘要 ({sim_years}年後)：**
                    * 🟢 **樂觀 (95%)**：${e95:,.0f} (年化 {c95:.2%})
                    * 🔵 **中性 (50%)**：${e50:,.0f} (年化 {c50:.2%})
                    * 🔴 **悲觀 (05%)**：${e05:,.0f} (年化 {c05:.2%})
                    """)

            except Exception as e:
                st.error(f"發生錯誤：{str(e)}")
