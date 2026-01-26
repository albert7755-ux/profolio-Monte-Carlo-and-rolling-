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
st.title('📈 智能投資組合優化器 (混合策略終極版)')
st.markdown("""
本系統結合 **「數學優化 (Solver)」** 與 **「蒙地卡羅隨機搜尋 (Monte Carlo Search)」**，讓您建構更穩健的混合投資組合。
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

# --- ★ 新增：策略混合器 (Strategy Blender) ---
st.sidebar.markdown("---")
st.sidebar.header("4. 策略混合權重 (Strategy Mix)")
st.sidebar.caption("調整兩種演算法在最終投組中的佔比")
mc_weight_ratio = st.sidebar.slider("蒙地卡羅 (MC) 佔比", 0.0, 1.0, 0.4, 0.1)
sharpe_weight_ratio = 1.0 - mc_weight_ratio
st.sidebar.text(f"配置：MC {mc_weight_ratio:.0%} + MaxSharpe {sharpe_weight_ratio:.0%}")

# --- 投資金額 ---
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
                num_assets = len(tickers)

                # ==========================
                # B1. 策略一：最大夏普 (Math Solver)
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
                
                # 計算 Solver 的指標
                ret_sharpe = np.sum(mean_returns * w_sharpe)
                vol_sharpe = np.sqrt(np.dot(w_sharpe.T, np.dot(cov_matrix, w_sharpe)))

                # ==========================
                # B2. 策略二：蒙地卡羅搜尋 (Monte Carlo Search)
                # ==========================
                num_sims = 3000
                all_weights = np.zeros((num_sims, num_assets))
                sim_results = np.zeros((3, num_sims)) # 0:Ret, 1:Vol, 2:Sharpe

                # 向量化生成隨機權重
                rand_w = np.random.random((num_sims, num_assets))
                rand_w = rand_w / rand_w.sum(axis=1)[:, None] # 歸一化
                all_weights = rand_w

                # 矩陣運算加速
                port_ret = np.dot(rand_w, mean_returns)
                # Vol需要迴圈或高階矩陣運算，這裡用簡單迴圈比較穩
                port_vol = np.zeros(num_sims)
                for i in range(num_sims):
                    port_vol[i] = np.sqrt(np.dot(rand_w[i].T, np.dot(cov_matrix, rand_w[i])))
                
                port_sharpe = (port_ret - risk_free_rate) / port_vol
                
                # 找出 MC 中夏普最高的
                best_mc_idx = port_sharpe.argmax()
                w_mc = all_weights[best_mc_idx]
                ret_mc = port_ret[best_mc_idx]
                vol_mc = port_vol[best_mc_idx]

                # ==========================
                # B3. 混合策略 (Blending)
                # ==========================
                w_final = (w_mc * mc_weight_ratio) + (w_sharpe * sharpe_weight_ratio)
                
                # 計算混合後的預期指標
                ret_final = np.sum(mean_returns * w_final)
                vol_final = np.sqrt(np.dot(w_final.T, np.dot(cov_matrix, w_final)))
                
                st.success(f"混合運算完成！(MC: {mc_weight_ratio:.0%} / Solver: {sharpe_weight_ratio:.0%})")

                # ==========================
                # C. 顯示：策略比較與效率前緣
                # ==========================
                col_c1, col_c2 = st.columns([1, 2])
                
                with col_c1:
                    st.subheader("📊 策略權重比較")
                    df_comp = pd.DataFrame({
                        '標的': tickers,
                        '🎲 MC最佳解': [f"{x:.1%}" for x in w_mc],
                        '🚀 最大夏普': [f"{x:.1%}" for x in w_sharpe],
                        '🏆 最終混合': [f"{x:.1%}" for x in w_final]
                    })
                    st.table(df_comp)
                    
                    st.markdown("#### 預期數據比較")
                    st.write(f"**🎲 MC策略**: 報酬 {ret_mc:.1%}, 波動 {vol_mc:.1%}")
                    st.write(f"**🚀 MaxSharpe**: 報酬 {ret_sharpe:.1%}, 波動 {vol_sharpe:.1%}")
                    st.info(f"**🏆 混合投組**: 報酬 {ret_final:.1%}, 波動 {vol_final:.1%}")

                with col_c2:
                    st.subheader("☁️ 效率前緣與策略落點 (Efficient Frontier)")
                    # 繪製散佈圖
                    fig_ef = go.Figure()
                    
                    # 3000 個隨機點
                    fig_ef.add_trace(go.Scatter(
                        x=port_vol, y=port_ret, mode='markers',
                        marker=dict(color=port_sharpe, colorscale='Viridis', size=5, showscale=True, colorbar=dict(title="Sharpe")),
                        name='隨機投組', text=[f"Sharpe: {s:.2f}" for s in port_sharpe], hoverinfo='text'
                    ))
                    
                    # 標記 MC 最佳點
                    fig_ef.add_trace(go.Scatter(
                        x=[vol_mc], y=[ret_mc], mode='markers+text',
                        marker=dict(color='orange', size=15, symbol='star'),
                        name='MC 最佳解', text=['MC Best'], textposition="top center"
                    ))
                    
                    # 標記 Solver 最佳點
                    fig_ef.add_trace(go.Scatter(
                        x=[vol_sharpe], y=[ret_sharpe], mode='markers+text',
                        marker=dict(color='red', size=15, symbol='diamond'),
                        name='最大夏普解', text=['Max Sharpe'], textposition="bottom center"
                    ))
                    
                    # 標記 混合 最佳點
                    fig_ef.add_trace(go.Scatter(
                        x=[vol_final], y=[ret_final], mode='markers+text',
                        marker=dict(color='blue', size=18, symbol='circle'),
                        name='最終混合投組', text=['Final Mix'], textposition="middle right"
                    ))
                    
                    fig_ef.update_layout(xaxis_title="年化波動度 (Risk)", yaxis_title="年化報酬率 (Return)", height=450)
                    st.plotly_chart(fig_ef, use_container_width=True)

                # ==========================
                # D. 回測與模擬 (使用 w_final)
                # ==========================
                
                # 計算混合投組的歷史淨值 (買入持有)
                raw_port_val = (normalized_prices * w_final).sum(axis=1)
                margin_port_val = calculate_margin_equity(raw_port_val, leverage, loan_ratio, margin_rate)
                margin_port_val.name = "🏆 混合策略投組"

                # 基礎回測圖表
                st.markdown("---")
                st.subheader("📈 資產成長回測 (基於混合權重)")
                fig_bt = px.line(margin_port_val, title='混合策略 vs Benchmark')
                fig_bt.update_traces(line=dict(color='blue', width=3))
                if normalized_bench is not None:
                    aligned_bench = normalized_bench.reindex(margin_port_val.index).ffill()
                    if aligned_bench.iloc[0] > 0: aligned_bench = aligned_bench / aligned_bench.iloc[0]
                    fig_bt.add_trace(go.Scatter(x=aligned_bench.index, y=aligned_bench, mode='lines', name=f'基準 ({bench_input})', line=dict(color='gray', width=2, dash='dash')))
                st.plotly_chart(fig_bt, use_container_width=True)

                # 績效指標
                def calculate_avg_annual_ret(series):
                    temp = series.copy()
                    if temp.index.tz is not None: temp.index = temp.index.tz_localize(None)
                    ann = temp.resample('Y').last().pct_change().dropna()
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

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("總報酬率", f"{total_ret:.2%}")
                c2.metric("平均年報酬 (歷史)", f"{avg_ret_hist:.2%}")
                c3.metric("年化波動 (歷史)", f"{vol_hist:.2%}")
                c4.metric("最大回撤", f"{mdd:.2%}")

                # 融資視覺化
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

                # 未來預測 (喇叭圖)
                st.markdown("---")
                with st.expander("🔮 未來情境模擬：蒙地卡羅壓力測試 (Trumpet Chart)", expanded=True):
                    sim_years = years
                    num_sims_fut = 1000
                    
                    # 使用「歷史回測出來的平均報酬與波動」來進行未來模擬
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
                    
                    # 95% / 5%
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
