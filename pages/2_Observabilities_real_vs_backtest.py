#################################
### Importando as bibliotecas ###
#################################

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import json
from datetime import datetime

#################################
###       Preâmbulo       ###
#################################

st.write("# Observabilities: Real vs Backtest")

st.write("#### Detailed analysis of differences between real trades and backtest simulations")

st.markdown('''
        In this analysis, we perform a trade-by-trade comparison between:
        - **Real account trades** (actual executions)
        - **Backtest simulations** (theoretical trades)
        
        Key metrics analyzed:
        1. **Trade Result Delta** - Profit difference between real and backtest
        2. **Entry Slippage** - Price difference at trade entry
        3. **Exit Slippage** - Price difference at trade exit
        4. **Timing Analysis** - Entry and exit timing differences
        5. **Hit Rate Analysis** - TP/SL hit accuracy
            
        -----
            
        ''')

#################################
###  0. Load Data  ###
#################################

# Carregar configuração da estratégia
with open('combined_strategy_teste.json', 'r') as f:
    strategy_params = json.load(f)

symbol = strategy_params['symbol']
timeframe = strategy_params['timeframe']
strategy_name = strategy_params['strategy']
magic_number = strategy_params['magic_number']
tp_points = strategy_params['hour_params']['9']['tp']  # Usando TP/SL da hora 9 como exemplo
sl_points = strategy_params['hour_params']['9']['sl']

# Mostrar informações da estratégia
st.write(f"### Strategy: {strategy_name}")
st.write(f"**Symbol:** {symbol} | **Timeframe:** {timeframe} | **Magic:** {magic_number}")
st.write(f"**Default TP:** {tp_points} points | **Default SL:** {sl_points} points")

# Carregar dados reais
df_real = pd.read_csv(f'bases/results_{symbol}_{timeframe}_{strategy_name}_magic_{magic_number}.csv', 
                      parse_dates=['time', 'time_ent', 'time_ext'])

# Carregar dados de backtest
df_backtest = pd.read_csv(f'bases/backtest_{symbol}_{timeframe}_{strategy_name}_magic_{magic_number}.csv', 
                          parse_dates=['time'])

# Preparar dados do backtest - filtrar apenas trades
df_backtest_trades = df_backtest[df_backtest['position'] != 0].copy()
df_backtest_trades['trade_time'] = df_backtest_trades.index

# Merge dos dados - matching por horário próximo
df_real['time_rounded'] = df_real['time'].dt.round('5min')
df_backtest_trades['time_rounded'] = df_backtest_trades['time'].dt.round('5min')

# Fazer o merge
df_merged = pd.merge(
    df_real,
    df_backtest_trades[['time_rounded', 'open', 'close', 'position', 'pts_final']],
    on='time_rounded',
    how='inner',
    suffixes=('_real', '_backtest')
)

# Calcular métricas
df_merged['slippage_entry'] = np.where(
    df_merged['posi'] == 'long',
    df_merged['price_ent'] - df_merged['close'],  # Long: compramos mais caro = slippage positivo
    df_merged['close'] - df_merged['price_ent']   # Short: vendemos mais barato = slippage positivo
)

df_merged['slippage_exit'] = np.where(
    df_merged['posi'] == 'long',
    df_merged['price_ext'] - (df_merged['price_ent'] + df_merged['pts_final_real']),
    (df_merged['price_ent'] - df_merged['pts_final_real']) - df_merged['price_ext']
)

df_merged['profit_delta'] = df_merged['profit'] - (df_merged['pts_final'] * 0.2)  # 0.2 = valor por ponto
df_merged['timing_delta'] = (df_merged['time_ext'] - df_merged['time_ent']).dt.total_seconds() / 60  # minutos

# Identificar hits de TP/SL
df_merged['hit_tp'] = df_merged['comment'].str.contains('tp', case=False, na=False)
df_merged['hit_sl'] = df_merged['comment'].str.contains('sl', case=False, na=False)

#################################
###  1. Overview Metrics  ###
#################################

st.write("## 1. Overview Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_trades = len(df_merged)
    st.metric("Matched Trades", total_trades)
    
with col2:
    avg_profit_delta = df_merged['profit_delta'].mean()
    st.metric("Avg Profit Delta", f"R$ {avg_profit_delta:.2f}")
    
with col3:
    avg_slippage_entry = df_merged['slippage_entry'].mean()
    st.metric("Avg Entry Slippage", f"{avg_slippage_entry:.1f} pts")
    
with col4:
    correlation = df_merged['profit'].corr(df_merged['pts_final'] * 0.2)
    st.metric("Real vs Backtest Correlation", f"{correlation:.3f}")

#################################
###  2. Trade Result Delta  ###
#################################

st.write("## 2. Trade Result Delta Analysis")

# Box plot por tipo de posição
fig_delta = px.box(
    df_merged, 
    y="profit_delta", 
    x="posi",
    color="posi",
    points="all",
    title="Profit Difference: Real - Backtest",
    color_discrete_map={'long': '#2ecc71', 'short': '#e74c3c'}
)

fig_delta.update_layout(
    yaxis_title="Delta (R$)",
    xaxis_title="Position Type",
    height=500,
    showlegend=False,
    template="plotly_white"
)

# Adicionar linha de referência em zero
fig_delta.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

st.plotly_chart(fig_delta, use_container_width=True)

# Estatísticas resumidas
col1, col2 = st.columns(2)
with col1:
    st.write("### Long Positions")
    long_stats = df_merged[df_merged['posi'] == 'long']['profit_delta'].describe()
    st.dataframe(long_stats.round(2))
    
with col2:
    st.write("### Short Positions")
    short_stats = df_merged[df_merged['posi'] == 'short']['profit_delta'].describe()
    st.dataframe(short_stats.round(2))

#################################
###  3. Slippage Analysis  ###
#################################

st.write("## 3. Slippage Analysis")

# Criar subplots para entry e exit slippage
col1, col2 = st.columns(2)

with col1:
    fig_slip_entry = px.histogram(
        df_merged,
        x="slippage_entry",
        color="posi",
        nbins=30,
        title="Entry Slippage Distribution",
        color_discrete_map={'long': '#2ecc71', 'short': '#e74c3c'}
    )
    fig_slip_entry.update_layout(
        xaxis_title="Slippage (points)",
        yaxis_title="Count",
        height=400,
        template="plotly_white"
    )
    st.plotly_chart(fig_slip_entry, use_container_width=True)

with col2:
    fig_slip_exit = px.histogram(
        df_merged,
        x="slippage_exit",
        color="posi",
        nbins=30,
        title="Exit Slippage Distribution",
        color_discrete_map={'long': '#2ecc71', 'short': '#e74c3c'}
    )
    fig_slip_exit.update_layout(
        xaxis_title="Slippage (points)",
        yaxis_title="Count",
        height=400,
        template="plotly_white"
    )
    st.plotly_chart(fig_slip_exit, use_container_width=True)

# Análise temporal do slippage
st.write("### Slippage by Hour of Day")

df_merged['entry_hour'] = pd.to_datetime(df_merged['time_ent']).dt.hour

hourly_slippage = df_merged.groupby('entry_hour').agg({
    'slippage_entry': ['mean', 'std', 'count'],
    'slippage_exit': ['mean', 'std']
}).round(2)

fig_hourly_slip = go.Figure()

# Entry slippage
fig_hourly_slip.add_trace(go.Scatter(
    x=hourly_slippage.index,
    y=hourly_slippage[('slippage_entry', 'mean')],
    mode='lines+markers',
    name='Entry Slippage',
    line=dict(color='blue', width=2),
    error_y=dict(
        type='data',
        array=hourly_slippage[('slippage_entry', 'std')],
        visible=True
    )
))

# Exit slippage
fig_hourly_slip.add_trace(go.Scatter(
    x=hourly_slippage.index,
    y=hourly_slippage[('slippage_exit', 'mean')],
    mode='lines+markers',
    name='Exit Slippage',
    line=dict(color='red', width=2),
    error_y=dict(
        type='data',
        array=hourly_slippage[('slippage_exit', 'std')],
        visible=True
    )
))

fig_hourly_slip.update_layout(
    title="Average Slippage by Hour",
    xaxis_title="Hour",
    yaxis_title="Slippage (points)",
    height=400,
    template="plotly_white",
    hovermode='x unified'
)

st.plotly_chart(fig_hourly_slip, use_container_width=True)

#################################
###  4. Timing Analysis  ###
#################################

st.write("## 4. Trade Timing Analysis")

# Distribuição do tempo de duração dos trades
fig_timing = px.histogram(
    df_merged,
    x="timing_delta",
    nbins=50,
    title="Trade Duration Distribution",
    labels={'timing_delta': 'Duration (minutes)', 'count': 'Number of Trades'}
)

fig_timing.update_layout(
    height=400,
    template="plotly_white"
)

st.plotly_chart(fig_timing, use_container_width=True)

# Análise de TP/SL hits
col1, col2, col3 = st.columns(3)

with col1:
    tp_hits = df_merged['hit_tp'].sum()
    tp_rate = (tp_hits / len(df_merged) * 100) if len(df_merged) > 0 else 0
    st.metric("TP Hits", f"{tp_hits} ({tp_rate:.1f}%)")

with col2:
    sl_hits = df_merged['hit_sl'].sum()
    sl_rate = (sl_hits / len(df_merged) * 100) if len(df_merged) > 0 else 0
    st.metric("SL Hits", f"{sl_hits} ({sl_rate:.1f}%)")

with col3:
    other_exits = len(df_merged) - tp_hits - sl_hits
    other_rate = (other_exits / len(df_merged) * 100) if len(df_merged) > 0 else 0
    st.metric("Other Exits", f"{other_exits} ({other_rate:.1f}%)")

#################################
###  5. Detailed Comparison  ###
#################################

st.write("## 5. Trade-by-Trade Comparison")

# Scatter plot: Real vs Backtest profits
fig_scatter = px.scatter(
    df_merged,
    x=df_merged['pts_final'] * 0.2,  # Backtest profit in R$
    y='profit',  # Real profit
    color='posi',
    title="Real vs Backtest Profits",
    labels={'x': 'Backtest Profit (R$)', 'profit': 'Real Profit (R$)'},
    color_discrete_map={'long': '#2ecc71', 'short': '#e74c3c'}
)

# Adicionar linha de referência diagonal
x_range = [df_merged['pts_final'].min() * 0.2, df_merged['pts_final'].max() * 0.2]
fig_scatter.add_trace(go.Scatter(
    x=x_range,
    y=x_range,
    mode='lines',
    name='Perfect Match',
    line=dict(color='gray', dash='dash')
))

fig_scatter.update_layout(
    height=500,
    template="plotly_white"
)

st.plotly_chart(fig_scatter, use_container_width=True)

#################################
###  6. Summary Table  ###
#################################

st.write("## 6. Recent Trades Summary")

# Preparar tabela resumida
summary_df = df_merged[['time', 'posi', 'price_ent', 'price_ext', 'profit', 
                       'pts_final_real', 'slippage_entry', 'slippage_exit', 
                       'profit_delta', 'comment']].copy()

summary_df = summary_df.sort_values('time', ascending=False).head(20)

# Formatar colunas
summary_df['profit'] = summary_df['profit'].apply(lambda x: f"R$ {x:.2f}")
summary_df['profit_delta'] = summary_df['profit_delta'].apply(lambda x: f"R$ {x:.2f}")
summary_df['slippage_entry'] = summary_df['slippage_entry'].apply(lambda x: f"{x:.1f}")
summary_df['slippage_exit'] = summary_df['slippage_exit'].apply(lambda x: f"{x:.1f}")

st.dataframe(
    summary_df,
    hide_index=True,
    use_container_width=True
)

#################################
###  7. Key Insights  ###
#################################

st.write("## 7. Key Insights")

# Calcular métricas resumidas
avg_entry_slip = df_merged['slippage_entry'].mean()
avg_exit_slip = df_merged['slippage_exit'].mean()
total_slip_cost = (avg_entry_slip + abs(avg_exit_slip)) * 0.2 * len(df_merged)

insights = f"""
### Performance Summary:

1. **Slippage Impact**: 
   - Average entry slippage: {avg_entry_slip:.1f} points
   - Average exit slippage: {avg_exit_slip:.1f} points
   - Estimated total slippage cost: R$ {total_slip_cost:.2f}

2. **Trade Accuracy**:
   - {len(df_merged)} trades matched between real and backtest
   - Average profit difference: R$ {df_merged['profit_delta'].mean():.2f}
   - Correlation between real and backtest: {correlation:.3f}

3. **Exit Analysis**:
   - TP hit rate: {tp_rate:.1f}%
   - SL hit rate: {sl_rate:.1f}%
   - Strategy exits well-calibrated: {'Yes' if abs(df_merged['profit_delta'].mean()) < 5 else 'Needs adjustment'}

4. **Recommendations**:
   - {'Slippage is within acceptable range' if abs(avg_entry_slip) < 10 else 'Consider adjusting entry timing'}
   - {'Exit strategy performing as expected' if tp_rate > 30 else 'Review TP/SL levels'}
"""

st.markdown(insights)