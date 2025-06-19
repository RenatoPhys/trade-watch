#################################
### Importando as bibliotecas ###
#################################

import numpy as np
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import json

#################################
###       Preâmbulo       ###
#################################

st.write("# Performance Monitoring")

st.write("#### On this page we will put all the performance metrics of the strategies into action")

st.markdown('''
        For an informative and complete performance analysis of algo trading strategies
        developed by me, I provide the following metrics and views:
        1. Cumulative Return (in R$) - real account and backtest
        2. Risk/return ratios: a) [Sharpe](https://pt.wikipedia.org/wiki/%C3%8Ddice_de_Sharpe),
            b) [Sortino](https://en.wikipedia.org/wiki/Sortino_ratio),
            c) [Calm](https://www.investopedia.com/terms/c/calmarratio.asp)
        3. Strategy Drawdown
        4. Daily/Weekly Performance
        5. Percentage of winning/losing trades
        
        Note: Whenever we are talking about profit, we mean gross profit before taxes.
        For example, on day trades (as is the case with my strategies), we must pay 20% IR. Read more at:
        https://blog.nubank.com.br/darf-day-trade-2023/ 
            
        -----
            
        ''')

#################################
### Carregar parâmetros da estratégia ###
#################################

# Carregar o arquivo JSON com os parâmetros da estratégia
with open('combined_strategy_teste.json', 'r') as f:
    strategy_params = json.load(f)

# Extrair informações importantes
symbol = strategy_params['symbol']
timeframe = strategy_params['timeframe']
strategy_name = strategy_params['strategy']
magic_number = strategy_params['magic_number']

# Mostrar informações da estratégia
st.write(f"### Strategy: {strategy_name}")
st.write(f"**Symbol:** {symbol} | **Timeframe:** {timeframe} | **Magic Number:** {magic_number}")

#################################
### Importar dados ###
#################################

# Importar dados de trades reais
df_real = pd.read_csv(f'bases/results_{symbol}_{timeframe}_{strategy_name}_{magic_number}.csv', 
                     parse_dates=['time', 'time_ent', 'time_ext'])

# Importar dados de backtest
df_backtest = pd.read_csv(f'bases/backtest_{symbol}_{timeframe}_{strategy_name}_{magic_number}.csv', 
                         index_col=['time'], parse_dates=['time'])

# Processar dados do real
df_real['time'] = pd.to_datetime(df_real['time'])
df_real.set_index('time', inplace=True)

#################################
###  1. Performance Acumulada  ###
#################################

st.write("## 1. Cumulative Return")

# Criar figura para retorno acumulado
fig_return = go.Figure()

# Adicionar linha do backtest
fig_return.add_trace(go.Scatter(
    x=df_backtest.index,
    y=df_backtest["equity"] - 30000,  # Subtrair capital inicial
    name='Backtest',
    line=dict(color="#d62728", width=2)
))

# Adicionar linha do real
fig_return.add_trace(go.Scatter(
    x=df_real.index,
    y=df_real["cstrategy"],
    name='Real',
    line=dict(color="#1f77b4", width=2)
))

# Configurar layout
fig_return.update_layout(
    title=dict(text="Cumulative Return", font=dict(size=24)),
    xaxis_title=dict(text="<b>Date</b>", font=dict(size=16)),
    yaxis_title=dict(text="<b>Return (R$)</b>", font=dict(size=16)),
    hovermode='x unified',
    showlegend=True,
    height=500,
    template="plotly_white"
)

st.plotly_chart(fig_return, use_container_width=True)

#################################
###  2. Métricas de Performance  ###
#################################

st.write("## 2. Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

# Calcular métricas do real
total_return_real = df_real['cstrategy'].iloc[-1] if len(df_real) > 0 else 0
total_trades_real = len(df_real)
winning_trades_real = len(df_real[df_real['profit'] > 0])
win_rate_real = (winning_trades_real / total_trades_real * 100) if total_trades_real > 0 else 0

# Calcular métricas do backtest
total_return_backtest = df_backtest['equity'].iloc[-1] - 30000 if len(df_backtest) > 0 else 0
total_trades_backtest = len(df_backtest[df_backtest['position'] != 0])

with col1:
    st.metric("Total Return (Real)", f"R$ {total_return_real:,.2f}")
    st.metric("Total Return (Backtest)", f"R$ {total_return_backtest:,.2f}")

with col2:
    st.metric("Total Trades (Real)", total_trades_real)
    st.metric("Total Trades (Backtest)", total_trades_backtest)

with col3:
    st.metric("Win Rate (Real)", f"{win_rate_real:.1f}%")

with col4:
    st.metric("Avg Trade (Real)", f"R$ {total_return_real/total_trades_real:.2f}" if total_trades_real > 0 else "N/A")

#################################
###  3. Drawdown  ###
#################################

st.write("## 3. Drawdown")

# Calcular drawdown do real
df_real['cummax'] = df_real['cstrategy'].cummax()
df_real['drawdown'] = df_real['cstrategy'] - df_real['cummax']

# Calcular drawdown do backtest
df_backtest['equity_adjusted'] = df_backtest['equity'] - 30000
df_backtest['cummax'] = df_backtest['equity_adjusted'].cummax()
df_backtest['drawdown'] = df_backtest['equity_adjusted'] - df_backtest['cummax']

# Criar figura do drawdown
fig_dd = go.Figure()

# Adicionar drawdown do real
fig_dd.add_trace(go.Scatter(
    x=df_real.index,
    y=df_real['drawdown'],
    name='Real',
    fill='tozeroy',
    line=dict(color="#1f77b4")
))

# Adicionar drawdown do backtest
fig_dd.add_trace(go.Scatter(
    x=df_backtest.index,
    y=df_backtest['drawdown'],
    name='Backtest',
    fill='tozeroy',
    line=dict(color="#d62728", dash='dash')
))

# Configurar layout
fig_dd.update_layout(
    title=dict(text="Drawdown", font=dict(size=24)),
    xaxis_title=dict(text="<b>Date</b>", font=dict(size=16)),
    yaxis_title=dict(text="<b>Drawdown (R$)</b>", font=dict(size=16)),
    hovermode='x unified',
    showlegend=True,
    height=400,
    template="plotly_white"
)

st.plotly_chart(fig_dd, use_container_width=True)

# Mostrar métricas do drawdown
col1, col2 = st.columns(2)
with col1:
    max_dd_real = df_real['drawdown'].min()
    st.metric("Max Drawdown (Real)", f"R$ {max_dd_real:,.2f}")
with col2:
    max_dd_backtest = df_backtest['drawdown'].min()
    st.metric("Max Drawdown (Backtest)", f"R$ {max_dd_backtest:,.2f}")

#################################
###  4. Performance Diária  ###
#################################

st.write("## 4. Daily Performance")

# Agrupar trades por dia
daily_real = df_real.groupby(df_real.index.date)['profit'].sum()
daily_real.index = pd.to_datetime(daily_real.index)

# Criar figura para performance diária
fig_daily = go.Figure()

# Adicionar barras
colors = ['green' if x > 0 else 'red' for x in daily_real.values]
fig_daily.add_trace(go.Bar(
    x=daily_real.index,
    y=daily_real.values,
    name='Daily P&L',
    marker_color=colors
))

# Configurar layout
fig_daily.update_layout(
    title=dict(text="Daily Performance", font=dict(size=24)),
    xaxis_title=dict(text="<b>Date</b>", font=dict(size=16)),
    yaxis_title=dict(text="<b>Profit/Loss (R$)</b>", font=dict(size=16)),
    showlegend=False,
    height=400,
    template="plotly_white"
)

st.plotly_chart(fig_daily, use_container_width=True)

#################################
###  5. Distribuição dos Trades  ###
#################################

st.write("## 5. Trade Distribution")

col1, col2 = st.columns(2)

with col1:
    # Histograma dos lucros
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df_real['profit'],
        nbinsx=30,
        name='Trade P&L',
        marker_color='lightblue',
        marker_line_color='darkblue',
        marker_line_width=1
    ))
    
    fig_hist.update_layout(
        title="Trade P&L Distribution",
        xaxis_title="Profit/Loss (R$)",
        yaxis_title="Frequency",
        height=350,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    # Gráfico de pizza para win/loss
    wins = len(df_real[df_real['profit'] > 0])
    losses = len(df_real[df_real['profit'] <= 0])
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Wins', 'Losses'],
        values=[wins, losses],
        hole=.3,
        marker_colors=['#2ecc71', '#e74c3c']
    )])
    
    fig_pie.update_layout(
        title="Win/Loss Ratio",
        height=350,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

#################################
###  6. Análise por Hora  ###
#################################

st.write("## 6. Performance by Hour")

# Criar coluna de hora
df_real['hour'] = pd.to_datetime(df_real['time_ent']).dt.hour

# Agrupar por hora
hourly_stats = df_real.groupby('hour').agg({
    'profit': ['sum', 'count', 'mean']
}).round(2)

# Criar figura
fig_hourly = go.Figure()

# Adicionar barras para lucro total por hora
fig_hourly.add_trace(go.Bar(
    x=hourly_stats.index,
    y=hourly_stats[('profit', 'sum')],
    name='Total Profit',
    marker_color='lightblue',
    yaxis='y'
))

# Adicionar linha para número de trades
fig_hourly.add_trace(go.Scatter(
    x=hourly_stats.index,
    y=hourly_stats[('profit', 'count')],
    name='Number of Trades',
    line=dict(color='red', width=2),
    yaxis='y2'
))

# Configurar layout com dois eixos Y
fig_hourly.update_layout(
    title=dict(text="Performance by Hour", font=dict(size=24)),
    xaxis_title=dict(text="<b>Hour</b>", font=dict(size=16)),
    yaxis=dict(title="Total Profit (R$)", side='left'),
    yaxis2=dict(title="Number of Trades", overlaying='y', side='right'),
    hovermode='x unified',
    height=400,
    template="plotly_white"
)

st.plotly_chart(fig_hourly, use_container_width=True)

# Tabela resumo
st.write("### Summary Statistics by Hour")
hourly_display = hourly_stats.copy()
hourly_display.columns = ['Total Profit (R$)', 'Number of Trades', 'Avg Profit (R$)']
st.dataframe(hourly_display.style.format({
    'Total Profit (R$)': 'R$ {:,.2f}',
    'Avg Profit (R$)': 'R$ {:,.2f}'
}))