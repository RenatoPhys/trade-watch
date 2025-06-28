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
import glob
import os

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
###  0. Strategy Selection  ###
#################################

# Encontrar todos os arquivos JSON de estratégia
strategy_files = glob.glob('combined_strategy_*.json')
strategy_names = {}

# Carregar informações básicas de cada estratégia
for file in strategy_files:
    with open(file, 'r') as f:
        data = json.load(f)
        # Criar nome descritivo para a estratégia
        name = f"{data['strategy']} - {data['symbol']} ({data['timeframe']}) - Magic: {data['magic_number']}"
        strategy_names[name] = file

# Seleção de modo
mode = st.radio(
    "Select Analysis Mode:",
    ["Single Strategy", "Combined Strategies"],
    horizontal=True
)

selected_strategies = []

if mode == "Single Strategy":
    # Dropdown para seleção única
    selected_name = st.selectbox(
        "Select Strategy:",
        options=list(strategy_names.keys())
    )
    if selected_name:
        selected_strategies = [strategy_names[selected_name]]
else:
    # Multiselect para combinação
    st.write("**Select strategies to combine:**")
    
    # Criar checkboxes para cada estratégia
    cols = st.columns(2)
    for idx, (name, file) in enumerate(strategy_names.items()):
        col_idx = idx % 2
        with cols[col_idx]:
            if st.checkbox(name, key=f"strategy_{idx}"):
                selected_strategies.append(file)
    
    if not selected_strategies:
        st.warning("Please select at least one strategy to analyze.")
        st.stop()

#################################
### Load and Combine Data ###
#################################

# Função para carregar dados de uma estratégia
@st.cache_data
def load_strategy_data(strategy_file):
    with open(strategy_file, 'r') as f:
        params = json.load(f)
    
    # Extrair informações
    symbol = params['symbol']
    timeframe = params['timeframe']
    strategy_name = params['strategy']
    magic_number = params['magic_number']
    
    # Construir nomes dos arquivos
    real_file = f'bases/results_{symbol}_{timeframe}_{strategy_name}_magic_{magic_number}.csv'
    backtest_file = f'bases/backtest_{symbol}_{timeframe}_{strategy_name}_magic_{magic_number}.csv'
    
    # Verificar se os arquivos existem
    if not os.path.exists(real_file):
        st.warning(f"Real trades file not found: {real_file}")
        df_real = pd.DataFrame()
    else:
        # Carregar dados reais
        df_real = pd.read_csv(real_file, parse_dates=['time', 'time_ent', 'time_ext'])
        df_real['time'] = pd.to_datetime(df_real['time'])
        df_real.set_index('time', inplace=True)
    
    if not os.path.exists(backtest_file):
        st.warning(f"Backtest file not found: {backtest_file}")
        df_backtest = pd.DataFrame()
    else:
        # Carregar dados de backtest
        df_backtest = pd.read_csv(backtest_file, index_col=['time'], parse_dates=['time'])
    
    return df_real, df_backtest, params

# Carregar dados de todas as estratégias selecionadas
all_real_data = []
all_backtest_data = []
all_params = []

for strategy_file in selected_strategies:
    df_real, df_backtest, params = load_strategy_data(strategy_file)
    if not df_real.empty or not df_backtest.empty:
        all_real_data.append(df_real)
        all_backtest_data.append(df_backtest)
        all_params.append(params)

if not all_real_data and not all_backtest_data:
    st.error("No valid data found for selected strategies.")
    st.stop()

# Mostrar informações das estratégias selecionadas
if len(all_params) == 1:
    st.write(f"### Strategy: {all_params[0]['strategy']} - {all_params[0]['symbol']}")
    st.write(f"**Symbol:** {all_params[0]['symbol']} | **Timeframe:** {all_params[0]['timeframe']} | **Magic Number:** {all_params[0]['magic_number']}")
else:
    st.write(f"### Combined Analysis: {len(all_params)} Strategies")
    with st.expander("Strategy Details"):
        for params in all_params:
            st.write(f"- **{params['strategy']}** on {params['symbol']} ({params['timeframe']}) - Magic: {params['magic_number']}")

# Combinar dados se múltiplas estratégias
if len(all_real_data) == 1:
    # Estratégia única
    df_real = all_real_data[0]
    df_backtest = all_backtest_data[0]
else:
    # Combinar múltiplas estratégias
    # Para dados reais
    combined_real = pd.DataFrame()
    
    for i, df in enumerate(all_real_data):
        if not df.empty:
            # Criar uma cópia para não modificar o original
            df_copy = df.copy()
            df_copy['strategy_id'] = i
            combined_real = pd.concat([combined_real, df_copy])
    
    # Ordenar por tempo
    if not combined_real.empty:
        combined_real = combined_real.sort_index()
        
        # Recalcular cstrategy para múltiplas estratégias
        combined_real['cstrategy'] = combined_real['profit'].cumsum()
        df_real = combined_real
    else:
        df_real = pd.DataFrame()
    
    # Para dados de backtest - somar equity ajustada
    if all_backtest_data and not all_backtest_data[0].empty:
        # Alinhar todos os DataFrames pelo índice
        combined_equity = pd.DataFrame()
        
        for i, df in enumerate(all_backtest_data):
            if not df.empty:
                # Equity ajustada (removendo capital inicial)
                equity_adj = df['equity'] - 30000
                combined_equity[f'strategy_{i}'] = equity_adj
        
        # Somar equities e adicionar capital inicial uma vez
        df_backtest = pd.DataFrame(index=combined_equity.index)
        df_backtest['equity'] = combined_equity.sum(axis=1) + 30000
    else:
        df_backtest = pd.DataFrame()

#################################
###  1. Performance Acumulada  ###
#################################

st.write("## 1. Cumulative Return")

# Criar figura para retorno acumulado
fig_return = go.Figure()

# Adicionar linha do backtest
if not df_backtest.empty:
    fig_return.add_trace(go.Scatter(
        x=df_backtest.index,
        y=df_backtest["equity"] - 30000,  # Subtrair capital inicial
        name='Backtest',
        line=dict(color="#d62728", width=2)
    ))

# Se múltiplas estratégias e modo Combined, mostrar contribuição individual
if len(all_real_data) > 1 and not df_real.empty:
    # Adicionar linhas individuais para real
    for i, (df, params) in enumerate(zip(all_real_data, all_params)):
        if not df.empty:
            fig_return.add_trace(go.Scatter(
                x=df.index,
                y=df["cstrategy"],
                name=f"{params['strategy']} - {params['symbol']} (Real)",
                line=dict(width=1),
                opacity=0.7
            ))

# Adicionar linha do real (total ou única)
if not df_real.empty:
    fig_return.add_trace(go.Scatter(
        x=df_real.index,
        y=df_real["cstrategy"],
        name='Real Total' if len(all_real_data) > 1 else 'Real',
        line=dict(color="#1f77b4", width=3)
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
if not df_real.empty:
    total_return_real = df_real['cstrategy'].iloc[-1] if len(df_real) > 0 else 0
    total_trades_real = len(df_real)
    winning_trades_real = len(df_real[df_real['profit'] > 0])
    win_rate_real = (winning_trades_real / total_trades_real * 100) if total_trades_real > 0 else 0
else:
    total_return_real = 0
    total_trades_real = 0
    winning_trades_real = 0
    win_rate_real = 0

# Calcular métricas do backtest
if not df_backtest.empty:
    total_return_backtest = df_backtest['equity'].iloc[-1] - 30000 if len(df_backtest) > 0 else 0
    # Para múltiplas estratégias, somar trades de todos os backtests
    if len(all_backtest_data) > 1:
        total_trades_backtest = sum(len(df[df['position'] != 0]) if 'position' in df.columns else 0 
                                   for df in all_backtest_data if not df.empty)
    else:
        total_trades_backtest = len(df_backtest[df_backtest['position'] != 0]) if 'position' in df_backtest.columns else 0
else:
    total_return_backtest = 0
    total_trades_backtest = 0

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

if not df_real.empty:
    # Calcular drawdown do real
    df_real['cummax'] = df_real['cstrategy'].cummax()
    df_real['drawdown'] = df_real['cstrategy'] - df_real['cummax']

if not df_backtest.empty:
    # Calcular drawdown do backtest
    df_backtest['equity_adjusted'] = df_backtest['equity'] - 30000
    df_backtest['cummax'] = df_backtest['equity_adjusted'].cummax()
    df_backtest['drawdown'] = df_backtest['equity_adjusted'] - df_backtest['cummax']

# Criar figura do drawdown
fig_dd = go.Figure()

# Adicionar drawdown do real
if not df_real.empty:
    fig_dd.add_trace(go.Scatter(
        x=df_real.index,
        y=df_real['drawdown'],
        name='Real',
        fill='tozeroy',
        line=dict(color="#1f77b4")
    ))

# Adicionar drawdown do backtest
if not df_backtest.empty:
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
    max_dd_real = df_real['drawdown'].min() if not df_real.empty else 0
    st.metric("Max Drawdown (Real)", f"R$ {max_dd_real:,.2f}")
with col2:
    max_dd_backtest = df_backtest['drawdown'].min() if not df_backtest.empty else 0
    st.metric("Max Drawdown (Backtest)", f"R$ {max_dd_backtest:,.2f}")

#################################
###  4. Performance Diária  ###
#################################

st.write("## 4. Daily Performance")

if not df_real.empty:
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
else:
    st.info("No real trading data available for daily performance analysis.")

#################################
###  5. Distribuição dos Trades  ###
#################################

st.write("## 5. Trade Distribution")

if not df_real.empty:
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
else:
    st.info("No real trading data available for trade distribution analysis.")

#################################
###  6. Análise por Hora  ###
#################################

st.write("## 6. Performance by Hour")

if not df_real.empty and 'time_ent' in df_real.columns:
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
else:
    st.info("No hourly data available for analysis.")

#################################
### 7. Strategy Comparison  ###
#################################

if len(all_real_data) > 1 and mode == "Combined Strategies":
    st.write("## 7. Individual Strategy Performance")
    
    # Calcular métricas individuais
    strategy_metrics = []
    
    for i, (df, params) in enumerate(zip(all_real_data, all_params)):
        if not df.empty:
            total_ret = df['cstrategy'].iloc[-1]
            trades = len(df)
            wins = len(df[df['profit'] > 0])
            wr = (wins / trades * 100) if trades > 0 else 0
            avg_trade = total_ret / trades if trades > 0 else 0
            
            strategy_metrics.append({
                'Strategy': f"{params['strategy']} - {params['symbol']}",
                'Total Return': total_ret,
                'Trades': trades,
                'Win Rate': wr,
                'Avg Trade': avg_trade,
                'Best Trade': df['profit'].max(),
                'Worst Trade': df['profit'].min()
            })
    
    metrics_df = pd.DataFrame(strategy_metrics)
    st.dataframe(metrics_df.style.format({
        'Total Return': 'R$ {:,.2f}',
        'Win Rate': '{:.1f}%',
        'Avg Trade': 'R$ {:,.2f}',
        'Best Trade': 'R$ {:,.2f}',
        'Worst Trade': 'R$ {:,.2f}'
    }))