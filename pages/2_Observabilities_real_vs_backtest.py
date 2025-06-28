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
import glob
import os

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
    st.write("**Select strategies to compare:**")
    
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
### Load and Process Data ###
#################################

# Função para carregar e processar dados de uma estratégia
@st.cache_data
def load_and_process_strategy_data(strategy_file):
    with open(strategy_file, 'r') as f:
        params = json.load(f)
    
    symbol = params['symbol']
    timeframe = params['timeframe']
    strategy_name = params['strategy']
    magic_number = params['magic_number']
    
    # Pegar TP/SL da hora 9 como exemplo
    tp_points = params['hour_params']['9']['tp']
    sl_points = params['hour_params']['9']['sl']
    
    # Construir nomes dos arquivos
    real_file = f'bases/results_{symbol}_{timeframe}_{strategy_name}_magic_{magic_number}.csv'
    backtest_file = f'bases/backtest_{symbol}_{timeframe}_{strategy_name}_magic_{magic_number}.csv'
    
    # Verificar se os arquivos existem
    if not os.path.exists(real_file) or not os.path.exists(backtest_file):
        st.warning(f"Files not found for {strategy_name} - {symbol}")
        return None, params
    
    # Carregar dados reais
    df_real = pd.read_csv(real_file, parse_dates=['time', 'time_ent', 'time_ext'])
    
    # Carregar dados de backtest
    df_backtest = pd.read_csv(backtest_file, parse_dates=['time'])
    
    # Preparar dados do backtest - filtrar apenas trades
    df_backtest_trades = df_backtest[df_backtest['position'] != 0].copy()
    
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
    
    # Adicionar identificação da estratégia
    df_merged['strategy_name'] = f"{strategy_name} - {symbol}"
    df_merged['strategy_id'] = f"{strategy_name}_{symbol}_{magic_number}"
    
    # Calcular métricas
    df_merged['slippage_entry'] = np.where(
        df_merged['posi'] == 'long',
        df_merged['price_ent'] - df_merged['close'],
        df_merged['close'] - df_merged['price_ent']
    )
    
    df_merged['slippage_exit'] = np.where(
        df_merged['posi'] == 'long',
        df_merged['price_ext'] - (df_merged['price_ent'] + df_merged['pts_final_real']),
        (df_merged['price_ent'] - df_merged['pts_final_real']) - df_merged['price_ext']
    )
    
    df_merged['profit_delta'] = df_merged['profit'] - (df_merged['pts_final'] * 0.2)
    df_merged['timing_delta'] = (df_merged['time_ext'] - df_merged['time_ent']).dt.total_seconds() / 60
    
    # Identificar hits de TP/SL
    df_merged['hit_tp'] = df_merged['comment'].str.contains('tp', case=False, na=False)
    df_merged['hit_sl'] = df_merged['comment'].str.contains('sl', case=False, na=False)
    
    # Adicionar informações extras
    df_merged['tp_points'] = tp_points
    df_merged['sl_points'] = sl_points
    
    return df_merged, params

# Carregar dados de todas as estratégias selecionadas
all_merged_data = []
all_params = []

for strategy_file in selected_strategies:
    df_merged, params = load_and_process_strategy_data(strategy_file)
    if df_merged is not None:
        all_merged_data.append(df_merged)
        all_params.append(params)

if not all_merged_data:
    st.error("No valid data found for selected strategies.")
    st.stop()

# Combinar todos os dados
df_combined = pd.concat(all_merged_data, ignore_index=True)

# Mostrar informações das estratégias selecionadas
if len(all_params) == 1:
    st.write(f"### Strategy: {all_params[0]['strategy']} - {all_params[0]['symbol']}")
    st.write(f"**Timeframe:** {all_params[0]['timeframe']} | **Magic:** {all_params[0]['magic_number']}")
else:
    st.write(f"### Analyzing {len(all_params)} Strategies")
    with st.expander("Strategy Details"):
        for params in all_params:
            st.write(f"- **{params['strategy']}** on {params['symbol']} ({params['timeframe']}) - Magic: {params['magic_number']}")

#################################
###  1. Overview Metrics  ###
#################################

st.write("## 1. Overview Metrics")

if mode == "Single Strategy":
    # Métricas simples para estratégia única
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trades = len(df_combined)
        st.metric("Matched Trades", total_trades)
        
    with col2:
        avg_profit_delta = df_combined['profit_delta'].mean()
        st.metric("Avg Profit Delta", f"R$ {avg_profit_delta:.2f}")
        
    with col3:
        avg_slippage_entry = df_combined['slippage_entry'].mean()
        st.metric("Avg Entry Slippage", f"{avg_slippage_entry:.1f} pts")
        
    with col4:
        correlation = df_combined['profit'].corr(df_combined['pts_final'] * 0.2)
        st.metric("Real vs Backtest Correlation", f"{correlation:.3f}")
else:
    # Métricas por estratégia
    strategy_metrics = []
    
    for strategy_id in df_combined['strategy_id'].unique():
        df_strategy = df_combined[df_combined['strategy_id'] == strategy_id]
        
        metrics = {
            'Strategy': df_strategy['strategy_name'].iloc[0],
            'Matched Trades': len(df_strategy),
            'Avg Profit Delta': df_strategy['profit_delta'].mean(),
            'Avg Entry Slippage': df_strategy['slippage_entry'].mean(),
            'Correlation': df_strategy['profit'].corr(df_strategy['pts_final'] * 0.2)
        }
        strategy_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(strategy_metrics)
    st.dataframe(metrics_df.style.format({
        'Avg Profit Delta': 'R$ {:.2f}',
        'Avg Entry Slippage': '{:.1f} pts',
        'Correlation': '{:.3f}'
    }))

#################################
###  2. Trade Result Delta  ###
#################################

st.write("## 2. Trade Result Delta Analysis")

if mode == "Single Strategy":
    # Box plot por tipo de posição
    fig_delta = px.box(
        df_combined, 
        y="profit_delta", 
        x="posi",
        color="posi",
        points="all",
        title="Profit Difference: Real - Backtest",
        color_discrete_map={'long': '#2ecc71', 'short': '#e74c3c'}
    )
else:
    # Box plot por estratégia
    fig_delta = px.box(
        df_combined, 
        y="profit_delta", 
        x="strategy_name",
        color="posi",
        title="Profit Difference by Strategy",
        color_discrete_map={'long': '#2ecc71', 'short': '#e74c3c'}
    )

fig_delta.update_layout(
    yaxis_title="Delta (R$)",
    xaxis_title="Strategy" if mode == "Combined Strategies" else "Position Type",
    height=500,
    template="plotly_white"
)

# Adicionar linha de referência em zero
fig_delta.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

st.plotly_chart(fig_delta, use_container_width=True)

# Estatísticas resumidas
if mode == "Single Strategy":
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Long Positions")
        long_stats = df_combined[df_combined['posi'] == 'long']['profit_delta'].describe()
        st.dataframe(long_stats.round(2))
        
    with col2:
        st.write("### Short Positions")
        short_stats = df_combined[df_combined['posi'] == 'short']['profit_delta'].describe()
        st.dataframe(short_stats.round(2))
else:
    # Tabela resumida por estratégia
    summary_by_strategy = df_combined.groupby(['strategy_name', 'posi'])['profit_delta'].agg(['mean', 'std', 'count']).round(2)
    st.write("### Profit Delta by Strategy and Position")
    st.dataframe(summary_by_strategy)

#################################
###  3. Slippage Analysis  ###
#################################

st.write("## 3. Slippage Analysis")

# Criar subplots para entry e exit slippage
col1, col2 = st.columns(2)

with col1:
    if mode == "Single Strategy":
        fig_slip_entry = px.histogram(
            df_combined,
            x="slippage_entry",
            color="posi",
            nbins=30,
            title="Entry Slippage Distribution",
            color_discrete_map={'long': '#2ecc71', 'short': '#e74c3c'}
        )
    else:
        fig_slip_entry = px.box(
            df_combined,
            y="slippage_entry",
            x="strategy_name",
            color="posi",
            title="Entry Slippage by Strategy",
            color_discrete_map={'long': '#2ecc71', 'short': '#e74c3c'}
        )
    
    fig_slip_entry.update_layout(
        yaxis_title="Slippage (points)" if mode == "Combined Strategies" else "Count",
        height=400,
        template="plotly_white"
    )
    st.plotly_chart(fig_slip_entry, use_container_width=True)

with col2:
    if mode == "Single Strategy":
        fig_slip_exit = px.histogram(
            df_combined,
            x="slippage_exit",
            color="posi",
            nbins=30,
            title="Exit Slippage Distribution",
            color_discrete_map={'long': '#2ecc71', 'short': '#e74c3c'}
        )
    else:
        fig_slip_exit = px.box(
            df_combined,
            y="slippage_exit",
            x="strategy_name",
            color="posi",
            title="Exit Slippage by Strategy",
            color_discrete_map={'long': '#2ecc71', 'short': '#e74c3c'}
        )
    
    fig_slip_exit.update_layout(
        yaxis_title="Slippage (points)" if mode == "Combined Strategies" else "Count",
        height=400,
        template="plotly_white"
    )
    st.plotly_chart(fig_slip_exit, use_container_width=True)

# Análise temporal do slippage
st.write("### Slippage by Hour of Day")

df_combined['entry_hour'] = pd.to_datetime(df_combined['time_ent']).dt.hour

if mode == "Single Strategy":
    hourly_slippage = df_combined.groupby('entry_hour').agg({
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
else:
    # Slippage por hora e estratégia
    hourly_slippage_by_strategy = df_combined.groupby(['entry_hour', 'strategy_name']).agg({
        'slippage_entry': 'mean',
        'slippage_exit': 'mean'
    }).round(2).reset_index()
    
    fig_hourly_slip = go.Figure()
    
    for strategy in df_combined['strategy_name'].unique():
        df_strategy = hourly_slippage_by_strategy[hourly_slippage_by_strategy['strategy_name'] == strategy]
        
        fig_hourly_slip.add_trace(go.Scatter(
            x=df_strategy['entry_hour'],
            y=df_strategy['slippage_entry'],
            mode='lines+markers',
            name=f'{strategy} - Entry',
            line=dict(width=2)
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
if mode == "Single Strategy":
    fig_timing = px.histogram(
        df_combined,
        x="timing_delta",
        nbins=50,
        title="Trade Duration Distribution",
        labels={'timing_delta': 'Duration (minutes)', 'count': 'Number of Trades'}
    )
else:
    fig_timing = px.box(
        df_combined,
        y="timing_delta",
        x="strategy_name",
        title="Trade Duration by Strategy",
        labels={'timing_delta': 'Duration (minutes)'}
    )

fig_timing.update_layout(
    height=400,
    template="plotly_white"
)

st.plotly_chart(fig_timing, use_container_width=True)

# Análise de TP/SL hits
if mode == "Single Strategy":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tp_hits = df_combined['hit_tp'].sum()
        tp_rate = (tp_hits / len(df_combined) * 100) if len(df_combined) > 0 else 0
        st.metric("TP Hits", f"{tp_hits} ({tp_rate:.1f}%)")
    
    with col2:
        sl_hits = df_combined['hit_sl'].sum()
        sl_rate = (sl_hits / len(df_combined) * 100) if len(df_combined) > 0 else 0
        st.metric("SL Hits", f"{sl_hits} ({sl_rate:.1f}%)")
    
    with col3:
        other_exits = len(df_combined) - tp_hits - sl_hits
        other_rate = (other_exits / len(df_combined) * 100) if len(df_combined) > 0 else 0
        st.metric("Other Exits", f"{other_exits} ({other_rate:.1f}%)")
else:
    # Tabela de TP/SL hits por estratégia
    hit_analysis = df_combined.groupby('strategy_name').agg({
        'hit_tp': ['sum', lambda x: (x.sum() / len(x) * 100)],
        'hit_sl': ['sum', lambda x: (x.sum() / len(x) * 100)]
    }).round(1)
    
    hit_analysis.columns = ['TP Hits', 'TP Rate (%)', 'SL Hits', 'SL Rate (%)']
    st.write("### TP/SL Hit Analysis by Strategy")
    st.dataframe(hit_analysis)

#################################
###  5. Detailed Comparison  ###
#################################

st.write("## 5. Trade-by-Trade Comparison")

# Scatter plot: Real vs Backtest profits
if mode == "Single Strategy":
    fig_scatter = px.scatter(
        df_combined,
        x=df_combined['pts_final'] * 0.2,
        y='profit',
        color='posi',
        title="Real vs Backtest Profits",
        labels={'x': 'Backtest Profit (R$)', 'profit': 'Real Profit (R$)'},
        color_discrete_map={'long': '#2ecc71', 'short': '#e74c3c'}
    )
else:
    fig_scatter = px.scatter(
        df_combined,
        x=df_combined['pts_final'] * 0.2,
        y='profit',
        color='strategy_name',
        symbol='posi',
        title="Real vs Backtest Profits by Strategy",
        labels={'x': 'Backtest Profit (R$)', 'profit': 'Real Profit (R$)'}
    )

# Adicionar linha de referência diagonal
x_range = [df_combined['pts_final'].min() * 0.2, df_combined['pts_final'].max() * 0.2]
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
summary_columns = ['time', 'strategy_name', 'posi', 'price_ent', 'price_ext', 'profit', 
                  'pts_final_real', 'slippage_entry', 'slippage_exit', 
                  'profit_delta', 'comment']

summary_df = df_combined[summary_columns].copy()
summary_df = summary_df.sort_values('time', ascending=False).head(50)

# Formatar colunas
summary_df['profit'] = summary_df['profit'].apply(lambda x: f"R$ {x:.2f}")
summary_df['profit_delta'] = summary_df['profit_delta'].apply(lambda x: f"R$ {x:.2f}")
summary_df['slippage_entry'] = summary_df['slippage_entry'].apply(lambda x: f"{x:.1f}")
summary_df['slippage_exit'] = summary_df['slippage_exit'].apply(lambda x: f"{x:.1f}")

# Renomear colunas para melhor visualização
summary_df.columns = ['Time', 'Strategy', 'Position', 'Entry Price', 'Exit Price', 
                     'Profit', 'Points', 'Entry Slip', 'Exit Slip', 'Delta', 'Comment']

st.dataframe(
    summary_df,
    hide_index=True,
    use_container_width=True
)

#################################
###  7. Key Insights  ###
#################################

st.write("## 7. Key Insights")

if mode == "Single Strategy":
    # Calcular métricas resumidas
    avg_entry_slip = df_combined['slippage_entry'].mean()
    avg_exit_slip = df_combined['slippage_exit'].mean()
    total_slip_cost = (avg_entry_slip + abs(avg_exit_slip)) * 0.2 * len(df_combined)
    correlation = df_combined['profit'].corr(df_combined['pts_final'] * 0.2)
    tp_rate = (df_combined['hit_tp'].sum() / len(df_combined) * 100)
    
    insights = f"""
### Performance Summary:

1. **Slippage Impact**: 
   - Average entry slippage: {avg_entry_slip:.1f} points
   - Average exit slippage: {avg_exit_slip:.1f} points
   - Estimated total slippage cost: R$ {total_slip_cost:.2f}

2. **Trade Accuracy**:
   - {len(df_combined)} trades matched between real and backtest
   - Average profit difference: R$ {df_combined['profit_delta'].mean():.2f}
   - Correlation between real and backtest: {correlation:.3f}

3. **Exit Analysis**:
   - TP hit rate: {tp_rate:.1f}%
   - SL hit rate: {(df_combined['hit_sl'].sum() / len(df_combined) * 100):.1f}%
   - Strategy exits well-calibrated: {'Yes' if abs(df_combined['profit_delta'].mean()) < 5 else 'Needs adjustment'}

4. **Recommendations**:
   - {'Slippage is within acceptable range' if abs(avg_entry_slip) < 10 else 'Consider adjusting entry timing'}
   - {'Exit strategy performing as expected' if tp_rate > 30 else 'Review TP/SL levels'}
"""
else:
    # Insights para múltiplas estratégias
    best_correlation = df_combined.groupby('strategy_name').apply(
        lambda x: x['profit'].corr(x['pts_final'] * 0.2)
    ).idxmax()
    
    worst_slippage = df_combined.groupby('strategy_name')['slippage_entry'].mean().abs().idxmax()
    
    insights = f"""
### Multi-Strategy Analysis Summary:

1. **Overall Performance**:
   - Total matched trades across all strategies: {len(df_combined)}
   - Average profit delta across all strategies: R$ {df_combined['profit_delta'].mean():.2f}
   - Strategies analyzed: {len(all_params)}

2. **Best Performers**:
   - Highest correlation (Real vs Backtest): **{best_correlation}**
   - Most consistent slippage: See table above for details

3. **Areas of Concern**:
   - Highest average slippage: **{worst_slippage}**
   - Review entry/exit timing for strategies with high slippage

4. **Strategy Comparison**:
   - Strategies show {'consistent' if df_combined.groupby('strategy_name')['profit_delta'].mean().std() < 5 else 'varying'} performance differences
   - {'Consider portfolio rebalancing' if len(all_params) > 2 else 'Monitor individual strategy performance'}
"""

st.markdown(insights)

# Adicionar gráfico de correlação se múltiplas estratégias
if mode == "Combined Strategies" and len(all_params) > 1:
    st.write("### Strategy Correlation Matrix")
    
    # Criar matriz de correlação entre estratégias
    correlation_data = df_combined.pivot_table(
        index=df_combined.index,
        columns='strategy_name',
        values='profit',
        aggfunc='first'
    )
    
    if len(correlation_data.columns) > 1:
        corr_matrix = correlation_data.corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig_corr.update_layout(
            title="Profit Correlation Between Strategies",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)