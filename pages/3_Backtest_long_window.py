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

st.write("# Long-Term Backtest Analysis ")

st.write("#### Comprehensive backtest performance analysis over extended periods")

st.markdown('''
        Before deploying any algorithmic trading strategy with real capital, it's crucial to:
        
        - **Analyze long-term performance** across different market conditions
        - **Identify periods of drawdown** and recovery patterns
        - **Evaluate consistency** of returns over time
        - **Understand risk metrics** throughout the backtest period
        
        This analysis uses the full backtest data to provide insights into strategy robustness.
                        
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

# Input para equity inicial
initial_equity = st.number_input(
    "Initial Equity (R$)", 
    min_value=1000.0, 
    value=10000.0, 
    step=1000.0,
    help="Enter the initial capital used in the backtest"
)

#################################
###  Load and Combine Data  ###
#################################

# Função para carregar dados de uma estratégia
@st.cache_data
def load_strategy_data(strategy_file):
    with open(strategy_file, 'r') as f:
        params = json.load(f)
    
    # Construir nome do arquivo CSV
    csv_file = f"bases/full_backtest_{params['symbol']}_{params['timeframe']}_{params['strategy']}_magic_{params['magic_number']}.csv"
    
    # Verificar se o arquivo existe
    if not os.path.exists(csv_file):
        st.error(f"Backtest file not found: {csv_file}")
        return None, params
    
    # Carregar dados
    df = pd.read_csv(csv_file, index_col=['time'], parse_dates=['time'])
    df['cstrategy'] = df['cstrategy'].ffill()
    
    return df, params

# Carregar dados de todas as estratégias selecionadas
all_data = []
all_params = []

for strategy_file in selected_strategies:
    df, params = load_strategy_data(strategy_file)
    if df is not None:
        all_data.append(df)
        all_params.append(params)

if not all_data:
    st.error("No valid backtest data found for selected strategies.")
    st.stop()

# Combinar dados se múltiplas estratégias
if len(all_data) == 1:
    # Estratégia única
    df = all_data[0]
    strategy_info = f"{all_params[0]['strategy']} - {all_params[0]['symbol']}"
else:
    # Combinar múltiplas estratégias
    # Alinhar todos os DataFrames pelo índice de tempo
    combined_returns = pd.DataFrame()
    
    for i, (data, params) in enumerate(zip(all_data, all_params)):
        # Calcular retornos diários
        daily_returns = data['cstrategy'].diff().fillna(0)
        combined_returns[f"strategy_{i}"] = daily_returns
    
    # Somar retornos de todas as estratégias
    df = pd.DataFrame(index=combined_returns.index)
    df['cstrategy'] = combined_returns.sum(axis=1).cumsum()
    
    strategy_info = f"Combined ({len(all_data)} strategies)"

# Adicionar equity inicial ao cstrategy para obter equity total
df['equity'] = df['cstrategy'] + initial_equity

# Calcular métricas derivadas
df['returns'] = df['cstrategy'].diff().fillna(0)
df['returns_pct'] = df['returns'] / initial_equity  # Retorno percentual baseado no capital inicial

# Calcular drawdown corretamente
df['equity_cummax'] = df['equity'].cummax()
df['drawdown'] = df['equity'] - df['equity_cummax']
df['drawdown_pct'] = (df['drawdown'] / df['equity_cummax'] * 100)

# Mostrar informações da estratégia
st.write(f"### Analyzing: {strategy_info}")

if len(all_data) > 1:
    # Mostrar detalhes das estratégias combinadas
    with st.expander("Combined Strategies Details"):
        for params in all_params:
            st.write(f"- **{params['strategy']}** on {params['symbol']} ({params['timeframe']}) - Magic: {params['magic_number']}")

#################################
###  1. Overview Metrics  ###
#################################

st.write("## 1. Performance Overview")

# Calcular métricas principais
total_return = df['cstrategy'].iloc[-1]
total_return_pct = (total_return / initial_equity) * 100
total_days = (df.index[-1] - df.index[0]).days
years = total_days / 365.25
annual_return = (total_return / years) if years > 0 else 0
annual_return_pct = (annual_return / initial_equity) * 100
max_drawdown = df['drawdown'].min()
max_drawdown_pct = df['drawdown_pct'].min()

# Calcular Sharpe Ratio corretamente
daily_returns_pct = df['returns_pct'].dropna()
if len(daily_returns_pct) > 1 and daily_returns_pct.std() > 0:
    sharpe_ratio = np.sqrt(252) * (daily_returns_pct.mean() / daily_returns_pct.std())
else:
    sharpe_ratio = 0

# Calcular win rate
daily_returns = df['returns'].dropna()
winning_days = len(daily_returns[daily_returns > 0])
total_trading_days = len(daily_returns[daily_returns != 0])
win_rate = (winning_days / total_trading_days * 100) if total_trading_days > 0 else 0

# Calcular Profit Factor
profits = daily_returns[daily_returns > 0].sum()
losses = abs(daily_returns[daily_returns < 0].sum())
profit_factor = (profits / losses) if losses > 0 else float('inf') if profits > 0 else 0

# Exibir métricas em cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Return", f"R$ {total_return:,.1f}", f"{total_return_pct:.1f}%")
    st.metric("Annual Return", f"R$ {annual_return:,.1f}", f"{annual_return_pct:.1f}%")

with col2:
    st.metric("Max Drawdown", f"R$ {max_drawdown:,.1f}")
    st.metric("Max Drawdown %", f"{max_drawdown_pct:.1f}%")

with col3:
    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    st.metric("Profit Factor", f"{profit_factor:.2f}")

with col4:
    st.metric("Total Days", f"{total_days:,}")
    st.metric("Years Tested", f"{years:.1f}")

#################################
###  2. Cumulative Return  ###
#################################

st.write("## 2. Cumulative Return Analysis")

# Criar figura principal
fig_return = go.Figure()

# Se múltiplas estratégias, mostrar contribuição individual
if len(all_data) > 1:
    # Adicionar linhas individuais
    cumsum = pd.DataFrame()
    for i, (data, params) in enumerate(zip(all_data, all_params)):
        returns = data['cstrategy'].diff().fillna(0)
        cumsum[f"strategy_{i}"] = returns.cumsum()
        
        fig_return.add_trace(go.Scatter(
            x=data.index,
            y=cumsum[f"strategy_{i}"],
            name=f"{params['strategy']} - {params['symbol']}",
            line=dict(width=1),
            opacity=0.7
        ))

# Adicionar linha de retorno acumulado total
fig_return.add_trace(go.Scatter(
    x=df.index,
    y=df['cstrategy'],
    name='Total Cumulative Return',
    line=dict(color='#1f77b4', width=3),
    fill='tozeroy' if len(all_data) == 1 else None,
    fillcolor='rgba(31, 119, 180, 0.1)' if len(all_data) == 1 else None
))

# Adicionar linha de tendência
from scipy import stats
x_numeric = np.arange(len(df))
slope, intercept, _, _, _ = stats.linregress(x_numeric, df['cstrategy'])
trend_line = slope * x_numeric + intercept

fig_return.add_trace(go.Scatter(
    x=df.index,
    y=trend_line,
    name='Trend',
    line=dict(color='red', width=2, dash='dash')
))

# Configurar layout
fig_return.update_layout(
    title="Cumulative Return Over Time",
    xaxis_title="Date",
    yaxis_title="Return (R$)",
    height=500,
    template="plotly_white",
    hovermode='x unified',
    showlegend=True
)

st.plotly_chart(fig_return, use_container_width=True)

# Análise por período
st.write("### Return by Period")

# Calcular retornos por diferentes períodos
monthly_returns = df['cstrategy'].resample('M').last().diff().dropna()
yearly_returns = df['cstrategy'].resample('Y').last().diff().dropna()

col1, col2 = st.columns(2)

with col1:
    # Histograma de retornos mensais
    fig_monthly = px.histogram(
        monthly_returns,
        nbins=30,
        title="Monthly Return Distribution",
        labels={'value': 'Return (R$)', 'count': 'Frequency'}
    )
    fig_monthly.update_layout(
        showlegend=False,
        height=300,
        template="plotly_white"
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

with col2:
    # Box plot de retornos anuais
    if len(yearly_returns) > 0:
        fig_yearly = px.box(
            yearly_returns,
            title="Yearly Return Distribution",
            labels={'value': 'Return (R$)'}
        )
        fig_yearly.update_layout(
            showlegend=False,
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig_yearly, use_container_width=True)

#################################
###  3. Drawdown Analysis  ###
#################################

st.write("## 3. Drawdown Analysis")

# Criar figura de drawdown
fig_dd = go.Figure()

# Adicionar área de drawdown
fig_dd.add_trace(go.Scatter(
    x=df.index,
    y=df['drawdown'],
    name='Drawdown (R$)',
    fill='tozeroy',
    fillcolor='rgba(255, 0, 0, 0.3)',
    line=dict(color='red', width=1)
))

# Identificar períodos de drawdown significativos
threshold = max_drawdown * 0.5  # 50% do max drawdown
significant_dd = df[df['drawdown'] < threshold]

if len(significant_dd) > 0:
    fig_dd.add_trace(go.Scatter(
        x=significant_dd.index,
        y=significant_dd['drawdown'],
        mode='markers',
        name='Significant Drawdowns',
        marker=dict(color='darkred', size=6)
    ))

# Configurar layout
fig_dd.update_layout(
    title="Drawdown Over Time",
    xaxis_title="Date",
    yaxis_title="Drawdown (R$)",
    height=400,
    template="plotly_white",
    hovermode='x unified'
)

st.plotly_chart(fig_dd, use_container_width=True)

# Análise de recuperação de drawdown
st.write("### Drawdown Recovery Analysis")

# Calcular períodos de drawdown
dd_periods = []
in_dd = False
start_dd = None

for idx, row in df.iterrows():
    if row['drawdown'] < 0 and not in_dd:
        in_dd = True
        start_dd = idx
    elif row['drawdown'] >= -0.01 and in_dd:  # Pequena tolerância para drawdown zero
        in_dd = False
        if start_dd:
            duration = (idx - start_dd).days
            max_dd_period = df.loc[start_dd:idx, 'drawdown'].min()
            dd_periods.append({
                'Start': start_dd,
                'End': idx,
                'Duration (days)': duration,
                'Max Drawdown': max_dd_period
            })

if dd_periods:
    dd_df = pd.DataFrame(dd_periods).sort_values('Max Drawdown').head(10)
    st.write("**Top 10 Largest Drawdown Periods:**")
    st.dataframe(dd_df.style.format({
        'Max Drawdown': 'R$ {:,.2f}',
        'Start': lambda x: x.strftime('%Y-%m-%d'),
        'End': lambda x: x.strftime('%Y-%m-%d')
    }))

#################################
###  4. Risk Metrics  ###
#################################

st.write("## 4. Risk Metrics Over Time")

# Calcular métricas rolantes
window = 252  # 1 ano
df['rolling_returns_pct'] = df['returns'].rolling(window).sum() / initial_equity
df['rolling_volatility'] = df['returns_pct'].rolling(window).std() * np.sqrt(252)

# Calcular Sharpe rolante
df['rolling_sharpe'] = df['returns_pct'].rolling(window).apply(
    lambda x: np.sqrt(252) * (x.mean() / x.std()) if len(x) > 1 and x.std() > 0 else 0
)

# Criar subplots
fig_risk = go.Figure()

# Sharpe ratio rolante
fig_risk.add_trace(go.Scatter(
    x=df.index[window:],
    y=df['rolling_sharpe'].iloc[window:],
    name='Rolling Sharpe Ratio (1Y)',
    line=dict(color='green', width=2)
))

# Adicionar linha de referência
fig_risk.add_hline(y=1, line_dash="dash", line_color="gray", 
                   annotation_text="Sharpe = 1", annotation_position="right")

fig_risk.update_layout(
    title="Rolling Sharpe Ratio (1 Year Window)",
    xaxis_title="Date",
    yaxis_title="Sharpe Ratio",
    height=400,
    template="plotly_white"
)

st.plotly_chart(fig_risk, use_container_width=True)

#################################
###  5. Performance by Year  ###
#################################

st.write("## 5. Annual Performance Breakdown")

# Criar análise anual
df['year'] = df.index.year
annual_summary = df.groupby('year').agg({
    'returns': ['sum', 'std', 'count'],
    'drawdown': 'min'
}).round(2)

annual_summary.columns = ['Total Return', 'Volatility', 'Trading Days', 'Max Drawdown']

# Calcular Sharpe anual corretamente
annual_summary['Sharpe'] = annual_summary.apply(
    lambda row: (row['Total Return'] / initial_equity) / (row['Volatility'] / initial_equity * np.sqrt(row['Trading Days'])) * np.sqrt(252) 
    if row['Volatility'] > 0 and row['Trading Days'] > 0 else 0, axis=1
)

# Gráfico de barras anuais
fig_annual = go.Figure()

colors = ['green' if x > 0 else 'red' for x in annual_summary['Total Return']]

fig_annual.add_trace(go.Bar(
    x=annual_summary.index,
    y=annual_summary['Total Return'],
    name='Annual Return',
    marker_color=colors
))

fig_annual.update_layout(
    title="Annual Returns",
    xaxis_title="Year",
    yaxis_title="Return (R$)",
    height=400,
    template="plotly_white"
)

st.plotly_chart(fig_annual, use_container_width=True)

# Tabela resumo anual
st.write("### Annual Performance Summary")
st.dataframe(annual_summary.style.format({
    'Total Return': 'R$ {:,.2f}',
    'Max Drawdown': 'R$ {:,.2f}',
    'Volatility': 'R$ {:,.2f}',
    'Sharpe': '{:.2f}'
}))

#################################
### 6. Strategy Comparison  ###
#################################

if len(all_data) > 1:
    st.write("## 6. Individual Strategy Performance")
    
    # Calcular métricas individuais
    strategy_metrics = []
    
    for i, (data, params) in enumerate(zip(all_data, all_params)):
        returns = data['cstrategy'].diff().fillna(0)
        total_ret = data['cstrategy'].iloc[-1]
        
        # Sharpe individual
        ret_pct = returns / (initial_equity/len(all_data))
        sharpe = np.sqrt(252) * (ret_pct.mean() / ret_pct.std()) if ret_pct.std() > 0 else 0
        
        # Win rate individual
        wins = len(returns[returns > 0])
        trades = len(returns[returns != 0])
        wr = (wins / trades * 100) if trades > 0 else 0
        
        # Profit Factor individual
        profits_ind = returns[returns > 0].sum()
        losses_ind = abs(returns[returns < 0].sum())
        pf_ind = (profits_ind / losses_ind) if losses_ind > 0 else float('inf') if profits_ind > 0 else 0
        
        strategy_metrics.append({
            'Strategy': f"{params['strategy']} - {params['symbol']}",
            'Total Return': total_ret,
            'Return %': (total_ret / (initial_equity/len(all_data))) * 100,
            'Sharpe Ratio': sharpe,
            'Win Rate': wr,
            'Profit Factor': pf_ind,
            'Max DD': returns.cumsum().cummax().sub(returns.cumsum()).min()
        })
    
    metrics_df = pd.DataFrame(strategy_metrics)
    st.dataframe(metrics_df.style.format({
        'Total Return': 'R$ {:,.2f}',
        'Return %': '{:.1f}%',
        'Sharpe Ratio': '{:.2f}',
        'Win Rate': '{:.1f}%',
        'Profit Factor': lambda x: f'{x:.2f}' if x != float('inf') else '∞',
        'Max DD': 'R$ {:,.2f}'
    }))

    
#################################
### 6.5 Strategy Correlation  ###
#################################

if len(all_data) > 1:
    st.write("## 6.5 Strategy Correlation Analysis")
    
    # Calcular retornos semanais para cada estratégia
    weekly_returns = pd.DataFrame()
    
    for i, (data, params) in enumerate(zip(all_data, all_params)):
        # Retornos absolutos semanais
        strategy_name = f"{params['strategy']} - {params['symbol']}"
        weekly_ret = data['cstrategy'].resample('W').last().diff().fillna(0)
        weekly_returns[strategy_name] = weekly_ret
    
    # Alinhar índices
    weekly_returns = weekly_returns.dropna()
    
    if len(weekly_returns) > 1:
        # Calcular matriz de correlação
        correlation_matrix = weekly_returns.corr()
        
        # Criar heatmap de correlação
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig_corr.update_layout(
            title="Weekly Returns Correlation Matrix",
            height=500,
            width=700,
            xaxis_title="",
            yaxis_title="",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Estatísticas de correlação
        st.write("### Correlation Statistics")
        
        # Extrair correlações únicas (triangular superior)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        upper_corr = correlation_matrix.where(mask)
        
        corr_values = upper_corr.values[mask]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Correlation", f"{corr_values.mean():.3f}")
        with col2:
            st.metric("Max Correlation", f"{corr_values.max():.3f}")
        with col3:
            st.metric("Min Correlation", f"{corr_values.min():.3f}")
        
        # Aviso sobre diversificação
        avg_corr = corr_values.mean()
        if avg_corr > 0.7:
            st.warning("⚠️ High average correlation detected. Strategies may not provide sufficient diversification.")
        elif avg_corr < 0.3:
            st.success("✅ Low correlation between strategies. Good diversification potential.")
        else:
            st.info("ℹ️ Moderate correlation between strategies.")

#################################
###  7. Summary Statistics  ###
#################################

st.write("## 7. Summary Statistics")

# Calcular Sortino Ratio
negative_returns = daily_returns[daily_returns < 0]
if len(negative_returns) > 0:
    sortino_ratio = np.sqrt(252) * (daily_returns_pct.mean() / (negative_returns / initial_equity).std())
else:
    sortino_ratio = np.nan

# Criar tabela resumo
summary_stats = {
    'Metric': ['Initial Equity', 'Total Return', 'Annual Return', 'Monthly Avg Return', 'Daily Avg Return',
               'Max Drawdown', 'Max DD Duration', 'Sharpe Ratio', 'Sortino Ratio', 'Profit Factor',
               'Win Rate', 'Best Day', 'Worst Day', 'Volatility (Annual)'],
    'Value': [
        f"R$ {initial_equity:,.2f}",
        f"R$ {total_return:,.2f} ({total_return_pct:.1f}%)",
        f"R$ {annual_return:,.2f} ({annual_return_pct:.1f}%)",
        f"R$ {monthly_returns.mean():,.2f}" if len(monthly_returns) > 0 else "N/A",
        f"R$ {daily_returns.mean():,.2f}",
        f"R$ {max_drawdown:,.2f} ({max_drawdown_pct:.1f}%)",
        f"{max([p['Duration (days)'] for p in dd_periods])} days" if dd_periods else "N/A",
        f"{sharpe_ratio:.2f}",
        f"{sortino_ratio:.2f}" if not np.isnan(sortino_ratio) else "N/A",
        f"{profit_factor:.2f}",
        f"{win_rate:.1f}%",
        f"R$ {daily_returns.max():,.2f}",
        f"R$ {daily_returns.min():,.2f}",
        f"{daily_returns_pct.std() * np.sqrt(252):.2%}" if daily_returns_pct.std() > 0 else "N/A"
    ]
}

summary_df = pd.DataFrame(summary_stats)
st.table(summary_df)

# Insights finais
st.write("### Key Insights")

# Personalizar insights baseado no modo
if len(all_data) == 1:
    strategy_text = f"The {all_params[0]['strategy']} strategy"
else:
    strategy_text = f"The combined portfolio of {len(all_data)} strategies"

insights = f"""
Based on the backtest analysis:

1. **Performance**: {strategy_text} generated R$ {total_return:,.2f} ({total_return_pct:.1f}% return) over {years:.1f} years, 
   averaging R$ {annual_return:,.2f} ({annual_return_pct:.1f}%) per year.

2. **Risk**: Maximum drawdown was R$ {max_drawdown:,.2f} ({max_drawdown_pct:.1f}%), 
   with a Sharpe ratio of {sharpe_ratio:.2f}.

3. **Consistency**: Win rate of {win_rate:.1f}% with {"positive" if sharpe_ratio > 1 else "moderate"} risk-adjusted returns.

4. **Recommendation**: {"Strategy shows strong historical performance" if sharpe_ratio > 1.5 else "Strategy shows acceptable performance" if sharpe_ratio > 0.5 else "Consider strategy adjustments"}.
"""

st.markdown(insights)