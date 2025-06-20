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

st.write("# Long-Term Backtest Analysis")

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
###  0. Load Configuration  ###
#################################

# Carregar configuração da estratégia
with open('combined_strategy_teste.json', 'r') as f:
    strategy_params = json.load(f)

symbol = strategy_params['symbol']
timeframe = strategy_params['timeframe']
strategy_name = strategy_params['strategy']
magic_number = strategy_params['magic_number']

# Mostrar informações da estratégia
st.write(f"### Strategy: {strategy_name}")
st.write(f"**Symbol:** {symbol} | **Timeframe:** {timeframe} | **Magic:** {magic_number}")

#################################
###  Load Backtest Data  ###
#################################

# Carregar dados de backtest completo
df = pd.read_csv(f'bases/full_backtest_{symbol}_{timeframe}_{strategy_name}_magic_{magic_number}.csv', 
                 index_col=['time'], parse_dates=['time'])

# Ajuste
df['cstrategy'] = df['cstrategy'].ffill()

# Calcular métricas derivadas
df['returns'] = df['cstrategy'].diff()
df['cummax'] = df['cstrategy'].cummax()
df['drawdown'] = df['cstrategy'] - df['cummax']
df['drawdown_pct'] = (df['drawdown'] / df['cummax'] * 100).fillna(0)

#################################
###  1. Overview Metrics  ###
#################################

st.write("## 1. Performance Overview")

# Calcular métricas principais
total_return = df['cstrategy'].iloc[-1]
total_days = (df.index[-1] - df.index[0]).days
years = total_days / 365.25
annual_return = (total_return / years) if years > 0 else 0
max_drawdown = df['drawdown'].min()
max_drawdown_pct = df['drawdown_pct'].min()

# Calcular Sharpe Ratio (assumindo 0% de taxa livre de risco)
daily_returns = df['returns'].dropna()
sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() > 0 else 0

# Calcular win rate
winning_days = len(daily_returns[daily_returns > 0])
total_trading_days = len(daily_returns[daily_returns != 0])
win_rate = (winning_days / total_trading_days * 100) if total_trading_days > 0 else 0

# Exibir métricas em cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Return", f"R$ {total_return:,.2f}")
    st.metric("Annual Return", f"R$ {annual_return:,.2f}")

with col2:
    st.metric("Max Drawdown", f"R$ {max_drawdown:,.2f}")
    st.metric("Max Drawdown %", f"{max_drawdown_pct:.1f}%")

with col3:
    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    st.metric("Win Rate", f"{win_rate:.1f}%")

with col4:
    st.metric("Total Days", f"{total_days:,}")
    st.metric("Years Tested", f"{years:.1f}")

#################################
###  2. Cumulative Return  ###
#################################

st.write("## 2. Cumulative Return Analysis")

print(df.head(7))

# Criar figura principal
fig_return = go.Figure()

# Adicionar linha de retorno acumulado
fig_return.add_trace(go.Scatter(
    x=df.index,
    y=df['cstrategy'],
    name='Cumulative Return',
    line=dict(color='#1f77b4', width=2),
    fill='tozeroy',
    fillcolor='rgba(31, 119, 180, 0.1)'
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
    elif row['drawdown'] == 0 and in_dd:
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
df['rolling_sharpe'] = df['returns'].rolling(window).apply(
    lambda x: np.sqrt(252) * (x.mean() / x.std()) if x.std() > 0 else 0
)
df['rolling_volatility'] = df['returns'].rolling(window).std() * np.sqrt(252)

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
annual_summary['Sharpe'] = (annual_summary['Total Return'] / annual_summary['Volatility'] / np.sqrt(annual_summary['Trading Days'])) * np.sqrt(252)

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
    'Volatility': '{:.2f}',
    'Sharpe': '{:.2f}'
}))

#################################
###  6. Monte Carlo Analysis  ###
#################################

st.write("## 6. Monte Carlo Simulation")

# Simular diferentes caminhos baseados na distribuição histórica de retornos
n_simulations = 1000
n_days = 252  # 1 ano

# Obter parâmetros da distribuição
returns_clean = daily_returns.dropna()
mu = returns_clean.mean()
sigma = returns_clean.std()

# Realizar simulações
simulations = np.zeros((n_simulations, n_days))
for i in range(n_simulations):
    random_returns = np.random.normal(mu, sigma, n_days)
    simulations[i] = np.cumprod(1 + random_returns) - 1

# Calcular percentis
percentiles = np.percentile(simulations, [5, 25, 50, 75, 95], axis=0)

# Criar gráfico
fig_mc = go.Figure()

# Adicionar faixa de confiança
x_days = list(range(n_days))
fig_mc.add_trace(go.Scatter(
    x=x_days + x_days[::-1],
    y=list(percentiles[4] * total_return) + list(percentiles[0][::-1] * total_return),
    fill='toself',
    fillcolor='rgba(0,100,200,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='90% Confidence Interval'
))

# Adicionar mediana
fig_mc.add_trace(go.Scatter(
    x=x_days,
    y=percentiles[2] * total_return,
    line=dict(color='blue', width=2),
    name='Median Path'
))

fig_mc.update_layout(
    title="Monte Carlo Simulation - 1 Year Forward",
    xaxis_title="Days",
    yaxis_title="Expected Return (R$)",
    height=400,
    template="plotly_white"
)

st.plotly_chart(fig_mc, use_container_width=True)

# Estatísticas da simulação
final_values = simulations[:, -1] * total_return
st.write("### Expected 1-Year Returns (Based on Historical Distribution)")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("5th Percentile", f"R$ {np.percentile(final_values, 5):,.2f}")
with col2:
    st.metric("Median", f"R$ {np.percentile(final_values, 50):,.2f}")
with col3:
    st.metric("95th Percentile", f"R$ {np.percentile(final_values, 95):,.2f}")

#################################
###  7. Summary Statistics  ###
#################################

st.write("## 7. Summary Statistics")

# Criar tabela resumo
summary_stats = {
    'Metric': ['Total Return', 'Annual Return', 'Monthly Avg Return', 'Daily Avg Return',
               'Max Drawdown', 'Max DD Duration', 'Sharpe Ratio', 'Sortino Ratio',
               'Win Rate', 'Best Day', 'Worst Day', 'Volatility (Annual)'],
    'Value': [
        f"R$ {total_return:,.2f}",
        f"R$ {annual_return:,.2f}",
        f"R$ {monthly_returns.mean():,.2f}" if len(monthly_returns) > 0 else "N/A",
        f"R$ {daily_returns.mean():,.2f}",
        f"R$ {max_drawdown:,.2f} ({max_drawdown_pct:.1f}%)",
        f"{max([p['Duration (days)'] for p in dd_periods])} days" if dd_periods else "N/A",
        f"{sharpe_ratio:.2f}",
        f"{np.sqrt(252) * (daily_returns[daily_returns > 0].mean() / daily_returns[daily_returns < 0].std()):.2f}" if len(daily_returns[daily_returns < 0]) > 0 else "N/A",
        f"{win_rate:.1f}%",
        f"R$ {daily_returns.max():,.2f}",
        f"R$ {daily_returns.min():,.2f}",
        f"{daily_returns.std() * np.sqrt(252):.2f}"
    ]
}

summary_df = pd.DataFrame(summary_stats)
st.table(summary_df)

# Insights finais
st.write("### Key Insights")

insights = f"""
Based on the backtest analysis:

1. **Performance**: The strategy generated R$ {total_return:,.2f} over {years:.1f} years, 
   averaging R$ {annual_return:,.2f} per year.

2. **Risk**: Maximum drawdown was R$ {max_drawdown:,.2f} ({max_drawdown_pct:.1f}%), 
   with a Sharpe ratio of {sharpe_ratio:.2f}.

3. **Consistency**: Win rate of {win_rate:.1f}% with {"positive" if sharpe_ratio > 1 else "moderate"} risk-adjusted returns.

4. **Recommendation**: {"Strategy shows strong historical performance" if sharpe_ratio > 1.5 else "Strategy shows acceptable performance" if sharpe_ratio > 0.5 else "Consider strategy adjustments"}.
"""

st.markdown(insights)