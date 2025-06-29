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
# Função cached apenas para processamento de dados (SEM widgets)
@st.cache_data
def load_strategy_data_cached(strategy_file):
    """
    Carrega e processa dados de estratégia - APENAS lógica de dados, sem widgets
    """
    try:
        with open(strategy_file, 'r') as f:
            params = json.load(f)
        
        symbol = params['symbol']
        timeframe = params['timeframe']
        strategy_name = params['strategy']
        magic_number = params['magic_number']
        
        # Pegar TP/SL - tentar diferentes estruturas
        tp_points = None
        sl_points = None
        
        # Tentar pegar da estrutura hour_params
        if 'hour_params' in params and params['hour_params']:
            # Pegar as chaves disponíveis (podem ser strings ou números)
            hour_keys = list(params['hour_params'].keys())
            
            # Tentar encontrar uma hora válida
            # Primeiro tentar horas comuns (9, 10, 11)
            for hour in ['9', 9, '10', 10, '11', 11]:
                if str(hour) in params['hour_params']:
                    hour_data = params['hour_params'][str(hour)]
                    tp_points = hour_data.get('tp')
                    sl_points = hour_data.get('sl')
                    break
                elif hour in params['hour_params']:
                    hour_data = params['hour_params'][hour]
                    tp_points = hour_data.get('tp')
                    sl_points = hour_data.get('sl')
                    break
            
            # Se não encontrou, pegar do primeiro horário disponível
            if tp_points is None and hour_keys:
                first_hour = hour_keys[0]
                hour_data = params['hour_params'][first_hour]
                tp_points = hour_data.get('tp')
                sl_points = hour_data.get('sl')
        
        # Tentar estrutura alternativa (tp/sl direto nos params)
        if tp_points is None:
            tp_points = params.get('tp', params.get('take_profit', 50))  # valor padrão 50
        if sl_points is None:
            sl_points = params.get('sl', params.get('stop_loss', 50))  # valor padrão 50
        
        # Construir nomes dos arquivos
        real_file = f'bases/results_{symbol}_{timeframe}_{strategy_name}_magic_{magic_number}.csv'
        backtest_file = f'bases/backtest_{symbol}_{timeframe}_{strategy_name}_magic_{magic_number}.csv'
        
        # Verificar se os arquivos existem
        files_status = {
            'real_file_exists': os.path.exists(real_file),
            'backtest_file_exists': os.path.exists(backtest_file),
            'real_file': real_file,
            'backtest_file': backtest_file
        }
        
        if not files_status['real_file_exists'] or not files_status['backtest_file_exists']:
            return None, params, files_status
        
        # Carregar dados reais
        try:
            df_real = pd.read_csv(real_file, parse_dates=['time', 'time_ent', 'time_ext'])
        except Exception as e:
            return None, params, {'error': f"Error reading real trades file: {str(e)}"}
        
        # Verificar se tem dados
        if df_real.empty:
            return None, params, {'error': "No real trades found"}
        
        # Carregar dados de backtest
        try:
            df_backtest = pd.read_csv(backtest_file, parse_dates=['time'])
        except Exception as e:
            return None, params, {'error': f"Error reading backtest file: {str(e)}"}
        
        # Verificar se tem a coluna position
        if 'position' not in df_backtest.columns:
            return None, params, {'error': "Backtest file missing 'position' column"}
        
        # Preparar dados do backtest - filtrar apenas trades
        df_backtest_trades = df_backtest[df_backtest['position'] != 0].copy()
        
        # Verificar se há trades no backtest
        if df_backtest_trades.empty:
            return None, params, {'error': "No trades found in backtest"}
        
        # Merge dos dados - matching por horário próximo
        df_real['time_rounded'] = df_real['time'].dt.round('5min')
        df_backtest_trades['time_rounded'] = df_backtest_trades['time'].dt.round('5min')
        
        # Verificar colunas do backtest
        backtest_columns = df_backtest_trades.columns.tolist()
        
        # Verificar se tem a coluna pts_final ou similar
        pts_col = None
        for col in ['pts_final', 'points', 'pnl', 'profit_points']:
            if col in backtest_columns:
                pts_col = col
                break
        
        if pts_col is None:
            return None, params, {
                'error': "No points/profit column found in backtest",
                'available_columns': backtest_columns
            }
        
        # Selecionar colunas necessárias do backtest
        merge_cols = ['time_rounded', 'open', 'close', 'position']
        if pts_col != 'pts_final':
            df_backtest_trades['pts_final'] = df_backtest_trades[pts_col]
        merge_cols.append('pts_final')
        
        # Fazer o merge
        df_merged = pd.merge(
            df_real,
            df_backtest_trades[merge_cols],
            on='time_rounded',
            how='inner',
            suffixes=('_real', '_backtest')
        )
        
        # Verificar se o merge resultou em dados
        if df_merged.empty:
            return None, params, {'error': "No matching trades found between real and backtest"}
        
        # Verificar colunas necessárias
        required_cols = ['posi', 'price_ent', 'price_ext', 'profit', 'time_ent', 'time_ext']
        missing_cols = [col for col in required_cols if col not in df_merged.columns]
        if missing_cols:
            return None, params, {
                'error': f"Missing required columns: {missing_cols}",
                'available_columns': df_merged.columns.tolist()
            }
        
        # Obter valor por ponto do arquivo de parâmetros
        valor_por_ponto = params.get('valor_lote', params.get('tc', 0.2))
        
        # Verificar se tem pts_final_real ou calcular
        if 'pts_final_real' not in df_merged.columns:
            # Tentar calcular pontos baseado no lucro e valor por ponto
            if 'profit' in df_merged.columns and valor_por_ponto > 0:
                df_merged['pts_final_real'] = df_merged['profit'] / valor_por_ponto
            else:
                # Usar diferença de preços
                df_merged['pts_final_real'] = np.where(
                    df_merged['posi'] == 'long',
                    df_merged['price_ext'] - df_merged['price_ent'],
                    df_merged['price_ent'] - df_merged['price_ext']
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
            df_merged['price_ext'] - df_merged['pts_final'] - df_merged['close'],
            df_merged['price_ext'] + df_merged['pts_final'] - df_merged['close']
        )
        
        df_merged['profit_delta'] = df_merged['profit'] - (df_merged['pts_final'] * valor_por_ponto)
        df_merged['timing_delta'] = (df_merged['time_ext'] - df_merged['time_ent']).dt.total_seconds() / 60
        
        # Identificar hits de TP/SL
        # Garantir que a coluna comment seja string
        if 'comment' in df_merged.columns:
            df_merged['comment'] = df_merged['comment'].fillna('').astype(str)
            df_merged['hit_tp'] = df_merged['comment'].str.contains('tp', case=False, na=False)
            df_merged['hit_sl'] = df_merged['comment'].str.contains('sl', case=False, na=False)
        else:
            # Se não tem coluna comment, criar com valores False
            df_merged['hit_tp'] = False
            df_merged['hit_sl'] = False
        
        # Adicionar informações extras
        df_merged['tp_points'] = tp_points
        df_merged['sl_points'] = sl_points
        df_merged['valor_por_ponto'] = valor_por_ponto
        
        # Retornar dados processados e informações de debug
        debug_info = {
            'real_columns': df_real.columns.tolist(),
            'real_shape': df_real.shape,
            'backtest_columns': backtest_columns,
            'backtest_shape': df_backtest_trades.shape,
            'merged_columns': df_merged.columns.tolist(),
            'merged_shape': df_merged.shape,
            'pts_col_used': pts_col,
            'valor_por_ponto': valor_por_ponto
        }
        
        return df_merged, params, debug_info
        
    except Exception as e:
        return None, None, {'error': f"Error processing {strategy_file}: {str(e)}"}


# Função NÃO cached para interface do usuário (COM widgets)
def load_and_process_strategy_data_with_ui(strategy_file):
    """
    Wrapper que chama a função cached e lida com a interface do usuário
    """
    # Chamar a função cached
    result = load_strategy_data_cached(strategy_file)
    
    if result is None or len(result) != 3:
        st.error(f"Error loading strategy data from {strategy_file}")
        return None, None
    
    df_merged, params, info = result
    
    # Se houve erro, mostrar na interface
    if df_merged is None:
        if 'error' in info:
            if 'real_file_exists' in info:
                # Erro de arquivo não encontrado
                if not info['real_file_exists']:
                    st.info(f"Real trades file not found for {params['strategy']} - {params['symbol']}")
                    st.info(f"Expected file: {info['real_file']}")
                if not info['backtest_file_exists']:
                    st.info(f"Backtest file not found for {params['strategy']} - {params['symbol']}")
                    st.info(f"Expected file: {info['backtest_file']}")
            else:
                # Outros tipos de erro
                st.error(f"Error processing {params['strategy']} - {params['symbol']}: {info['error']}")
                if 'available_columns' in info:
                    st.info(f"Available columns: {', '.join(info['available_columns'])}")
        return None, params
    
    # Mostrar informações de debug se solicitado
    strategy_name = params['strategy']
    magic_number = params['magic_number']
    
    if st.checkbox(f"Show debug info for {strategy_name}", key=f"debug_{strategy_name}_{magic_number}"):
        st.write(f"Real data columns: {', '.join(info['real_columns'])}")
        st.write(f"Real data shape: {info['real_shape']}")
        st.write("First few rows:")
        st.dataframe(df_merged.head())
    
    if st.checkbox(f"Show backtest columns for {strategy_name}", key=f"debug_bt_{strategy_name}_{magic_number}"):
        st.write(f"Backtest columns: {', '.join(info['backtest_columns'])}")
        st.write(f"Backtest shape: {info['backtest_shape']}")
    
    if st.checkbox(f"Show merged data info for {strategy_name}", key=f"debug_merged_{strategy_name}_{magic_number}"):
        st.write(f"Merged columns: {', '.join(info['merged_columns'])}")
        st.write(f"Merged shape: {info['merged_shape']}")
        st.write(f"Points column used: {info['pts_col_used']}")
        st.write(f"Value per point: {info['valor_por_ponto']}")
        if not df_merged.empty:
            st.write("Sample merged data:")
            st.dataframe(df_merged.head())
    
    return df_merged, params

# Carregar dados de todas as estratégias selecionadas
all_merged_data = []
all_params = []

for strategy_file in selected_strategies:
    df_merged, params = load_and_process_strategy_data_with_ui(strategy_file)
    if df_merged is not None and params is not None:
        all_merged_data.append(df_merged)
        all_params.append(params)

if not all_merged_data:
    st.error("No valid data found for selected strategies.")
    st.stop()

# Combinar todos os dados
df_combined = pd.concat(all_merged_data, ignore_index=True)

# Adicionar aviso se nenhuma estratégia tem dados de comparação
if df_combined.empty:
    st.warning("No matched trades found between real and backtest data for any selected strategy.")
    st.info("This could happen if:\n- The real trading data doesn't match backtest timeframes\n- The strategies haven't generated trades yet\n- There's a mismatch in the data files")
    st.stop()
if len(all_params) == 1:
    params = all_params[0]
    st.write(f"### Strategy: {params['strategy']} - {params['symbol']}")
    st.write(f"**Timeframe:** {params['timeframe']} | **Magic:** {params['magic_number']}")
    
    # Mostrar informações adicionais
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'tc' in params:
            st.write(f"**TC:** {params['tc']}")
    with col2:
        if 'lote' in params:
            st.write(f"**Lote:** {params['lote']}")
    with col3:
        if 'daytrade' in params:
            st.write(f"**Daytrade:** {'Yes' if params['daytrade'] else 'No'}")
    
    # Mostrar TP/SL se disponível
    if all_merged_data and 'tp_points' in all_merged_data[0].columns:
        tp = all_merged_data[0]['tp_points'].iloc[0] if not all_merged_data[0].empty else 'N/A'
        sl = all_merged_data[0]['sl_points'].iloc[0] if not all_merged_data[0].empty else 'N/A'
        st.write(f"**TP/SL Points:** {tp}/{sl}")
        
        # Mostrar horas de operação se disponível
        if 'hours' in params:
            st.write(f"**Operating Hours:** {', '.join(map(str, params['hours']))}")
else:
    st.write(f"### Analyzing {len(all_params)} Strategies")
    with st.expander("Strategy Details"):
        for i, params in enumerate(all_params):
            st.write(f"**{i+1}. {params['strategy']}** on {params['symbol']} ({params['timeframe']}) - Magic: {params['magic_number']}")
            
            # Mostrar detalhes adicionais
            details = []
            if 'tc' in params:
                details.append(f"TC: {params['tc']}")
            if 'lote' in params:
                details.append(f"Lote: {params['lote']}")
            if 'hours' in params:
                details.append(f"Hours: {', '.join(map(str, params['hours']))}")
            
            if details:
                st.write(f"   {' | '.join(details)}")
            
            # Mostrar TP/SL se disponível
            if i < len(all_merged_data) and all_merged_data[i] is not None and not all_merged_data[i].empty:
                tp = all_merged_data[i]['tp_points'].iloc[0] if 'tp_points' in all_merged_data[i].columns else 'N/A'
                sl = all_merged_data[i]['sl_points'].iloc[0] if 'sl_points' in all_merged_data[i].columns else 'N/A'
                st.write(f"   TP/SL: {tp}/{sl} points")

#################################
###  1. Overview Metrics  ###
#################################

st.write("## 1. Overview Metrics")

if mode == "Single Strategy":
    # Métricas simples para estratégia única
    col1, col2, col3, col4, col5 = st.columns(5)
    
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
        avg_slippage_exit = df_combined['slippage_exit'].mean()
        st.metric("Avg Exit Slippage", f"{avg_slippage_exit:.1f} pts")
        
    with col5:
        correlation = df_combined['profit'].corr(df_combined['pts_final'] * df_combined['valor_por_ponto'])
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
            'Avg Exit Slippage': df_strategy['slippage_exit'].mean(),
            'Correlation': df_strategy['profit'].corr(df_strategy['pts_final'] * df_strategy['valor_por_ponto'].iloc[0])
        }
        strategy_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(strategy_metrics)
    st.dataframe(metrics_df.style.format({
        'Avg Profit Delta': 'R$ {:.2f}',
        'Avg Entry Slippage': '{:.1f} pts',
        'Avg Exit Slippage': '{:.1f} pts',
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

# Box plots dedicados para Slippage
st.write("### Slippage Distribution Overview")

col1, col2 = st.columns(2)

with col1:
    # Box plot para Entry Slippage
    fig_box_entry = go.Figure()
    
    if mode == "Single Strategy":
        # Box plot por tipo de posição
        for posi_type in ['long', 'short']:
            data = df_combined[df_combined['posi'] == posi_type]['slippage_entry']
            fig_box_entry.add_trace(go.Box(
                y=data,
                name=posi_type.capitalize(),
                boxpoints='outliers',
                marker_color='#2ecc71' if posi_type == 'long' else '#e74c3c'
            ))
    else:
        # Box plot por estratégia
        for strategy in df_combined['strategy_name'].unique():
            data = df_combined[df_combined['strategy_name'] == strategy]['slippage_entry']
            fig_box_entry.add_trace(go.Box(
                y=data,
                name=strategy,
                boxpoints='outliers'
            ))
    
    fig_box_entry.update_layout(
        title="Entry Slippage Distribution",
        yaxis_title="Slippage (points)",
        height=400,
        template="plotly_white",
        showlegend=True
    )
    
    st.plotly_chart(fig_box_entry, use_container_width=True)

with col2:
    # Box plot para Exit Slippage
    fig_box_exit = go.Figure()
    
    if mode == "Single Strategy":
        # Box plot por tipo de posição
        for posi_type in ['long', 'short']:
            data = df_combined[df_combined['posi'] == posi_type]['slippage_exit']
            fig_box_exit.add_trace(go.Box(
                y=data,
                name=posi_type.capitalize(),
                boxpoints='outliers',
                marker_color='#2ecc71' if posi_type == 'long' else '#e74c3c'
            ))
    else:
        # Box plot por estratégia
        for strategy in df_combined['strategy_name'].unique():
            data = df_combined[df_combined['strategy_name'] == strategy]['slippage_exit']
            fig_box_exit.add_trace(go.Box(
                y=data,
                name=strategy,
                boxpoints='outliers'
            ))
    
    fig_box_exit.update_layout(
        title="Exit Slippage Distribution",
        yaxis_title="Slippage (points)",
        height=400,
        template="plotly_white",
        showlegend=True
    )
    
    st.plotly_chart(fig_box_exit, use_container_width=True)


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
    # Criar coluna com profit do backtest em R$
    df_combined['backtest_profit_rs'] = df_combined['pts_final'] * df_combined['valor_por_ponto']
    
    fig_scatter = px.scatter(
        df_combined,
        x='backtest_profit_rs',
        y='profit',
        color='posi',
        title="Real vs Backtest Profits",
        labels={'backtest_profit_rs': 'Backtest Profit (R$)', 'profit': 'Real Profit (R$)'},
        color_discrete_map={'long': '#2ecc71', 'short': '#e74c3c'}
    )
else:
    # Criar coluna com profit do backtest em R$
    df_combined['backtest_profit_rs'] = df_combined['pts_final'] * df_combined['valor_por_ponto']
    
    fig_scatter = px.scatter(
        df_combined,
        x='backtest_profit_rs',
        y='profit',
        color='strategy_name',
        symbol='posi',
        title="Real vs Backtest Profits by Strategy",
        labels={'backtest_profit_rs': 'Backtest Profit (R$)', 'profit': 'Real Profit (R$)'}
    )

# Adicionar linha de referência diagonal
x_range = [df_combined['backtest_profit_rs'].min(), df_combined['backtest_profit_rs'].max()]
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
    valor_medio = df_combined['valor_por_ponto'].iloc[0]
    total_slip_cost = (avg_entry_slip + abs(avg_exit_slip)) * valor_medio * len(df_combined)
    correlation = df_combined['profit'].corr(df_combined['pts_final'] * df_combined['valor_por_ponto'])
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
        lambda x: x['profit'].corr(x['pts_final'] * x['valor_por_ponto'].iloc[0]) if len(x) > 1 else 0
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