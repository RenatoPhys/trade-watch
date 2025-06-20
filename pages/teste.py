import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import json
from datetime import datetime

# Carregar configuração da estratégia
with open('combined_strategy_teste.json', 'r') as f:
    strategy_params = json.load(f)

symbol = strategy_params['symbol']
timeframe = strategy_params['timeframe']
strategy_name = strategy_params['strategy']
magic_number = strategy_params['magic_number']

# Carregar dados de backtest completo
df = pd.read_csv(f'bases/full_backtest_{symbol}_{timeframe}_{strategy_name}_magic_{magic_number}.csv', 
                 index_col=['time'], parse_dates=['time'])

# Ajuste
df['cstrategy'] = df['cstrategy'].ffill()

print(df.head())