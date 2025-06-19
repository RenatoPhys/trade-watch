#################################
### Importando as bibliotecas ###
#################################

import numpy as np
import datetime as dt
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

#################################
###       Preâmbulo       ###
#################################

st.write("# Observabilities: Real vs Backtest")

st.write("#### On this page we will dive into some important details of the difference in results \
          from the real account and the backtest ")

st.markdown('''
        In summary, we want to make an accurate diagnosis of how the **trades** of the strategies
        that I use **are different** (or not) from the **backtest** carried out. Such an analysis 
        is important for eventual recalibrations in estimating the "costs" of trades.
        
        I will then do the following analyses:
        1. **Delta Strategy =** how the return of each trade (entry and exit) differs from the backtest
        2. **Entry slippage =** as the entry price (long/short) is different from the backtest price
        3. **Exit Slippage =** as the entry price (long/short) is different from the backtest price
            
        -----
            
        ''')

#################################
###  0. Data Prep  ###
#################################

# Selecting the desired strategy
option_strat = st.selectbox(
    'Which Strategy would you like to select?',
    ('2-Surfing the trend', '3-Iron box', '4-gold 13','5-gold 1011'))

st.write('You selected:', option_strat)

# this parameter will be used to import some tables
strat = option_strat.split('-')[0]

### Importando json com parametros de data inicial - data final
df_params = pd.read_json(f'params_strat{strat}.json')
data_ini = df_params['data_ini'][0]
data_fim = df_params['data_fim'][0]

# Importando dados de trades reais
dfmt5_2 = pd.read_csv(f'bases/dados_real_mt5_strat{strat}.csv', index_col=['time'], parse_dates=['time'])

# Importando dados teóricos 
dft = pd.read_csv(f'bases/backtest_deploy_strat{strat}.csv', index_col=['time'], parse_dates=['time'])
# Selecionando apenas entradas
dft = dft[dft['strategy_2']!=0]
dft.columns = dft.columns + '_teo'
dft.rename(columns = {'safra_teo': 'safra'}, inplace = True)


# Concat para união do real e teórico
df = pd.concat([dfmt5_2[['price_ent','price_ext','lucro','cstrategy_2','comment']], dft.iloc[:,2:]], axis =1)
#df = pd.merge(df, dft[['open_teo','close_teo','position_teo']], left_index=True, right_index=True)
df.loc[df.position_teo==1, 'posi'] = 'long'
df.loc[df.position_teo==-1, 'posi'] = 'short'
df = df[~df['posi'].isnull()]

# Calculando slippages
df['slippage_ent'] = df['position_teo']*(df['price_ent'] - df['close_teo'])
df['slippage_ext'] = (df['price_ext'] - df['position_teo']*df['pts_final_teo'] - df['close_teo'])
df['dif_strat'] = df['lucro'] - df['strategy_2_teo']
df['hit_alvo'] = 0
# Vendo se acertamos o sl ou tp
df.loc[df['comment'].str.contains('\[', na=False), 'hit_alvo'] = 1


#################################
###  1. Delta Strategy ###
#################################

# Observando a diferença no resultado dos trades
figdelstr = px.box(df, y="dif_strat", color='posi', points="all", color_discrete_sequence=['red','green'])

# adicionando elementos de layout
figdelstr.update_layout(
    title = dict(text="1. Delta Strategy per trade", font=dict(size=27), automargin=False, yref='paper'),
    xaxis_title= dict(text="<b> Hit </b>", font=dict(size=20)),
    yaxis_title= dict(text="<b>Difference (R$) </b>", font=dict(size=20)),
    font_family="Arial",
    font_color="black",
    title_font_family="Arial",
    title_font_color="black",
    legend_title_font_color="green",
    showlegend=True,
    autosize=False,
    width=800,
    height=500,
    
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=15,
            color='black',
        ),
    ),
    yaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=15,
            color='black',
        ),
    )
)

# Plot!
figdelstr.update_layout(legend_title_text='Position')
st.plotly_chart(figdelstr, use_container_width=True)

st.write('The mean and median being above or equal to 0 indicates that we are **not** underestimating trading costs.\
          This is a good sign because it gives us confidence in the results of the backtest that was carried out')

#################################
###  2. slippage entrada/saida  ###
#################################

# Observando a diferença no resultado dos trades - entries
figslp = px.box(df, y="slippage_ent", color='posi', points="all", color_discrete_sequence=['red','green'])

# adicionando elementos de layout
figslp.update_layout(
    title = dict(text="2. Slippage - entries", font=dict(size=27), automargin=False, yref='paper'),
    xaxis_title= dict(text="<b> Hit </b>", font=dict(size=20)),
    yaxis_title= dict(text="<b> Slippage (WINFUT points) </b>", font=dict(size=20)),
    font_family="Arial",
    font_color="black",
    title_font_family="Arial",
    title_font_color="black",
    legend_title_font_color="green",
    showlegend=True,
    autosize=False,
    width=800,
    height=500,
    
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=15,
            color='black',
        ),
    ),
    yaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=15,
            color='black',
        ),
    )
)

# Plot!
figslp.update_layout(legend_title_text='Position')
st.plotly_chart(figslp, use_container_width=True)


# Observando a diferença no resultado dos trades - exits
figslp = px.box(df, y="slippage_ext", color='posi', points="all", color_discrete_sequence=['red','green'])

# adicionando elementos de layout
figslp.update_layout(
    title = dict(text="2. Slippage - exits", font=dict(size=27), automargin=False, yref='paper'),
    xaxis_title= dict(text="<b> Hit </b>", font=dict(size=20)),
    yaxis_title= dict(text="<b> Slippage (WINFUT points) </b>", font=dict(size=20)),
    font_family="Arial",
    font_color="black",
    title_font_family="Arial",
    title_font_color="black",
    legend_title_font_color="green",
    showlegend=True,
    autosize=False,
    width=800,
    height=500,
    
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=15,
            color='black',
        ),
    ),
    yaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='white',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=15,
            color='black',
        ),
    )
)

# Plot!
figslp.update_layout(legend_title_text='Position')
st.plotly_chart(figslp, use_container_width=True)

st.write('The slippage...')