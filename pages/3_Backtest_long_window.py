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

st.write("# Backtests")

st.write("#### On this page I detail how the backtests are performing in the long run")

st.markdown('''
        
        Before deploying an *algo trading* strategy on a real account, it is necessary to study in detail
        the backtest result for a relatively long period of time.
                        
        -----
            
        ''')

#################################
###  0. Data Prep  ###
#################################

# Selecting the desired strategy
option_strat = st.selectbox(
    'Which Strategy would you like to select?',
    ('0-All','2-Surfing the trend', '3-Iron box', '4-gold 13','5-gold 1011'))

st.write('You selected:', option_strat)

# this parameter will be used to import some tables
strat = option_strat.split('-')[0]

# Importando backtest

# Visualizando estratégias individuais
if strat !='0':
    df = pd.read_csv(f'bases/backtest_full_strat{strat}.csv', index_col=['time'], parse_dates=['time'])
    df = df.dropna()

# Visualizando resultado consolidado
else:
    df = pd.DataFrame()
    for i in [2,3,4,5]:
        dftmp = pd.read_csv(f'bases/backtest_full_strat{i}.csv', index_col=['time'], parse_dates=['time'])
        dftmp.rename(columns={'cstrategy':f'cstrategy_{i}'}, inplace=True)
        dftmp.drop(columns=['dd'], inplace=True)
        dftmp = dftmp.resample('d').last().ffill()
        df = pd.concat([df, dftmp], axis = 1)

    # estratégia acumulada
    df['cstrategy'] = df.sum(axis=1)

    # Cálculo do drawdown
    df['cummax'] = df['cstrategy'].cummax()
    df["dd"] = df['cstrategy'] - df['cummax']




#################################
###  1. Retorno Acumulado  ###
#################################


# Ploting all together

figback = go.Figure()
figback.add_trace(go.Scatter(
    x=df.index,
    y=df["cstrategy"],
    name='Backtest'
))

# adicionando elementos de layout
figback.update_layout(
    title = dict(text="1. Cumulative Return - all together", font=dict(size=27), automargin=False, yref='paper'),
    xaxis_title= dict(text="<b> Date </b>", font=dict(size=20)),
    yaxis_title= dict(text="<b>Return (R$) </b>", font=dict(size=20)),
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
figback.update_layout(legend_title_text='Legend')
st.plotly_chart(figback, use_container_width=True)



## Separating the strategies

if strat == '0':

    figback2 = go.Figure()
    for i in [2,3,4,5]:
        figback2.add_trace(go.Scatter(
            x=df.index,
            y=df[f"cstrategy_{i}"],
            name=f'Strat_{i}'
        ))


    # adicionando elementos de layout
    figback2.update_layout(
        title = dict(text="1. Cumulative Return - by strategy", font=dict(size=27), automargin=False, yref='paper'),
        xaxis_title= dict(text="<b> Date </b>", font=dict(size=20)),
        yaxis_title= dict(text="<b>Return (R$) </b>", font=dict(size=20)),
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
    figback2.update_layout(legend_title_text='Legend')
    st.plotly_chart(figback2, use_container_width=True)



#################################
###  2. Drawdown  ###
#################################


figback = go.Figure()
figback.add_trace(go.Scatter(
    x=df.index,
    y=df["dd"],
    name='Backtest',
    fill='tozeroy'
))

# adicionando elementos de layout
figback.update_layout(
    title = dict(text="2. Drawdown", font=dict(size=27), automargin=False, yref='paper'),
    xaxis_title= dict(text="<b> Date </b>", font=dict(size=20)),
    yaxis_title= dict(text="<b> DD (R$) </b>", font=dict(size=20)),
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
figback.update_layout(legend_title_text='Legend')
st.plotly_chart(figback, use_container_width=True)


#################################
###      3. Correlation    ###
#################################


if strat == '0':

    st.markdown('''
        
        Measuring weekly return correlations between strategies
            
        ''')
    

    df_ret = df.resample('w').first() - df.resample('w').last()
    corr = df_ret.iloc[:, :-3].corr()
    
    figcorr = px.imshow(corr, text_auto=True)

    # adicionando elementos de layout
    figcorr.update_layout(
        title = dict(text="3. Correlation between strats", font=dict(size=27), automargin=False, yref='paper'),
        xaxis_title= dict(text="<b> Strategies </b>", font=dict(size=20)),
        yaxis_title= dict(text="<b> Strategies </b>", font=dict(size=20)),
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

    st.plotly_chart(figcorr, use_container_width=True)