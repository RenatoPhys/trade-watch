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

# Selecting the desired strategy
option_strat = st.selectbox(
    'Which Strategy would you like to select?',
    ('0-All','2-Surfing the trend', '3-Iron box', '4-gold 13','5-gold 1011'))

st.write('You selected:', option_strat)

# this parameter will be used to import some tables
strat = option_strat.split('-')[0]

# Visualizando estratégias individuais
if strat !='0':
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

else:
    # Importando todas as estratégias
    df = pd.DataFrame()
    for i in [2,3,4,5]:
        dftmp = pd.read_csv(f'bases/dados_real_mt5_strat{i}.csv', index_col=['time'], parse_dates=['time'])
        dftmp.rename(columns={'cstrategy_2':f'cstrategy_{i}'}, inplace=True)
        dftmp = dftmp[[f'cstrategy_{i}']]
        dftmp = dftmp.resample('d').last().ffill()
        df = pd.concat([df, dftmp], axis = 1)

    # estratégia acumulada
    df['cstrategy'] = df.sum(axis=1)

    # Cálculo do drawdown
    df['cummax'] = df['cstrategy'].cummax()
    df["dd"] = df['cstrategy'] - df['cummax']


#################################
###  1. Performance Acumulada  ###
#################################

## Plotando resultados
if strat !='0':
    figret = go.Figure()
    figret.add_trace(go.Scatter(
        x=dfmt5_2.index,
        y=dfmt5_2["cstrategy_2"],
        name='Real'
    ))
    figret.add_trace(go.Scatter(
        x=dft.index,
        y=dft["cstrategy_2_teo"],
        name='Backtest',
        line=dict(color="#d62728")
    ))

    # adicionando elementos de layout
    figret.update_layout(
        title = dict(text="1. Cumulative Return", font=dict(size=27), automargin=False, yref='paper'),
        xaxis_title= dict(text="<b> Data </b>", font=dict(size=20)),
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
    figret.update_layout(legend_title_text='Legend')
    st.plotly_chart(figret, use_container_width=True)

else:
    
    # Plot returns -- all together
    figret2 = go.Figure()
    figret2.add_trace(go.Scatter(
        x=df.index,
        y=df[f"cstrategy"],
        name=f'Real'
    ))


    # adicionando elementos de layout
    figret2.update_layout(
        title = dict(text="1. Cumulative Return - All together", font=dict(size=27), automargin=False, yref='paper'),
        xaxis_title= dict(text="<b> Data </b>", font=dict(size=20)),
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
    figret2.update_layout(legend_title_text='Legend')
    st.plotly_chart(figret2, use_container_width=True)

    #### Plot Returns by strategy
    figret = go.Figure()
    for i in [2,3,4,5]:
        figret.add_trace(go.Scatter(
            x=df.index,
            y=df[f"cstrategy_{i}"],
            name=f'Real_{i}'
        ))


    # adicionando elementos de layout
    figret.update_layout(
        title = dict(text="1. Cumulative Return - by Strategy", font=dict(size=27), automargin=False, yref='paper'),
        xaxis_title= dict(text="<b> Data </b>", font=dict(size=20)),
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
    figret.update_layout(legend_title_text='Legend')
    st.plotly_chart(figret, use_container_width=True)

st.write('As we can see above, the algorithm running on a real account adheres to the backtest results')


#################################
###  2. Indices risco/retorno  ###
#################################


#################################
###        3. Drawdown        ###
#################################

if strat !='0':
    # Cálculo do drawdown
    dfmt5_2['cummax'] = dfmt5_2['cstrategy_2'].cummax()
    dfmt5_2["dd"] = dfmt5_2['cstrategy_2'] - dfmt5_2['cummax']
    dfmt5_2["max_dd"] = dfmt5_2["dd"].cummin()

    # Plotando
    #figdd = px.line(dfmt5_2, x=dfmt5_2.index, y=["dd"])

    figdd = go.Figure()
    figdd.add_trace(go.Scatter(
        x=dfmt5_2.index,
        y=dfmt5_2["dd"],
        name='Real',
        fill='tozeroy'
    ))

    figdd.update_layout(
        title = dict(text="3. Drawdown", font=dict(size=27), automargin=False, yref='paper'),
        xaxis_title= dict(text="<b> Data </b>", font=dict(size=20)),
        yaxis_title= dict(text="<b>Drawdown (R$) </b>", font=dict(size=20)),
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
    st.plotly_chart(figdd, use_container_width=True)

else:
    figdd = go.Figure()
    figdd.add_trace(go.Scatter(
        x=df.index,
        y=df["dd"],
        name='Real',
        fill='tozeroy'
    ))

    figdd.update_layout(
        title = dict(text="3. Drawdown", font=dict(size=27), automargin=False, yref='paper'),
        xaxis_title= dict(text="<b> Data </b>", font=dict(size=20)),
        yaxis_title= dict(text="<b>Drawdown (R$) </b>", font=dict(size=20)),
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
    st.plotly_chart(figdd, use_container_width=True)


st.write('We can observe a controlled drawdown without many surprises.')

#################################
###  3. Performance Mensal  ###
#################################

if strat !='0':
    # Concat para união do real e teórico
    df = pd.concat([dfmt5_2[['lucro','cstrategy_2','comment']], dft.iloc[:,2:]], axis =1)

    # Dataframe auxiliar para o plot
    aux = df.resample('MS').sum()
    aux = aux[aux['lucro']!=0]
    aux.rename(columns={'lucro':'Real', 'strategy_2_teo': 'Backtest'}, inplace = True)

    # Plot
    figd = px.bar(aux, x=aux.index, y=["Real",'Backtest'], barmode='group',
                color_discrete_sequence=['#1f77b4','#d62728'])

    # adicionando elementos de layout
    figd.update_layout(
        title = dict(text="4. Monthly Return", font=dict(size=27), automargin=False, yref='paper'),
        xaxis_title= dict(text="<b> Data </b>", font=dict(size=20)),
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
    st.plotly_chart(figd, use_container_width=True)

st.write('Again, we can see the good result agreement between the real account result and the \
          expected backtest result.')

#################################
###  4. Trades vencedeores/perdedores  ###
#################################