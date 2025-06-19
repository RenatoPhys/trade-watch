#################################
### Importando as bibliotecas ###
#################################

import numpy as np
import datetime as dt
import pandas as pd
import streamlit as st

st.write("# Welcome to the AlgoTrader BR results page! 👋")
st.sidebar.success("Select page above")

st.markdown(
    """
    ## Introduction

    On this page you can find the results of my daily trades in
    **real account**! The idea is that the update is almost *real-time*.

    Currently my **algo trading** systems are specialized in *day trade* for the brazilian mini-index
    (**WINFUT**), where trades are executed **100% automated** using MetaTrader 5.

   👈 Select the page on the side to get the information you want.
    Be it **performance** information (backtest and/or real live trading), **trade statistics** (e.g. profit/loss),
    and also the breakdown of **backtest and real account differences**.

    ### Contact

    - Questions and suggestions, please contact me by email: algotraderbr@gmail.com

"""
)