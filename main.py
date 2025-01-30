import datetime
import numpy as np
import scipy
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from data_collection_Vnstock import load_stock_data, load_index_data
from classical_strategies import calc_performance_metrics_subset, calc_sharpe_by_year,winTrade_prob
from momentum_strategies import TSMOM_strategy
import streamlit as st

st.set_page_config(layout="wide")


def main():
    st.subheader(f'1. TSMOM result')
    start = '2018-01-01'
    # end = '2022-12-05'
    now = datetime.datetime.now()
    end = now.strftime("%Y-%m-%d")
    @st.cache_data
    def data_stock():
        stock = load_stock_data(symbol,start,end)
        stock.index = pd.to_datetime(stock.index)
        index = load_index_data('VNINDEX', start, end)
        index.index = pd.to_datetime(index.index)
        
        # Ensure indices are unique
        stock = stock[~stock.index.duplicated(keep='first')]
        index = index[~index.index.duplicated(keep='first')]

        return stock, index
    stock,index = data_stock()
    # index = load_index_data('VNINDEX',start,end)
    # index.index = pd.to_datetime(index.index)

    df = pd.DataFrame(stock['Close'].copy())
    df.reset_index(inplace=True)
    df.rename(columns={"time":'Date'},inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date',inplace=True)

    index = index.loc[df.index[0]:]

    fig = px.line(df,title=f"{symbol} stock price")
    fig.update_layout(
        autosize=False,
        width=1000,
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig,use_container_width=True)


    TSMOM_df = pd.DataFrame(columns=['Buy and Hold','TSMOM without vol target','TSMOM with vol target','VNINDEX'])
    return_df = pd.DataFrame(columns=['Buy and Hold', 'TSMOM without vol target', 'TSMOM with vol target', 'VNINDEX'])
    return_df['Buy and Hold'] = stock.pct_return
    return_df['VNINDEX'] = index.pct_return


    TSMOM = TSMOM_strategy(df, VOL_LOOKBACK, VOL_TARGET, volatility_scaling=False)
    trend = TSMOM.trend_estimation(TS_LENGTH)
    signal = TSMOM.position_sizing(trend, activation='sign')
    returns = TSMOM.cal_strategy_returns(signal)
    returns.rename(columns={'Close': 'returns'}, inplace=True)
    my_rets = returns.dropna()
    my_rets.rename(columns={'returns': 'captured_returns'}, inplace=True)
    return_df['TSMOM without vol target'] =my_rets.captured_returns

    TSMOM = TSMOM_strategy(df, VOL_LOOKBACK, VOL_TARGET, volatility_scaling=True)
    returns1 = TSMOM.cal_strategy_returns(signal)
    returns1.rename(columns={'Close': 'returns'}, inplace=True)
    vol_target_map = TSMOM.volatility_target_map()
    my_rets1 = returns1.dropna()
    my_rets1.rename(columns={'returns': 'captured_returns'}, inplace=True)
    return_df['TSMOM with vol target'] =my_rets1.captured_returns
    return_df.iloc[0,:]=0
    return_df.dropna(inplace=True)

    TSMOM_df['Buy and Hold'] = ((1 + return_df['Buy and Hold']).cumprod() - 1)
    TSMOM_df['VNINDEX'] = ((1 + return_df['VNINDEX']).cumprod() - 1)
    TSMOM_df['TSMOM with vol target'] = ((1 + return_df['TSMOM with vol target']).cumprod() - 1)
    TSMOM_df['TSMOM without vol target'] = ((1 + return_df['TSMOM without vol target']).cumprod() - 1)

    fig = px.line(TSMOM_df,title="Cumulative return comparison")
    fig.update_layout(
        autosize=False,
        width=705,
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig,use_container_width=True)


    # fig = go.Figure()
    signal.rename(columns={'Close':'signal'},inplace=True)
    signal = signal[-250:]
    fig = px.line(signal,title='250 days signal for TSMOM')
    # Set title
    fig.update_layout(
        autosize=False,
        width=705,
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig,use_container_width=True)

    fig = px.line(vol_target_map, title='Volatility target map')
    # Set title
    fig.update_layout(
        autosize=False,
        width=705,
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig,use_container_width=True)

    ts1 = stock.pct_return
    ## ts2: return of TSMOM with vol target
    ts2 = my_rets1.captured_returns
    ## ts3: return of TSMOM without vol target
    ts3 = my_rets.captured_returns

    statistics1 = scipy.stats.describe(ts1)
    statistics2 = scipy.stats.describe(ts2)
    statistics3 = scipy.stats.describe(ts3)

    stats = pd.DataFrame(index=['mean', 'std', 'skewness', 'kurtosis'],
                         columns=['normal return', 'TSMOM vol target', 'TSMOM no vol target'])
    stats.loc['mean', :] = round(statistics1.mean, 4), round(statistics2.mean, 4), round(statistics3.mean, 4)
    stats.loc['std', :] = round(np.sqrt(statistics1.variance), 4), round(np.sqrt(statistics2.variance), 4), round(
        np.sqrt(statistics3.variance), 4)
    stats.loc['skewness', :] = round(statistics1.skewness, 4), round(statistics2.skewness, 4), round(
        statistics3.skewness, 4)
    stats.loc['kurtosis', :] = round(statistics1.kurtosis, 4), round(statistics2.kurtosis, 4), round(
        statistics3.kurtosis, 4)

    st.write('### Performance metrics')
    ## TSMOM
    df_ret_TSMOM_vol_target = pd.DataFrame(calc_performance_metrics_subset(my_rets1.captured_returns),
                                           index=['TSMOM with vol target'])
    df_ret_TSMOM_no_vol_target = pd.DataFrame(calc_performance_metrics_subset(my_rets.captured_returns),
                                              index=['TSMOM no vol target'])
    df_ret_long_only = pd.DataFrame(calc_performance_metrics_subset(stock.pct_return), index=['buy-and-hold'])

    df_comparison = pd.concat((df_ret_TSMOM_vol_target, df_ret_TSMOM_no_vol_target))
    df_comparison = pd.concat((df_comparison, df_ret_long_only))
    df_comparison = df_comparison.applymap("{0:.2%}".format)
    st.table(df_comparison.T)

    sharpe_df_TSMOM_vol_target = pd.DataFrame(calc_sharpe_by_year(my_rets1[['captured_returns']]),
                                              index=['TSMOM with vol target'])
    sharpe_df_TSMOM_no_vol_target = pd.DataFrame(calc_sharpe_by_year(my_rets[['captured_returns']]),
                                                 index=['TSMOM no vol target'])

    long_only = stock[['pct_return']]
    long_only.rename(columns={"pct_return": "captured_returns"}, inplace=True)

    sharpe_df_long_only = pd.DataFrame(long_only.captured_returns)
    sharpe_df_long_only = pd.DataFrame(calc_sharpe_by_year(sharpe_df_long_only), index=['buy-and-hold'])

    sharpe_df = pd.concat((sharpe_df_TSMOM_vol_target, sharpe_df_TSMOM_no_vol_target))
    sharpe_df = pd.concat((sharpe_df, sharpe_df_long_only))
    sharpe_df.dropna(inplace=True,axis=1)
    st.table(sharpe_df.T)
    st.write('### Daily return distribution')
    st.table(stats)

    y1 = ts1
    y2 = ts2
    y3 = ts3
    x = ts3.index

    colors = ["#e2e2e2", "#e1a692", "#1984c5"]

    fig = make_subplots(
        rows=3, cols=2,
        column_widths=[0.55, 0.45],
        row_heights=[1., 1., 1.],
        specs=[[{"type": "scatter"}, {"type": "xy"}],
               [{"type": "scatter"}, {"type": "xy", "rowspan": 2}],
               [{"type": "scatter"}, None]])

    fig.add_trace(
        go.Scatter(x=x,
                   y=y1,
                   hoverinfo='x+y',
                   mode='lines',
                   line=dict(color=colors[0],
                             width=1),
                   showlegend=False,
                   ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=x,
                   y=y2,
                   hoverinfo='x+y',
                   mode='lines',
                   line=dict(color=colors[1],
                             width=1),
                   showlegend=False,
                   ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=x,
                   y=y3,
                   hoverinfo='x+y',
                   mode='lines',
                   line=dict(color=colors[2],
                             width=1),
                   showlegend=False,
                   ),
        row=3, col=1
    )

    boxfig = go.Figure(data=[go.Box(x=y1, showlegend=False, notched=True, marker_color=colors[0], name='3'),
                             go.Box(x=y2, showlegend=False, notched=True, marker_color=colors[1], name='2'),
                             go.Box(x=y3, showlegend=False, notched=True, marker_color=colors[2], name='1')])

    for k in range(len(boxfig.data)):
        fig.add_trace(boxfig.data[k], row=1, col=2)

    group_labels = ['Buy and hold', 'TSMOM with vol target', 'TSMOM without vol target']
    hist_data = [y1, y2, y3]

    distplfig = ff.create_distplot(hist_data, group_labels, colors=colors,
                                   bin_size=0.01, show_rug=False)

    for k in range(len(distplfig.data)):
        fig.add_trace(distplfig.data[k],
                      row=2, col=2
                      )
    fig.update_layout(barmode='overlay',
                      width=1000,
                      height=800,
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=1.02,
                          xanchor="right",
                          x=1
                      )
                      )
    st.plotly_chart(fig, use_container_width=True)



    st.write('### Win rate (Cornish–Fisher expansion)')
    st.write(f'**{symbol}** || TSMOM win trades probability with vol target:',
          "**{:.2%}**".format(winTrade_prob(statistics2.skewness)))
    st.write(f'**{symbol}** || TSMOM win trades probability without vol target:',
          "**{:.2%}**".format(winTrade_prob(statistics3.skewness)))




def methodology():
    st.subheader('2. Methodology')
    st.write('#### 2.1 Strategy')
    st.latex(r'''
        r^{TSMOM}_{t,t+1} = \frac{1}{N} \sum X_t^{(i)}k r^{(i)}_{t,t+1} \;\;\;\; \text{(1)}
        ''')
    st.write('With:')
    st.latex(r'''Y^{(i)}_t = r^{(i)}_{t−252,t} \;\;\text{(2)}''')
    st.latex(r'''X^{(i)}_t = sgn(Y^{(i)}_t) \;\;\text{(3)}''')
    st.latex(r'''\text{Long only:} \; X^{(i)}_t = max(0,sgn(Y^{(i)}_t)) \;\;\;\text{(4)}''')
    st.latex(r''' k=
    \begin{cases}
      \frac{\sigma_{tgt}}{\sigma_t^{(i)}} & \text{if      }\frac{\sigma_{tgt}}{\sigma_t^{(i)}}\leqslant2\\
      2 & \text{if }\frac{\sigma_{tgt}}{\sigma_t^{(i)}}>2\\
    \end{cases}    \;\;\;\text{(5)}''')
    st.write("## ")
    st.write('#### 2.2 Win rate (Cornish–Fisher expansion)')
    st.latex(r'''P(Y>E[Y]) \approx \frac{1}{2} - \frac{k_3}{6\sqrt{2\pi}} \;\;\;\text{(6)}''')

    st.write("#### 2.3 Analysis")
    st.write('''
    Trend-following, or momentum, strategies have the attractive property of generating trading returns 
    with a positively skewed statistical distribution. 
    Consequently, they tend to hold on to their profits and are unlikely to have severe ‘drawdowns’.
    They are very scalable and are employed in most asset classes.
    The basic premise behind momentum is to buy what has been going up recently, 
    and sell what has been going down. In other words, if recent returns have been positive then 
    future ones are more likely to be positive, 
    and similarly with negative. 
    Systematic strategies formalise this notion by 
    (i) measuring momentum, essentially by smoothing out recent returns to obtain a signal that is not too rapidly-varying, and 
    (ii) having a law that turns this signal into a trading position, i.e. how may contracts or what notional to have on.
    
    One of the advantage of this strategy is that, with an appropriate stock and parameters, we can achieve higher annual return
    and lower annual volatility. However, the strategy is not a magical tools and not all stocks achieve the same result.
    Hence, choosing the right stock and the right paremeter is crucial for a successful momentum strategy.''')



def references():
    st.subheader('3. References')
    st.write("""
    Moskowitz, TJ, Ooi, YH & Pedersen, LH 2012, ‘Time series momentum’, Journal of Financial Economics, vol. 104, no. 2, pp. 228–250, <https://www.sciencedirect.com/science/article/pii/S0304405X11002613>.
    """)
    st.write("""
    Martin, R 2023, 'Design and analysis of momentum trading strategies', <https://arxiv.org/pdf/2101.01006.pdf>
    """)

if __name__ =='__main__':

    # def run_program():
    # st.title('TSMOM for Vietnam Stock market')
    st.markdown("<h1 style='text-align: center;'>TSMOM for Vietnam Stock market</h1>", unsafe_allow_html=True)
    st.markdown('##')

    expander = st.expander("How to use this")
    expander.write(
    '''#### Change symbol and adjust parameters in the sidebar\n
    1.Stock symbol
    2.Look back for TSMOM: window for calculating cumulative return
    3.Look back for volatility target: window for ex ante volatility (EWM)
    4.Volatility target: between (0,1) it can be consider to be the risk appetite
    ''')
    expander.write('''#### Description
    This is the daily signal for old school time series momentum (TSMOM), which used the 
    methodology from Moskowitz et al. (2012), and apply in Vietnamese stock market. 
    There are 2 key elements of a TSMOM strategy: (i) trend estimation, (ii) psition sizing. 
    - Trend estimation will use the cumulative return in an determined period
    - Position sizing: will use sign signal (however, we can short in Vietnam, we will skip the -1 signal)
    ''')
    expander.write("""
    Note: **The model will fixed the start date at 2018-01-01 due to lack of data before that period. End date will be today.**""")
    symbol = st.text_input('Enter stock symbol:').upper()
    isValidSymbol = False
    stock_list = pd.read_csv(r'C:\Users\LNV\OneDrive\Minh Triết Đầu Tư\Vietnam_TSMOM\upload folder\StockList.csv')
    stock_list = stock_list['Ticker'].to_list()
    # st.write(stock_list)
    for stock in stock_list:
        if (symbol == stock):
             isValidSymbol = True

    with st.sidebar:
        # symbol = st.text_input('Enter stock symbol:').upper()
        TS_LENGTH = st.number_input('Look back for TSMOM:', min_value=0, value=21)
        VOL_LOOKBACK = st.number_input('Look back for volatility target:', min_value=0, value=60)
        VOL_TARGET = st.number_input('Volatility target:', min_value=0.00, value=0.35)
    st.markdown(f'''
        <style>
        section[data-testid="stSidebar"] .css-ng1t4o {{width: 14rem;}}
        </style>
    ''', unsafe_allow_html=True)

    if symbol and isValidSymbol: #symbol true
        main()
    if symbol and not isValidSymbol:
        st.write('## Invalid Stock')
    else:
        st.write('## Please choose a stock')


    st.write('''------------''')
    methodology()
    st.write('''------------''')
    references()


# run_program()