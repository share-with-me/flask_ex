from datetime import datetime
import pandas.io.data as web
import plotly.plotly as py
import plotly.graph_objs as go

df = web.DataReader("aapl", 'yahoo',
                    datetime(2007, 10, 1),
                    datetime(2009, 4, 1))


trace = go.Scatter(x=df.index,
                   y=df.High)

data = [trace]

layout = dict(
    title='Time series with range slider and selectors',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                    label='YTD',
                    step='year',
                    stepmode='todate'),
                dict(count=1,
                    label='1y',
                    step='year',
                    stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    )
)

fig = dict(data=data, layout=layout)
plot_url = py.plot(fig, filename='Range')


