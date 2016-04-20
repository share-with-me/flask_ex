import plotly.plotly as py
import plotly.graph_objs as go

fig = {
    'data': [{'labels': ['Residential', 'Non-Residential', 'Utility'],
              'values': [45, 16, 23],
              'type': 'pie'}],
    'layout': {'title': 'Market Segment'}
}

url = py.plot(fig, filename='Pie Chart Example')