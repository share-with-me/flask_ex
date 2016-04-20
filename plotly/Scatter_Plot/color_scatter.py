import plotly.graph_objs as go
import plotly.plotly as py

import numpy as np

trace = go.Scatter(
    y = np.random.randn(1000),
    mode='markers',
    marker=dict(
        size='16',
        color = np.random.randn(1000), #set color equal to a variable
        colorscale='Viridis',
        showscale=True
    )
)
data = [trace]

plot_url = py.plot(data, filename='color-scatter')