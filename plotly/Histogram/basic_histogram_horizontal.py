import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np
y = np.random.randn(500)

data = [
    go.Histogram(
        y=y
    )
]
plot_url = py.plot(data, filename='horizontal-histogram')