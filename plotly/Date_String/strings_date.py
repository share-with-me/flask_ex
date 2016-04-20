import plotly.plotly as py
import plotly.graph_objs as go

data = [
    go.Scatter(
        x=['2013-10-04 22:23:00', '2013-11-04 22:23:00', '2013-12-04 22:23:00'],
        y=[1, 3, 6]
    )
]
plot_url = py.plot(data, filename='date-axes')
