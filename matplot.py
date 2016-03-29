from flask import Flask, make_response
import datetime
import StringIO
import random

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
app = Flask(__name__)

@app.route("/")
def index():
    fig=Figure()
    ax=fig.add_subplot(1,1,1)
    x=[] #Empty x array
    y=[] #Empty y array
    now=datetime.datetime.now() #Current date
    delta=datetime.timedelta(days=1) #difference of 1 day
    for i in range(10): #10 x values
        x.append(now) #Append to the x array
        now+=delta
        y.append(random.randint(0, 1000)) #Randomised y values
    ax.plot_date(x, y, '-')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d')) #Format of date
    fig.autofmt_xdate()
    canvas=FigureCanvas(fig)
    output = StringIO.StringIO() #Stored via StringIO.StringIO(), refer documentation
    canvas.print_png(output)
    response=make_response(output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response


if __name__ == "__main__":
    app.run(debug = True,
        host = '127.0.0.1',
        port = int(8080)
        )