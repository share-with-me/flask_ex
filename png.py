import random
import StringIO

from flask import Flask, make_response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


app = Flask(__name__)


@app.route('/')
def plot():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1) 

    xs = range(100) #Range 100
    ys = [random.randint(1, 50) for x in xs] #Generate random data between 1 to 50 for 100 points

    axis.plot(xs, ys) #Plot
    canvas = FigureCanvas(fig) 
    output = StringIO.StringIO() #refer documentation for the use of StringIO
    canvas.print_png(output) #Print
    response = make_response(output.getvalue())
    response.mimetype = 'image/png' #type
    return response

if __name__ == '__main__':
    app.run(debug = True,
        host = '127.0.0.1',
        port = int(8080)
        )