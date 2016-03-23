import json
from flask import Flask, render_template
import numpy as np 

app = Flask(__name__)

@app.route('/')

def index():
    return render_template("circles-d3.html")

@app.route("/data")
@app.route("/data/<int:ndata>")

#Write a function under this path that generates random number of data points in json format which is used by the html page as the source for data

def data(ndata = 50):
    x = 10*np.random.randn(ndata)
    y = 0.3*x + 2 * np.random.randn(ndata)
    A = 10. ** np.random.rand(ndata)
    c = np.random.rand(ndata)

    #return the generated list in the form of jsonify dump and store it the /data where it can be used by html page to render the visualisation
    return json.dumps([{"_id": i, "x": x[i], "y": y[i], "area": A[i], "color": c[i]}
            for i in range(ndata)])


if __name__ == "__main__":
    app.run(
        debug = True,
        port = int(80),
        host = "127.0.0.1"
    )
