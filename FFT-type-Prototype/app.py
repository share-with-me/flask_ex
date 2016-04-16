from model import *
from flask import Flask, render_template, request
from compute import *
import sys

app = Flask(__name__)

from flask_bootstrap import Bootstrap
Bootstrap(app)

@app.route('/')
def index():
	return render_template('view.html')


@app.route('/Convolution', methods=['GET', 'POST'])
def one():
    result = compute_Convolution()

    return render_template('view.html', result=result)


@app.route('/FFT', methods=['GET', 'POST'])
def two():
    result = compute_FFT()

    return render_template('view.html',result=result)


@app.route('/Gaussian', methods=['GET', 'POST'])
def ND():
    result = compute_Gaussian()

    return render_template('view.html',result=result)



@app.route('/Kernel', methods=['GET', 'POST'])
def DCT():
    result = compute_Kernel()

    return render_template('view.html', result=result)


@app.route('/AF', methods=['GET', 'POST'])
def IDCT():
    result = compute_AF()

    return render_template('view.html', result=result)

@app.route('/Wavelet', methods=['GET', 'POST'])
def DST():
    result = compute_Wavelet()
    return render_template('view.html', result=result)


@app.route('/Sampling', methods=['GET', 'POST'])
def IDST():
    result = compute_Sampling()

    return render_template('view.html', result=result)

@app.route('/Power', methods=['GET', 'POST'])
def Damping():
    result = compute_Power()
    return render_template('view.html', result=result)

@app.route('/Chirp', methods=['GET', 'POST'])
def formula():
    result = compute_Chirp()
    return render_template('view.html', result=result)

@app.route('/Weiner', methods=['GET', 'POST'])
def formulaa():
    result = compute_Weiner()
    return render_template('view.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
