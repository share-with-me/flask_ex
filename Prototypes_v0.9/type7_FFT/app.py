from model import *
from flask import Flask, render_template, request
from compute import *
import sys

app = Flask(__name__)

from flask_bootstrap import Bootstrap
Bootstrap(app)



@app.route('/1D', methods=['GET', 'POST'])
def one():
    form = InputForm_1D(request.form)
    if form.validate():
        result = compute_1D(form.Fsr.data, form.freq.data)
    else:
        result = None

    return render_template('view.html',form=form, result=result)


@app.route('/2D', methods=['GET', 'POST'])
def two():
    form = InputForm_2D(request.form)
    if form.validate():
        result = compute_2D(form.N.data)
    else:
        result = None

    return render_template('view.html',form=form, result=result)


@app.route('/ND', methods=['GET', 'POST'])
def ND():
    form = InputForm_ND(request.form)
    if form.validate():
        result = compute_ND(form.N.data)
    else:
        result = None

    return render_template('view.html',form=form, result=result)



@app.route('/DCT', methods=['GET', 'POST'])
def DCT():
    form = InputForm_DCT(request.form)
    if form.validate():
        result = compute_DCT(form.N.data)
    else:
        result = None

    return render_template('view.html',form=form, result=result)


@app.route('/IDCT', methods=['GET', 'POST'])
def IDCT():
    form = InputForm_IDCT(request.form)
    if form.validate():
        result = compute_IDCT(form.N.data)
    else:
        result = None

    return render_template('view.html',form=form, result=result)

@app.route('/DST', methods=['GET', 'POST'])
def DST():
    form = InputForm_DST(request.form)
    if form.validate():
        result = compute_DST(form.N.data)
    else:
        result = None

    return render_template('view.html',form=form, result=result)


@app.route('/IDST', methods=['GET', 'POST'])
def IDST():
    form = InputForm_IDST(request.form)
    if form.validate():
        result = compute_IDST(form.N.data)
    else:
        result = None

    return render_template('view.html',form=form, result=result)

@app.route('/Damping', methods=['GET', 'POST'])
def Damping():
    form = InputForm_Damping(request.form)
    if form.validate():
        result = compute_Damping(form.A.data, form.b.data,form.w.data, form.T.data)
    else:
        result = None

    return render_template('view.html',form=form, result=result)

@app.route('/formula', methods=['GET', 'POST'])
def formula():
    form = InputForm_formula(request.form)
    if form.validate():
        result = compute_formula(form.A.data, form.B.data)
    else:
        result = None

    return render_template('view.html',form=form, result=result)


if __name__ == '__main__':
    app.run(debug=True)
