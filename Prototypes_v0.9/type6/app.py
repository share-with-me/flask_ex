from model import InputForm
from flask import Flask, render_template, request
from fft import compute
import sys

app = Flask(__name__)

from flask_bootstrap import Bootstrap
Bootstrap(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    if form.validate():
        result = compute(form.Fsr.data, form.freq.data)
    else:
        result = None

    return render_template('view.html',form=form, result=result)

if __name__ == '__main__':
    app.run(debug=True)
