from model import InputForm   #Uses model from inputform py
from flask import Flask, render_template, request
from compute import compute #Imports compute fn from compute py

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def index():
    form = InputForm(request.form)  #form represents an instance of the InputForm
    if form.validate():
        result = compute(form.A.data, form.B.data) #Result holds the output image
    else:
        result = None

    return render_template('view.html', form=form, result=result)

@app.route('/view')
def view():
	form = InputForm(request.form)
	return str(form.A.data)

if __name__ == '__main__':
    app.run(debug=True)