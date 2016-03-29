#uses ajax to convert the inputted text to uppercase
from flask import Flask, request, render_template
app = Flask(__name__)
@app.route('/')

def index():
	return render_template("formajax.html")

@app.route('/convert', methods = ['POST'])
def convert(data):
	return request.form['text'].upper()

if __name__ == '__main__':
	app.run(
		debug = True,
		host = '127.0.0.1',
		port = 8080)