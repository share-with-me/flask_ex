#Converts the inputted text to uppercase

from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')

def index():
	return render_template("form.html")

@app.route('/', methods = ['POST'])#POST because client is sending data to the server

def convert():
	text = request.form['text']
	converted = text.upper()
	return converted

if __name__ == '__main__':
	app.run(
		host = '127.0.0.1',
		port = 8080,
		debug = True
		)