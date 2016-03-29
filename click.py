#uses ajax to convert the inputted text to uppercase
from flask import Flask, request, render_template
app = Flask(__name__)
@app.route('/')

def index():
	return render_template("click.html")

@app.route('/click')
def convert():
	return 'Hey, you just clicked me!'

if __name__ == '__main__':
	app.run(
		debug = True,
		host = '127.0.0.1',
		port = 8080)