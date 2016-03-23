from flask import Flask
app = Flask(__name__)

@app.route('/')

def hello():
	a = 4
	b = 5
	c = a*b
	return c

@app.route('/hello/')

def helo():
	return 'Hola World'

if __name__ == '__main__':
	app.run(host = '0.0.0.0')

