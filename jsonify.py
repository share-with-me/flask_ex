from flask import Flask, render_template, jsonify
app = Flask(__name__)

@app.route('/')
def json_data():
	colors = [{'type': 'primary', 'value':1},
			  {'type': 'secondary', 'value':2} ]
	return jsonify(results = colors)

if __name__ == '__main__':
	app.run(
		host = '127.0.0.1',
		port = int(80),
		debug = True)