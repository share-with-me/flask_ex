#Read the contents of a file without saving it in flask
from flask import Flask, Response, render_template, request

app = Flask(__name__)
@app.route('/')
def index():
	return render_template('upload_files.html')

@app.route('/read', methods = ['GET','POST'])
def readfile():
	if request.method == 'POST':
		file = request.files['file']
		if file:
			return file.read()

if __name__ == '__main__':
	app.run(
		debug = True,
		host = '127.0.0.1',
		port = int(80)
		)
