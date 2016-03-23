from flask import Flask, Response, render_template
app = Flask(__name__)

@app.route('/')

def index():
	return render_template("dl.html")

@app.route("/Download")
def download():
	with open("outputs/data.csv") as fp:
		csv = fp.read()

	return Response(
		csv,
		mimetype = "text/csv",
		headers = {"Content-disposition":
				"attachment; filename = mplot.csv"}

		)
if __name__ == "__main__":
	app.run(
		debug = True,
		port = int(80),
		host = "127.0.0.1"
		)