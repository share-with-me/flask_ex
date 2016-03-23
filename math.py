
from flask import Flask, jsonify, render_template, request
app = Flask(__name__)


@app.route('/_add_numbers')
def add_numbers():
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    return jsonify(result=a + b)


@app.route('/_prod_numbers')
def prod_numbers():
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    return jsonify(result=a*b)


@app.route('/_sq_numbers')
def sq_number():
    a = request.args.get('a', 0, type=int)
    return jsonify(result=a*a)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
