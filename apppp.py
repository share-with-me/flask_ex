from flask import Flask, render_template, request, url_for
app = Flask(__name__)

@app.route('/')
def form():
    return render_template('form_submit.html')

@app.route('/hello/', methods=['POST'])
def hello():
    name=request.form['yourname']
    number=request.form['yournumber']
    minn = request.form['price-min']
    maxx = request.form['price-max']
    return render_template('form_action.html', name=name, number=number, minn = minn, maxx = maxx)

if __name__ == '__main__':
  app.run( 
        host="0.0.0.0",
        port=int("80")
  )
