from flask import Flask, render_template, jsonify, request, Response
app = Flask(__name__)

@app.route('/')
def form():
    return render_template('form_submit.html')

@app.route('/hello/', methods=['POST'])
def hello():
    name= str(request.form['yourname'])
    number= str(request.form['yournumber'])
    minn = str(request.form['price-min'])
    maxx = str(request.form['price-max'])
    inputs = [{'name' : name, 'number' : number, 'price-min' : minn,'price-max' : maxx }]
    #content = str(request.form['yourname'])
    return Response(inputs, 
            mimetype='application/json',
            headers={'Content-Disposition':'attachment;filename=inputs.json'})

if __name__ == '__main__':
  app.run( 
        host="0.0.0.0",
        port=int("80"),
        debug = True
  )
