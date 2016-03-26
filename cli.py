from flask import Flask, request
app = Flask(__name__)
     
@app.route("/")
def hello():
    return '<form action="/echo" method="GET"><button type="submit" value="Echo">Click Me!</button></form>'
     
@app.route("/echo")
def echo(): 
    return "Hey you just clicked me!"
     
if __name__ == "__main__":
    app.run()