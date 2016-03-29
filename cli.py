from flask import Flask, request
app = Flask(__name__)
     
@app.route("/")
<<<<<<< HEAD
def hello1():
    return '<form action="/echo" method="GET"><button type="submit" value="Echo">Click Meeee!</button></form>'
def hello():
    return '<form action="/echo" method="GET"><button type="submit" value="Echo">Click Me!</button></form>'
def hello1():
    return '<form action="/echo" method="GET"><button type="submit" value="Echo">Click Meeee!</button></form>'
=======
def hello():
    return '<form action="/echo" method="GET"><button type="submit" value="Echo">Click Me!</button></form>'
>>>>>>> 7f401c0c71672b41b6893370c96feaab71682882
     
@app.route("/echo")
def echo(): 
    return "Hey you just clicked me!"
     
if __name__ == "__main__":
    app.run()