from flask import Flask, flash, redirect, render_template, request
import os
import json
import urllib2

app = Flask(__name__)
 
def getExchangeRates():
    rates = [] #Array initialisation
    response = urllib2.urlopen('http://api.fixer.io/latest')
    data = response.read()
    rdata = json.loads(data, parse_float=float) #load the data
 
    rates.append( rdata['rates']['USD'] ) #Append the data to array
    rates.append( rdata['rates']['GBP'] )
    rates.append( rdata['rates']['HKD'] )
    rates.append( rdata['rates']['AUD'] )
    return rates
 
@app.route("/")
def index():
    rates = getExchangeRates() #Call this function in '/' directory
    return render_template('test.html',**locals())      
 
if __name__ == "__main__":
    app.run(debug = True)