from wtforms import Form, FloatField, validators
from math import pi
import wtforms

#validates inputs in the form
#label holds the label
#default holds the default values
class InputForm_RRLyrae(Form):
    Fsr = FloatField(
        label='Sampling rate(/s)', default=150.0,
        validators=[validators.InputRequired()]) 
    freq = FloatField(
        label='Frequency of the signal(1/s)', default=5,
        validators=[validators.InputRequired()])

class InputForm_Convolution(Form):
    N = FloatField(
        label='N', default=30.0,
        validators=[validators.InputRequired()]) 
    

class InputForm_FFT(Form):
    N = FloatField(
        label='N', default=30.0,
        validators=[validators.InputRequired()]) 
    

class InputForm_Gaussian(Form):
    N = FloatField(
        label='N', default=100.0,
        validators=[validators.InputRequired()]) 
    

class InputForm_LSA(Form):
    N = FloatField(
        label='N', default=100.0,
        validators=[validators.InputRequired()]) 


class InputForm_AF(Form):
    N = FloatField(
        label='N', default=100.0,
        validators=[validators.InputRequired()]) 
    

