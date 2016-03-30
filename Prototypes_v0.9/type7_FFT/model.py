from wtforms import Form, FloatField, validators
from math import pi
import wtforms

#validates inputs in the form
#label holds the label
#default holds the default values
class InputForm_1D(Form):
    Fsr = FloatField(
        label='Sampling rate(/s)', default=150.0,
        validators=[validators.InputRequired()]) 
    freq = FloatField(
        label='Frequency of the signal(1/s)', default=5,
        validators=[validators.InputRequired()])

class InputForm_2D(Form):
    N = FloatField(
        label='N', default=30.0,
        validators=[validators.InputRequired()]) 
    

class InputForm_ND(Form):
    N = FloatField(
        label='N', default=30.0,
        validators=[validators.InputRequired()]) 
    

class InputForm_DCT(Form):
    N = FloatField(
        label='N', default=100.0,
        validators=[validators.InputRequired()]) 
    

class InputForm_IDCT(Form):
    N = FloatField(
        label='N', default=100.0,
        validators=[validators.InputRequired()]) 


class InputForm_DST(Form):
    N = FloatField(
        label='N', default=100.0,
        validators=[validators.InputRequired()]) 
    

class InputForm_IDST(Form):
    N = FloatField(
        label='N', default=100.0,
        validators=[validators.InputRequired()]) 
    

class InputForm_Damping(Form):
    A = FloatField(
        label='amplitude (m)', default=1.0,
        validators=[validators.InputRequired()]) 
    b = FloatField(
        label='damping factor (kg/s)', default=0,
        validators=[validators.InputRequired()])
    w = FloatField(
        label='frequency (1/s)', default=2*pi, 
        validators=[validators.InputRequired()])
    T = FloatField(
        label='time interval (s)', default=18,
        validators=[validators.InputRequired()])
    
    
class InputForm_formula(Form):
    A = wtforms.fields.TextField(
        label='Expression in x',
        validators=[validators.InputRequired()]) 
    B = wtforms.fields.TextField(
        label='Domain [minx, maxx] ',
        validators=[validators.InputRequired()])

