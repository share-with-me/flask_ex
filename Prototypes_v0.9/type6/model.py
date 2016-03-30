from wtforms import Form, FloatField, validators
from math import pi

#validates inputs in the form
#label holds the label
#default holds the default values
class InputForm(Form):
    Fsr = FloatField(
        label='Sampling rate(/s)', default=150.0,
        validators=[validators.InputRequired()]) 
    freq = FloatField(
        label='Frequency of the signal(1/s)', default=5,
        validators=[validators.InputRequired()])
    