import wtforms
from wtforms import Form, FloatField, validators
from math import pi

#validates inputs in the form
#label holds the label
#default holds the default values
class InputForm(Form):
    A = wtforms.fields.TextField(
        label='Expression in x',
        validators=[validators.InputRequired()]) 
    B = wtforms.fields.TextField(
        label='Domain [minx, maxx] ',
        validators=[validators.InputRequired()])
    