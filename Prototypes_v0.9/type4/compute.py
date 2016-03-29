from numpy import exp, cos, linspace
import bokeh.plotting as plt
import os, re

def damped_vibrations(t, A, b, w):
    return A*exp(-b*t)*cos(w*t)

def compute(A, b, w, T, resolution=500):
    t = linspace(0, T, resolution+1)
    u = damped_vibrations(t, A, b, w)

    # create a new plot with a title and axis labels
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"
    p = plt.figure(title="simple line example", tools=TOOLS,
                   x_axis_label='t', y_axis_label='y')

    # add a line renderer with legend and line thickness
    p.line(t, u, legend="u(t)", line_width=2)

    from bokeh.resources import CDN
    from bokeh.embed import components
    script, div = components(p)
   
    return script, div

if __name__ == '__main__':
    print compute(1, 0.1, 2, 20)