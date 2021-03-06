import numpy as np
import os

def compute_mean_std(filename=None):
    data = np.loadtxt(os.path.join('uploads', filename))
    return """
Data from file <tt>%s</tt>:
<p>
<table border=3>
<tr><td> mean    </td><td> %.6g </td></tr>
<tr><td> st.dev. </td><td> %.6g </td></tr>
""" % (filename, np.mean(data), np.std(data))
