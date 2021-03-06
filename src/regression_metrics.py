# the metric used in indoor-location-navigation competition
import numpy as np

def iln_comp_metric(xhat, yhat, fhat, x, y, f):
    intermediate = np.sqrt(np.power(xhat - x,2) + np.power(yhat-y,2)) + 15 * np.abs(fhat-f)
    return intermediate.sum()/xhat.shape[0]
