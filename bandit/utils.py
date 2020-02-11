import sys
from time import sleep
import numpy as np

def progress_bar(i, horizon):
    bar = "[" + "#"*int(i/10) + " "*(int(horizon/10) - int(i/10)) + "]"
    sys.stdout.write("%s(%d/%d)\r"%(bar, (i), horizon))
    sys.stdout.flush()
    sleep(1e-9)

def sigmoid(x):
    """Sigmoid function."""
    return float(1 / (1 + np.exp(-x)))
