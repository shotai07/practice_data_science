import sys
from time import sleep

def progress_bar(i, horizon):
    bar = "[" + "#"*int(i/10) + " "*(int(horizon/10) - int(i/10)) + "]"
    sys.stdout.write("%s(%d/%d)\r"%(bar, (i), horizon))
    sys.stdout.flush()
    sleep(1e-9)
