"Testing melanoma gene import"
from os.path import join, dirname
import numpy as np
import pandas as pd

def importTorre():  
    """ Imports all Torre genes. """
    path_here = dirname(dirname(__file__))
    data = pd.read_csv(join(path_here, "de/data/torre_metadata.txt"), header = None, index_col="geneName", sep='', dtype = None)
    data = data[:,[0,2,3]]
    return data