"Testing melanoma gene import"
from os.path import join, dirname
import numpy as np
import pandas as pd

def importmelanoma(cellLine):  
    """ Imports all Torre genes with Fig 5D data"""
    path_here = dirname(dirname(__file__))
    xdata = pd.read_csv(join(path_here, "de/data/sumarizedResults.txt"), header = True, index_col="geneName", sep='', dtype = None)
    ydata = pd.read_csv(join(path_here, "de/data/colonyGrowthResults_allhits.txt", header = True, index_col="geneName", sep = '', dtype = None))
    xdata = xdata[:,[0,4]]
    ydata = xdata[:,[0,2]]
    return xdata, ydata

def mergedData(xdata, ydata):
    data = pd.merge(xdata, ydata, on='target',how='inner')
    return data

def splitnodes(data):
    above = [""]
    below = [""]
    for key, data[0] in data:
        if data[1] < data[2]:
            above.append(data[0])
            return above
        elif data[1] > data[2]:
            below.append(data[0])
            return below 
        

