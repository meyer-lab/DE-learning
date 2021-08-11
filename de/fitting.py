""" Methods implementing the model as a fitting process. """

import numpy as np
import pandas as pd
from scipy.special import expit
from .factorization import alpha, factorizeEstimate, cellLineComparison
from .importData import importLINCS
from .linearModel import runFitting


def mergedFitting(cellLine1, cellLine2):
    """Given two cell lines, compute the cost of fitting each of them individually and the cost of fitting a shared w matrix."""
    index_list1, index_list2 = cellLineComparison(cellLine1, cellLine2)

    data1, _ = importLINCS(cellLine1)
    data2, _ = importLINCS(cellLine2)
    data1_df = pd.DataFrame(data1)
    data2_df = pd.DataFrame(data2)
    data1_edited = data1_df.iloc[index_list1, index_list1]
    data2_edited = data2_df.iloc[index_list2, index_list2]
    data1_final = data1_edited.values
    data2_final = data2_edited.values
    shared_data = [data1_final, data2_final]

    w_shared, eta_list = factorizeEstimate(shared_data)

    return w_shared, eta_list

def impute(data, test_data=None, linear=False):
    """ Impute by repeated fitting. """
    missing = np.isnan(data)
    data = np.nan_to_num(data)
    test_data = np.nan_to_num(test_data)

    for ii in range(10):
        U = np.copy(data)
        np.fill_diagonal(U, 0.0)
        data_last = np.copy(data)

        # Fit
        if linear:
            model = runFitting(data)
        else:
            w, eta = factorizeEstimate(data)

        # Fill-in with model prediction
        if linear:
            predictt = model.predict(test_data)
            #cost
            diff = np.absolute(data - predictt)
            square = np.power(diff,2)
            sum_errors = np.sum(square)
            print(f"cost: {sum_errors}")
        else:
            predictt = eta[0][:, np.newaxis] * expit(w @ U) / alpha
        data[missing] = predictt[missing]

        print(np.linalg.norm(data - data_last))

    return data