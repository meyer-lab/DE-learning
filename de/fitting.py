""" Methods implementing the model as a fitting process. """

import numpy as np
import pandas as pd
from scipy.special import expit
from .factorization import alpha, factorizeEstimate, cellLineComparison
from .importData import importLINCS


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


def impute(data):
    """ Impute by repeated fitting. """
    missing = np.isnan(data)
    data = np.nan_to_num(data)

    for ii in range(10):
        U = np.copy(data)
        np.fill_diagonal(U, 0.0)
        data_last = np.copy(data)

        # Fit
        w, eta = factorizeEstimate(data)

        # Fill-in with model prediction
        predictt = eta[0][:, np.newaxis] * expit(w @ U) / alpha
        data[missing] = predictt[missing]

        print(np.linalg.norm(data - data_last))

    return data
