import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from factorization import cellLineFactorization


def cellLineComparision(cellLine1, cellLine2):
    w1, eta1, annotation1 = cellLineFactorization(cellLine1)
    w2, eta2, annotation2 = cellLineFactorization(cellLine2)

    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)

    line1_as_set = set(annotation1)
    intersection = line1_as_set.intersection(annotation2)
    intersection_annotation = list(intersection)

    index_list1 = []
    index_list2 = []

    for x in intersection_annotation:
        index_value1 = annotation1.index(x)
        index_list1.append(index_value1)

    index_list1.sort()
    return index_list1, norm1, norm2


index_listHTA3, norm1, norm2 = cellLineComparision('A375', 'HT29')
index_listA3A5, norm3, norm4 = cellLineComparision('A375', 'A549')
index_listHTA5, norm5, norm6 = cellLineComparision('A375', 'HT29')
