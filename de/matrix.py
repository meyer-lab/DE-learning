import numpy as np
from factorization import cellLineFactorization


def cellLineComparision(cellLine1, cellLine2):
    w1, eta1, annotation1 = cellLineFactorization(cellLine1)
    w2, eta2, annotation2 = cellLineFactorization(cellLine2)


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


