"""Contains function for importing and handling knockout RNAseq data"""
from os.path import join, dirname
import pandas as pd

def importRNAseqKO():
    data = pd.read_csv("data/rpmCounts_allRuns_matrix.tsv.xz", index_col = "GeneSymbol", delim_whitespace = True)
    data = data.drop(["gene_id"], axis=1) # using GeneSymbol as index
    data = data.dropna(axis=1) # remove columns with no measurements
    return data