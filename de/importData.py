"""Contains function for importing and handling knockout RNAseq data"""
import pandas as pd

def importRNAseqKO():
    "Imports knockout RNAseq data, sets index to Gene Symbol, and removes knockouts without measurements"
    data = pd.read_csv("data/rpmCounts_allRuns_matrix.tsv.xz", index_col="GeneSymbol", delim_whitespace=True)
    data = data.drop(["gene_id"], axis=1) # using GeneSymbol as index
    data = data.dropna(axis=1) # remove columns with no measurements
    # remove excess info from knockout names to make replicate names identical
    KO_genes = list(list(zip(*data.columns.str.split("-")))[0])
    data.columns = KO_genes
    return data
