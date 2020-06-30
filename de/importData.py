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

def form_matrix(data_in):
    """Takes in parameter of dataframe read in by importRNAseqKO() and forms matrix: rows = gene, columns = knockout model.
    There are 85 knockout models (including negative control) and 84 corresponding genes measured."""
    # average knockout replicate values and remove duplicate gene rows
    data_combined = data_in.groupby(by=data.columns, axis=1).mean() # knockout replicates
    data_combined = data_combined.groupby(["GeneSymbol"]).max() # duplicate genes

    # average negative controls into 1 value, drop WT (control = neg)
    data_combined['neg'] = data_combined[['neg01', 'neg10']].mean(axis=1)
    for i in range(1, 10):
        data_combined = data_combined.drop(["neg0"+str(i)], axis=1)
    data_combined = data_combined.drop(["neg10"], axis=1)
    data_combined = data_combined.drop(["WT"], axis=1)

    # Loop through column names to identify rows to keep and add to new matrix dataframe
    matrix = pd.DataFrame()
    for i, gene in enumerate(data_combined.columns):
        if gene != "neg":
            matrix[gene] = data_combined.loc[gene,:]
    matrix = matrix.T
    return matrix
