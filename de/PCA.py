""" Performs PCA analysis on RNAseq knockout data using functions from PCA_helpers.py"""
from .PCA_helpers import performPCA, r2x, KOdataframe, plottingPCs, plottingPCreplicates
from .importData import importRNAseqKO

data = importRNAseqKO()
# Analyze data with replicates
pca_list = performPCA(data.T, 25)
r2x(25, pca_list[0])
KO_list = KOdataframe(data, pca_list[1])
plottingPCs(KO_list)
plottingPCreplicates(KO_list, 1, 2)
plottingPCreplicates(KO_list, 1, 3)
plottingPCreplicates(KO_list, 1, 4)
plottingPCreplicates(KO_list, 2, 3)
plottingPCreplicates(KO_list, 2, 4)
plottingPCreplicates(KO_list, 3, 4)

# Analyze data with replicates averaged together
data_combined = data.groupby(by=data.columns, axis=1).mean()
pca_list2 = performPCA(data_combined.T, 25)
r2x(25, pca_list2[0])
KO_list2 = KOdataframe(data_combined, pca_list2[1])
plottingPCs(KO_list2)
