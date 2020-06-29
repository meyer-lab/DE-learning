from PCA_helpers import *

data = importRNAseqKO()
# Analyze data with replicates
pca_list = pca(data.T, 25)
r2x(25, pca_list[0], "replicates_r2x.png")
KO_list = KOdataframe(data, pca_list[1])
plottingPCs(KO_list, "replicates_PCA.png")
plottingPCreplicates(KO_list, 1, 2)
plottingPCreplicates(KO_list, 1, 3)
plottingPCreplicates(KO_list, 1, 4)
plottingPCreplicates(KO_list, 2, 3)
plottingPCreplicates(KO_list, 2, 4)
plottingPCreplicates(KO_list, 3, 4)

# Analyze data with replicates averaged together
data_combined = data.groupby(by=data.columns, axis=1).mean()
pca_list2 = pca(data_combined.T, 25)
r2x(25, pca_list2[0], "combined_r2x.png")
KO_list2 = KOdataframe(data_combined, pca_list2[1])
plottingPCs(KO_list2, "combined_PCA.png")