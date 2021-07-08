"""Contains function for importing and handling knockout RNAseq data"""
from os.path import join, dirname
import numpy as np
import pandas as pd


def importLINCS(cellLine):
    """ Import processed LINCS data. """
    path_here = dirname(dirname(__file__))

    data = np.load(join(path_here, "de/data/", cellLine + "_RNAi_matrix.npy"))
    annotation = pd.read_csv(join(path_here, "de/data/", cellLine + "_genes.txt"), header=None)

    return data, annotation


def importRNAseqKO():
    """Imports knockout RNAseq data, sets index to Gene Symbol, and removes knockouts without measurements"""
    path_here = dirname(dirname(__file__))

    data = pd.read_csv(join(path_here, "de/data/rpmCounts_allRuns_matrix.tsv.xz"), index_col="GeneSymbol", delim_whitespace=True)
    data = data.drop(["gene_id"], axis=1)  # using GeneSymbol as index
    data = data.dropna(axis=1)  # remove columns with no measurements

    # remove excess info from knockout names to make replicate names identical
    KO_genes = list(list(zip(*data.columns.str.split("-")))[0])
    data.columns = KO_genes
    return data


def ImportMelanoma():
    """Takes in parameter of dataframe read in by importRNAseqKO() and forms matrix: rows = gene, columns = knockout model.
    There are 84 knockout models (including negative control) and 83 corresponding genes measured."""
    data_in = importRNAseqKO()
    # average knockout replicate values and remove duplicate gene rows
    data_combined = data_in.groupby(by=data_in.columns, axis=1).mean()  # knockout replicates
    data_combined = data_combined.groupby(["GeneSymbol"]).max()  # duplicate genes

    # average negative controls into 1 value, drop WT (control = neg)
    data_combined["neg"] = data_combined[["neg01", "neg10"]].mean(axis=1)
    for i in range(1, 10):
        data_combined = data_combined.drop(["neg0" + str(i)], axis=1)
    data_combined = data_combined.drop(["neg10"], axis=1)
    data_combined = data_combined.drop(["WT"], axis=1)

    # Loop through column names to identify rows to keep and add to new matrix dataframe
    matrix = pd.DataFrame()
    for i, gene in enumerate(data_combined.columns):
        if gene != "neg":
            matrix[gene] = data_combined.loc[gene, :]
    matrix = matrix.T
    # Convert dataframe to numpy array for comparison with model
    matrix = matrix.to_numpy()
    return matrix


def prepData():
    """ Load RNAseq data then average replicates and negative controls for PCA """
    d = importRNAseqKO()
    data_prepped = d.groupby(by=d.columns, axis=1).mean()
    data_prepped["neg"] = data_prepped[["neg01", "neg10"]].mean(axis=1)
    for i in range(1, 10):
        data_prepped = data_prepped.drop(["neg0" + str(i)], axis=1)
    data_prepped = data_prepped.drop(["neg10"], axis=1)
    return data_prepped


def importGenomeData():
    """ Loads genome-wide RNAseq data, perturbation information, and landmark gene information. """
    path_here = dirname(dirname(__file__))
    # Import perturbation data and join 2 datasets
    data1 = pd.read_csv(join(path_here, "de/data/GSE92742_Broad_LINCS_Level2.csv.xz"), compression="xz")
    data2 = pd.read_csv(join(path_here, "de/data/GSE70138_Broad_LINCS_Level2.csv.xz"), compression="xz")
    data = data1.join(data2)
    # Import perturbation IDs and names
    inst_info = pd.read_csv(join(path_here, "de/data/GSE106127_inst_info.txt"), sep="\t").set_index("inst_id")
    inst_info = inst_info.drop(["pert_time", "pert_time_unit", "seed_seq_6mer", "seed_seq_7mer", "pert_itime"], axis=1)
    # Import gene IDs and names
    gene_info = pd.read_csv(join(path_here, "de/data/GSE92742_Broad_LINCS_gene_info.txt"), sep='\t', index_col="pr_gene_id")
    gene_info = gene_info.drop(["pr_gene_title", "pr_is_lm", "pr_is_bing"], axis=1)
    return data, inst_info, gene_info


def determineCellTypes(inst_info):
    """ Returns list of cell types and the amount of perturbations performed on each. """
    cell_types_counts = inst_info.groupby(by="cell_id").count()
    cell_types_counts = cell_types_counts.drop(["pert_iname", "pert_type", "pert_time", "pert_time_unit", "seed_seq_6mer", "seed_seq_7mer", "pert_itime"], axis=1)
    return cell_types_counts


def cell_type_perturbations(data, inst_info, gene_info, cell_id):
    """ Returns matrix with rows corresponding to landmark genes measured and columns corresponding to average value of each perturbation performed. """
    inst_info_celltype = inst_info.loc[((inst_info["cell_id"] == cell_id) | (inst_info["cell_id"] == (cell_id + ".311"))) & ((inst_info["pert_type"] == "ctl_vector") | (inst_info["pert_type"] == "trt_sh"))]
    drop_cols = []
    rename_cols = []
    new_names = []
    # Find perturbations by RNAi and controls
    for col in data.columns:
        if col not in inst_info_celltype.index:
            drop_cols.append(col)
        else:
            rename_cols.append(col)
            new_names.append(inst_info_celltype.loc[col, "pert_iname"])
    out_celltype = data.drop(drop_cols, axis=1)
    # Rename columns with perturbation name
    out_celltype.rename(columns=dict(zip(rename_cols, new_names)), inplace=True)
    # Average replicates
    out_celltype = out_celltype.groupby(by=out_celltype.columns, axis=1).mean()
    # Replace row numbers with name of gene measured
    out_celltype.index = list(map(int, out_celltype.index))
    out_celltype = out_celltype.join(gene_info)
    out_celltype = out_celltype.set_index("pr_gene_symbol")
    out_celltype.sort_index(inplace=True)
    # Remove columns of perturbations not corresponding to a measured gene or control
    drop_cols = []
    for x in out_celltype.columns:
        if x not in out_celltype.index and x != "LUCIFERASE":
            drop_cols.append(x)
    out_celltype.drop(drop_cols, axis=1, inplace=True)
    # Remove genes that were measured but not perturbed
    drop_rows = []
    for x in out_celltype.index:
        if x not in out_celltype.columns:
            drop_rows.append(x)
    out_celltype.drop(drop_rows, inplace=True)
    # Move control to end of dataframe
    out_celltype = out_celltype[[x for x in out_celltype if x not in ["LUCIFERASE"]] + ["LUCIFERASE"]]
    return out_celltype

def importmelanoma():
    """ Imports all Torre genes with Fig 5D data and merges into one matrix"""
    path_here = dirname(dirname(__file__))
    xdata = pd.read_csv(join(path_here, "de/data/sumarizedResults.txt", sep = " "))
    ydata = pd.read_csv(join(path_here, "de/data/colonyGrowthResults_allhits.txt", sep = " "))
    xdata = xdata[:,[0,4]] # meanlFC_IF values
    ydata = xdata[:,[0,2]] # Rcolonies_lFC values 
    #merges data for genes with both meanlFC_IF and Rcolonies_lFC values 
    melan_genes = pd.merge(xdata, ydata, on='target',how='inner') 
    data = pd.DataFrame(melan_genes, index=melan_genes[0])
    return data

def splitnodes(data):
    """Separates genes into resistant and preresistant"""
    above = [""]
    below = [""]
    for i, row in data.iterrows():
        if data[1] < data[2]:
            above.append(data[0])
        elif data[1] > data[2]:
            below.append(data[0])
        return above, below


