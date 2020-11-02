"""Contains function for importing and handling knockout RNAseq data"""
from os.path import join, dirname
import pandas as pd


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


def formMatrix():
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
    data = pd.read_csv(join(path_here, "de/data/GSE92742_Broad_LINCS_Level2.csv.xz"), compression="xz")
    inst_info = pd.read_csv(join(path_here, "de/data/GSE106127_inst_info.txt"), sep="\t").set_index("inst_id")
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
    inst_info_celltype = inst_info.loc[inst_info["cell_id"] == cell_id]
    drop_cols = []
    rename_cols = []
    new_names = []
    # Find perturbations by RNAi or CRISPR
    for col in data.columns:
        if col not in inst_info_celltype.index:
            drop_cols.append(col)
        else:
            rename_cols.append(col)
            new_names.append(inst_info_celltype.loc[col, "pert_iname"])
    # Drop perturbations not by RNAi or CRISPR
    out_celltype = data.drop(drop_cols, axis=1)
    # Relabel perturbations with gene knocked out
    out_celltype = out_celltype.rename(columns=dict(zip(rename_cols, new_names)))
    # Average replicates
    out_celltype = out_celltype.groupby(by=out_celltype.columns, axis=1).mean()
    # Convert landmark gene numbers to ints
    out_celltype.index = list(map(int, out_celltype.index))
    # Change landmark gene numbers to gene symbols
    out_celltype = out_celltype.join(gene_info)
    out_celltype = out_celltype.set_index("pr_gene_symbol")
    out_celltype = out_celltype.sort_index()
    return out_celltype
