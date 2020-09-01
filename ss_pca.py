import numpy as np
from sklearn.decomposition import PCA
from de.figures.figureCommon import getSetup

ss_vals = np.loadtxt("de/data/steady_state_values.csv", delimiter=',')

pca_object = PCA(n_components=2)
X_r = pca_object.fit_transform(ss_vals)
loadings = pca_object.components_

ax, f = getSetup((8, 8), (1, 1))

i = 0
ax[i].scatter(X_r[:, 0], X_r[:, 1])
ax[i].set_xlabel("PC1 (" + str(round(pca_object.explained_variance_ratio_[0] * 100, 2)) + "%)")
ax[i].set_ylabel("PC2 (" + str(round(pca_object.explained_variance_ratio_[1] * 100, 2)) + "%)")
ax[i].set_title("PC2 vs PC1")

for x in range(10):
    time_sols = np.loadtxt("de/data/time_sol" + str(x + 1) + ".csv", delimiter=',')
    t_scores = np.matmul(loadings, time_sols)
    ax[i].plot(t_scores[0, 0:999], t_scores[1, 0:999])
