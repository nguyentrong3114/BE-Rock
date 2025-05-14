
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class RockClusteringHybrid:
    def __init__(self, k=2, theta=0.5, alpha=0.5, min_goodness=0.001,
                 max_cluster_size_ratio=0.8, sample_ratio=None, verbose=False,
                 categorical_cols=None, numeric_cols=None):
        self.k = k
        self.theta = theta
        self.alpha = alpha
        self.min_goodness = min_goodness
        self.max_cluster_size_ratio = max_cluster_size_ratio
        self.verbose = verbose
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols

    def _similarity_matrix(self, X_df):
        X_num = X_df[self.numeric_cols].to_numpy() if self.numeric_cols else np.zeros((X_df.shape[0], 0))
        X_cat = X_df[self.categorical_cols] if self.categorical_cols else pd.DataFrame()

        self.ohe_ = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat_enc = self.ohe_.fit_transform(X_cat) if not X_cat.empty else np.zeros((X_df.shape[0], 0))

        if X_num.shape[1] > 0:
            D_num = pairwise_distances(X_num, metric='euclidean')
            D_num /= D_num.max() if D_num.max() != 0 else 1
        else:
            D_num = np.zeros((X_df.shape[0], X_df.shape[0]))

        if X_cat_enc.shape[1] > 0:
            D_cat = pairwise_distances(X_cat_enc, metric='hamming')
            D_cat /= D_cat.max() if D_cat.max() != 0 else 1
        else:
            D_cat = np.zeros((X_df.shape[0], X_df.shape[0]))

        D_mix = self.alpha * D_num + (1 - self.alpha) * D_cat if X_num.shape[1] > 0 else D_cat
        S = 1 - D_mix
        return (S + S.T) / 2

    def _calculate_goodness(self, ci, cj, link_matrix):
        num_links = link_matrix[np.ix_(ci, cj)].sum()
        size_ci, size_cj = len(ci), len(cj)
        size_ratio = min(size_ci, size_cj) / max(size_ci, size_cj)
        denom = (size_ci + size_cj) ** (1 + 2 * self.theta)
        if denom == 0:
            return -np.inf
        return (num_links / denom) * size_ratio

    def fit(self, X_df):
        self._X_df = X_df.reset_index(drop=True)
        n_samples = X_df.shape[0]

        if self.verbose:
            print(f"📊 Running ROCK clustering on {n_samples} rows")

        S = self._similarity_matrix(self._X_df)
        links = (S >= self.theta).astype(int)
        link_matrix = links @ links

        clusters = [{i} for i in range(n_samples)]
        max_cluster_size = int(n_samples * self.max_cluster_size_ratio)

        while len(clusters) > self.k:
            best_pair, max_goodness = None, -np.inf
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if len(clusters[i]) + len(clusters[j]) > max_cluster_size:
                        continue
                    goodness = self._calculate_goodness(list(clusters[i]), list(clusters[j]), link_matrix)
                    if goodness > max_goodness and goodness > self.min_goodness:
                        best_pair, max_goodness = (i, j), goodness
            if best_pair is None:
                if self.verbose:
                    print("⚠ No more mergeable clusters.")
                break
            i, j = best_pair
            clusters[i].update(clusters[j])
            clusters.pop(j)
            if self.verbose:
                print(f"✅ Merged clusters {i} and {j}, remaining: {len(clusters)}")

        labels = np.zeros(n_samples, dtype=int)
        for idx, cluster in enumerate(clusters):
            for point in cluster:
                labels[point] = idx

        if self.verbose:
            print("🎯 Clustering complete.")
        return labels
