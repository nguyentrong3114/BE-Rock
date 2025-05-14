# rock.py
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class RockClusteringHybrid:
    def __init__(self, k=2, theta=0.5, alpha=0.5, min_goodness=0.001,
                 max_cluster_size_ratio=0.8, sample_ratio=0.2, verbose=False,
                 categorical_cols=None, numeric_cols=None):
        self.k = k
        self.theta = theta
        self.alpha = alpha
        self.min_goodness = min_goodness
        self.max_cluster_size_ratio = max_cluster_size_ratio
        self.sample_ratio = sample_ratio
        self.verbose = verbose
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols

    def _similarity_matrix(self, X_df):
        X_num = X_df[self.numeric_cols].to_numpy() if self.numeric_cols else np.zeros((X_df.shape[0], 0))
        X_cat = X_df[self.categorical_cols] if self.categorical_cols else pd.DataFrame()

        self.ohe_ = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat_enc = self.ohe_.fit_transform(X_cat) if not X_cat.empty else np.zeros((X_df.shape[0], 0))

        if X_cat_enc.shape[1] == 0 and X_num.shape[1] == 0:
            raise ValueError("Encoded data has 0 features. Check selected columns.")

        if X_num.shape[1] > 0:
            D_num = pairwise_distances(X_num, metric='euclidean')
            D_num /= D_num.max() if D_num.max() != 0 else 1
        else:
            D_num = np.zeros((X_df.shape[0], X_df.shape[0]))
            self.alpha = 0

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

    def _assign_remaining_points(self, X_all, sample_idx, clusters_sample, labels_sample):
        remaining_idx = np.setdiff1d(np.arange(X_all.shape[0]), sample_idx)
        labels_all = np.full(X_all.shape[0], -1)

        for s_idx, sample_id in enumerate(sample_idx):
            labels_all[sample_id] = labels_sample[s_idx]

        X_df = self._X_df

        for idx in remaining_idx:
            point = X_df.iloc[idx:idx+1]
            max_sim = -1
            best_cluster = -1
            for cluster_id in np.unique(labels_sample):
                cluster_points_idx = sample_idx[labels_sample == cluster_id]
                cluster_points = X_df.iloc[cluster_points_idx]

                if self.categorical_cols:
                    cluster_points_encoded = self.ohe_.transform(cluster_points[self.categorical_cols])
                    point_encoded = self.ohe_.transform(point[self.categorical_cols])
                    sim_cat = 1 - pairwise_distances(cluster_points_encoded, point_encoded, metric='hamming').mean()
                else:
                    sim_cat = 0

                if self.numeric_cols:
                    sim_num = 1 - pairwise_distances(cluster_points[self.numeric_cols], point[self.numeric_cols], metric='euclidean').mean()
                else:
                    sim_num = 0

                sim = self.alpha * sim_num + (1 - self.alpha) * sim_cat if self.numeric_cols else sim_cat

                if sim > max_sim:
                    max_sim = sim
                    best_cluster = cluster_id

            if best_cluster == -1:
                best_cluster = np.random.choice(np.unique(labels_sample))

            labels_all[idx] = best_cluster

        return labels_all

    def fit(self, X_df):
        self._X_df = X_df.reset_index(drop=True)
        n_samples = X_df.shape[0]
        sample_size = max(int(self.sample_ratio * n_samples), self.k + 1)
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = self._X_df.iloc[sample_idx].reset_index(drop=True)

        if self.verbose:
            print(f"\n📊 Sampled {sample_size}/{n_samples} rows")

        S = self._similarity_matrix(X_sample)
        links = (S >= self.theta).astype(int)
        link_matrix = links @ links

        clusters = [{i} for i in range(sample_size)]
        max_cluster_size = int(sample_size * self.max_cluster_size_ratio)

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
                    print("⚠ No more mergeable clusters. Forcing merge of smallest clusters.")
                # Gộp các cụm nhỏ nhất nếu chưa đủ k
                while len(clusters) > self.k:
                    clusters = sorted(clusters, key=lambda c: len(c))
                    clusters[0].update(clusters[1])
                    clusters.pop(1)
                break
            i, j = best_pair
            clusters[i].update(clusters[j])
            clusters.pop(j)
            if self.verbose:
                print(f"✅ Merged clusters {i} and {j}, remaining: {len(clusters)}")

        labels_sample = np.zeros(sample_size, dtype=int)
        for idx, cluster in enumerate(clusters):
            for point in cluster:
                labels_sample[point] = idx

        labels_all = self._assign_remaining_points(self._X_df, sample_idx, clusters, labels_sample)

        if self.verbose:
            print("🎯 Clustering complete.")
        return labels_all
