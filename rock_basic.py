import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import Binarizer

class RockClustering:
    def __init__(self, k=2, theta=0.5, metric='jaccard', min_goodness=0.001,
                 max_cluster_size_ratio=0.8, sample_ratio=0.2, verbose=False):
        # Số cụm mong muốn
        self.k = k
        # Ngưỡng similarity (theta) để xác định neighbor
        self.theta = theta
        # Loại similarity metric (mặc định: jaccard)
        self.metric = metric
        # Ngưỡng tối thiểu cho goodness để gộp cụm
        self.min_goodness = min_goodness
        # Tỉ lệ tối đa cho kích thước cụm khi gộp
        self.max_cluster_size_ratio = max_cluster_size_ratio
        # Tỉ lệ sampling trên toàn bộ dữ liệu
        self.sample_ratio = sample_ratio
        # Bật/tắt log thông tin
        self.verbose = verbose

    def preprocess(self, X):
        # Nếu dùng jaccard, cần đảm bảo dữ liệu nhị phân (0/1)
        if self.metric == 'jaccard':
            if not np.all(np.isin(np.unique(X), [0, 1])):
                if self.verbose:
                    print("🔧 Data is not binary. Applying Binarizer (threshold=0).")
                X = Binarizer(threshold=0.0).fit_transform(X)
        return X

    def _similarity(self, X):
        # Tính similarity matrix (1 - khoảng cách) giữa các điểm
        S = 1 - pairwise_distances(X, metric=self.metric)
        # Làm đối xứng ma trận để tránh sai số
        return (S + S.T) / 2

    def _calculate_goodness(self, ci, cj, link_matrix):
        # Tính số lượng link (neighbor chung) giữa hai cụm ci và cj
        num_links = link_matrix[np.ix_(ci, cj)].sum()
        size_ci = len(ci)
        size_cj = len(cj)
        # Tính tỉ lệ kích thước giữa cụm nhỏ và cụm lớn
        size_ratio = min(size_ci, size_cj) / max(size_ci, size_cj)
        # Tính mẫu số theo công thức (ni + nj)^(1 + 2θ)
        denom = (size_ci + size_cj) ** (1 + 2 * self.theta)
        if denom == 0:
            return -np.inf  # Tránh chia 0
        # Công thức goodness, có nhân thêm size_ratio để tránh gộp lệch
        return (num_links / denom) * size_ratio

    def _assign_remaining_points(self, X_all, sample_idx, clusters_sample, labels_sample):
        # Tìm các điểm chưa nằm trong sample
        remaining_idx = np.setdiff1d(np.arange(X_all.shape[0]), sample_idx)
        X_remaining = X_all[remaining_idx]
        # Khởi tạo mảng nhãn cho toàn bộ dữ liệu
        labels_all = np.full(X_all.shape[0], -1)

        # Gán nhãn cho sample
        for idx, point_idx in enumerate(sample_idx):
            labels_all[point_idx] = labels_sample[idx]

        # Gán nhãn cho các điểm còn lại
        for idx, point in zip(remaining_idx, X_remaining):
            max_sim = -1
            best_cluster = -1
            # Tính similarity trung bình với từng cụm
            for cluster_id in np.unique(labels_sample):
                cluster_points = X_all[sample_idx][labels_sample == cluster_id]
                sims = 1 - pairwise_distances(cluster_points, point.reshape(1, -1), metric=self.metric)
                avg_sim = np.mean(sims)
                if avg_sim > max_sim:
                    max_sim = avg_sim
                    best_cluster = cluster_id
            labels_all[idx] = best_cluster

        return labels_all

    def fit(self, X):
        # Bước tiền xử lý
        X = self.preprocess(X)
        n_samples = X.shape[0]

        # Bước 1: Lấy sample ngẫu nhiên
        sample_size = int(self.sample_ratio * n_samples)
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = X[sample_idx]

        if self.verbose:
            print(f"📦 Sampled {sample_size}/{n_samples} points.")

        # Bước 2: Phân cụm trên sample
        S = self._similarity(X_sample)
        links = (S >= self.theta).astype(int)
        link_matrix = links @ links  # Nhân ma trận để đếm số neighbor chung
        clusters = [{i} for i in range(sample_size)]
        max_cluster_size = int(sample_size * self.max_cluster_size_ratio)

        while len(clusters) > self.k:
            best_pair = None
            max_goodness = -np.inf

            # Duyệt tất cả cặp cụm để tìm cặp có goodness cao nhất
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if len(clusters[i]) + len(clusters[j]) > max_cluster_size:
                        continue
                    ci = list(clusters[i])
                    cj = list(clusters[j])
                    goodness = self._calculate_goodness(ci, cj, link_matrix)

                    if goodness > max_goodness and goodness > self.min_goodness:
                        max_goodness = goodness
                        best_pair = (i, j)

            if best_pair is None:
                if self.verbose:
                    print("⚠ No more clusters to merge (goodness below threshold).")
                break

            # Gộp hai cụm tốt nhất
            i, j = best_pair
            clusters[i].update(clusters[j])
            clusters.pop(j)
            if self.verbose:
                print(f"✅ Merged cluster {i} and {j} | Total clusters: {len(clusters)}")

        # Gán nhãn cho sample
        labels_sample = np.zeros(sample_size, dtype=int)
        for idx, cluster in enumerate(clusters):
            for sample in cluster:
                labels_sample[sample] = idx

        if self.verbose:
            print("🎯 Sample clustering complete. Assigning remaining points...")

        # Bước 3: Gán các điểm còn lại vào cụm gần nhất
        labels_all = self._assign_remaining_points(X, sample_idx, clusters, labels_sample)

        if self.verbose:
            print("🎉 All points assigned. Clustering complete.")
        return labels_all