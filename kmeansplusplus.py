import torch


def kmeans_plusplus(features, n_clusters, random_state):
    # features (N, D)
    n_samples = features.shape[0]
    generator = torch.Generator(device=str(features.device))
    generator.manual_seed(random_state)
    center_id = torch.randint(n_samples, (1,), generator=generator, device=features.device)
    centroids = features[center_id] # (1, D)
    # centroids = features[-1, :].unsqueeze(0)  # (1, D)
    for _ in range(n_clusters - 1):
        dis = torch.cdist(features, centroids)  # (N, C)
        min_dis = torch.min(dis, dim=1).values  # (N)
        new_centroid_id = torch.argmax(min_dis)  # ()
        new_centroid = features[new_centroid_id].unsqueeze(0)  # (1, D)
        centroids = torch.cat([centroids, new_centroid], dim=0)
    return centroids


def kmeans(features, n_clusters=2, max_iter=300, random_state=0, device='cuda'):
    features = features.to(device)  # features (N, D)
    centroids = kmeans_plusplus(features, n_clusters, random_state)  # (C, D)
    cluster_label = torch.tensor(0, device=device)
    label_matrix = torch.tensor(0, device=device)
    converged = False
    for i in range(max_iter):
        pre_centroids = centroids
        dis = torch.cdist(features, centroids)  # (N, C)
        cluster_label = torch.argmin(dis, dim=1)  # (N)
        label_matrix = torch.zeros(features.size(0), n_clusters, device=device)  # (N, C)
        label_matrix.scatter_(1, cluster_label.unsqueeze(1), 1)
        label_sum = torch.sum(label_matrix, dim=0)  # (C)
        label_matrix = label_matrix / label_sum.unsqueeze(0)  # Broadcasting
        centroids = torch.mm(label_matrix.t(), features)  # (C, N)*(N, D) -> (C, D)
        if torch.allclose(pre_centroids, centroids):
            converged = True
            break
    if not converged:
        print('Warning: Clustering did not converge.')
    return cluster_label, label_matrix, centroids


if __name__ == "__main__":
    features = torch.randn(10240, 512)
    n_clusters = 512
    labels, label_matrix, centroids = kmeans(features, n_clusters)
    print("Labels:", labels)
    print("Centroids:", centroids)
