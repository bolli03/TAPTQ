import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from vggt.models.vggt import VGGT
from PTQ.vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================
#  QuantVGGT: 校准样本噪声过滤函数
# =========================================
def filter_noisy_scans_by_quantvggt(all_scan_features, retain_ratio=0.9):
    """
    按照 QuantVGGT 的策略筛选稳定样本（噪声过滤）
    输入：
        all_scan_features: List[List[Tensor]]
            每个元素是一个 scan 的 aggregated_tokens_list（每层的 token 特征）
        retain_ratio: float
            保留稳定样本比例，例如 0.9 表示保留稳定性前 90% 的样本
    返回：
        keep_indices: List[int] 需保留的样本下标
        stability_scores: np.ndarray 稳定性指标（数值越小越稳定）
    """
    stability_scores = []
    for scan_layers in all_scan_features:
        # scan_layers: List[Tensor] 每层 [B, S, P, D]
        tokens = torch.stack(scan_layers, dim=0)  # [L, B, S, P, D]
        L, B, S, P, D = tokens.shape
        tokens = tokens.squeeze(1).reshape(L, S * P, D)  # [L, N, D]
        
        # 计算层间波动：跨层方差均值（全 token 全维度）
        layer_var = tokens.var(dim=0).mean().item()  # 一个标量：该 scan 的整体不稳定度
        stability_scores.append(layer_var)
    
    stability_scores = np.array(stability_scores)
    
    # 根据方差从小到大排序（越小越稳定）
    sorted_idx = np.argsort(stability_scores)
    keep_count = int(len(sorted_idx) * retain_ratio)
    keep_indices = sorted_idx[:keep_count]

    print(f"\n🧹 QuantVGGT 噪声过滤：共 {len(stability_scores)} 个样本，保留最稳定的 {keep_count} 个 ({retain_ratio*100:.1f}%)")
    print(f"过滤掉 {len(stability_scores) - keep_count} 个噪声样本")
    return keep_indices.tolist(), stability_scores


# =========================================
# 模型与数据加载
# =========================================
pretrained_model_name_or_path = "/root/autodl-tmp/hf_hub/models--facebook--VGGT-1B"
model = VGGT.from_pretrained(pretrained_model_name_or_path).to(device).eval()

root_dir = "/root/Pi3-evaluation/data/dtu"
scan_dirs = sorted([
    os.path.join(root_dir, d, "images")
    for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d, "images"))
])

pose_tokens_list = []
scan_names = []
all_scan_features = []  # 存每个 scan 的多层特征，用于噪声筛选

print(f"共找到 {len(scan_dirs)} 个样本：")
for d in scan_dirs:
    print("  ", d)

# =========================================
# 提取多层特征（不立即过滤）
# =========================================
for scan_path in scan_dirs:
    image_names = [
        os.path.join(scan_path, f)
        for f in os.listdir(scan_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
    ]
    image_names.sort()
    if len(image_names) == 0:
        print(f"[WARN] {scan_path} 没有图片，跳过。")
        continue

    images = load_and_preprocess_images(image_names).to(device)

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):
            if len(images.shape) == 4:
                images = images.unsqueeze(0)

            aggregated_tokens_list, patch_start_idx = model.aggregator(images)
            all_scan_features.append(aggregated_tokens_list)

            # 提取最后一层 pose token 作为 scan 初步特征（先存起来）
            tokens = aggregated_tokens_list[-1]
            pose_tokens = tokens[:, :, 0]
            pose_tokens = model.camera_head.token_norm(pose_tokens)
            pose_feat = pose_tokens.reshape(-1).cpu().numpy()
            pose_tokens_list.append(pose_feat)
            scan_names.append(scan_path.split("/")[-2])

# =========================================
# 噪声过滤阶段（对所有 scan 特征）
# =========================================
keep_indices, stability_scores = filter_noisy_scans_by_quantvggt(all_scan_features, retain_ratio=0.9)

# 只保留稳定 scan
pose_tokens_np = np.stack(pose_tokens_list)[keep_indices]
scan_names = [scan_names[i] for i in keep_indices]
print(f"✅ 过滤后剩余 {len(scan_names)} 个稳定样本")

# =========================================
# 聚类（逻辑保持不变）
# =========================================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(pose_tokens_np)
centers = kmeans.cluster_centers_

center_indices = []
for i in range(4):
    cluster_points = pose_tokens_np[labels == i]
    if len(cluster_points) == 0:
        center_indices.append(None)
        continue
    dists = np.linalg.norm(cluster_points - centers[i], axis=1)
    idx_in_cluster = np.argmin(dists)
    global_idx = np.where(labels == i)[0][idx_in_cluster]
    center_indices.append(global_idx)

# =========================================
# 输出 + 可视化
# =========================================
print("\n每个样本的聚类编号：")
for name, label in zip(scan_names, labels):
    print(f"{name}: Cluster {label}")

print("🧩 对应样本名称：", [scan_names[i] if i is not None else None for i in center_indices])

pca = PCA(n_components=2)
pose_2d = pca.fit_transform(pose_tokens_np)

plt.figure(figsize=(7, 6))
for i in range(4):
    idx = labels == i
    plt.scatter(pose_2d[idx, 0], pose_2d[idx, 1], label=f"Cluster {i}", alpha=0.7)

for i, name in enumerate(scan_names):
    plt.text(pose_2d[i, 0], pose_2d[i, 1], name.replace("scan", ""), fontsize=8, alpha=0.7)

plt.title("Pose Tokens Clustering (After QuantVGGT Noise Filtering)")
plt.xlabel("PCA Dim 1")
plt.ylabel("PCA Dim 2")
plt.legend()
plt.tight_layout()

save_path = os.path.join(os.getcwd(), "pose_cluster_after_quantvggt_filter.png")
plt.savefig(save_path, dpi=300)
print(f"\n✅ 聚类图已保存到: {save_path}")
plt.close()
