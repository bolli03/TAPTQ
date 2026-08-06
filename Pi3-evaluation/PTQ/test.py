import os
import torch
import pdb
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from vggt.models.vggt import VGGT
from PTQ.vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
# dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
# model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

pretrained_model_name_or_path = "/root/autodl-tmp/hf_hub/models--facebook--VGGT-1B"
# pretrained_model_name_or_path = "facebook/VGGT-1B"

model = VGGT.from_pretrained(pretrained_model_name_or_path).to(device).eval()

# Load and preprocess example images (replace with your own image paths)
# image_names = ["./examples/kitchen/images/00.png", "./examples/kitchen/images/01.png", "./examples/kitchen/images/02.png"]
# 定义目标目录路径
# ===============================
# 基础路径
# ===============================
root_dir = "/root/Pi3-evaluation/data/dtu"

# 找到所有 scan*/images 目录
scan_dirs = sorted([
    os.path.join(root_dir, d, "images")
    for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d, "images"))
])

print(f"共找到 {len(scan_dirs)} 个样本：")
for d in scan_dirs:
    print("  ", d)

pose_tokens_list = []
scan_names = []

# ===============================
# 遍历每个 scan 提取 pose tokens
# ===============================
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
            tokens = aggregated_tokens_list[-1]

            # 提取 pose tokens
            pose_tokens = tokens[:, :, 0]
            pose_tokens = model.camera_head.token_norm(pose_tokens)
            # pdb.set_trace()

            # 聚合成一个样本整体特征（平均
            pose_feat = pose_tokens.reshape(-1).cpu().numpy()
            # pose_feat = PCA(n_components=128).fit_transform(pose_feat)

            # pose_feat = pose_tokens.mean(dim=(0, 1)).cpu().numpy()
            pose_tokens_list.append(pose_feat)
            scan_names.append(scan_path.split("/")[-2])  # scan1, scan2, ...

pose_tokens_np = np.stack(pose_tokens_list)
print(f"\n提取到 {pose_tokens_np.shape[0]} 个样本的 pose_tokens，维度 {pose_tokens_np.shape[1]}")

# ===============================
# 聚类
# ===============================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(pose_tokens_np)
centers = kmeans.cluster_centers_

# 找每个聚类中心最近的样本编号
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

# ===============================
# 输出结果
# ===============================
print("\n每个样本的聚类编号：")
for name, label in zip(scan_names, labels):
    print(f"{name}: Cluster {label}")

print("🧩 对应样本名称：", [scan_names[i] if i is not None else None for i in center_indices])

# ===============================
# 可视化（PCA降到2维）
# ===============================
pca = PCA(n_components=2)
pose_2d = pca.fit_transform(pose_tokens_np)

plt.figure(figsize=(7, 6))
for i in range(4):
    idx = labels == i
    plt.scatter(pose_2d[idx, 0], pose_2d[idx, 1], label=f"Cluster {i}", alpha=0.7)

for i, name in enumerate(scan_names):
    plt.text(pose_2d[i, 0], pose_2d[i, 1], name.replace("scan", ""), fontsize=8, alpha=0.7)

plt.title("Pose Tokens Clustering (K=4)")
plt.xlabel("PCA Dim 1")
plt.ylabel("PCA Dim 2")
plt.legend()
plt.tight_layout()

# === 保存到当前目录 ===
save_path = os.path.join(os.getcwd(), "pose_cluster.png")
plt.savefig(save_path, dpi=300)
print(f"\n✅ 聚类图已保存到: {save_path}")
plt.close()
# print("over")
