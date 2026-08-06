import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils as pn2_utils
from auction_match.auction_match import AuctionMatch
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist, hausdorff_distance
from pytorch3d.ops import knn
from einops import rearrange, repeat
from pointops.funtions import knnquery_heap, furthestsampling
# from pointops import pointops
# from point
# from pointops.functions import knnquery
# from knn_cuda import KNN

# def knn_point(group_size, point_cloud, query_cloud, transpose_mode=False):
#     knn_obj = KNN(k=group_size, transpose_mode=transpose_mode)
#     dist, idx = knn_obj(point_cloud, query_cloud)
#     return dist, idx


class UpsampleLoss(nn.Module):
    def __init__(self, alpha=1.0, nn_size=5, radius=0.07, h=0.03, eps=1e-12):
        super().__init__()
        self.alpha = alpha
        self.nn_size = nn_size
        self.radius = radius
        self.h = h
        self.eps = eps
        self.chamfer_distance = chamfer_3DDist()

    def get_emd_loss(self, pred, gt, pcd_radius):
        idx, _ = AuctionMatch.apply(pred, gt)
        matched_out = pn2_utils.gather_operation(gt.transpose(1, 2).contiguous(), idx)
        matched_out = matched_out.transpose(1, 2).contiguous()
        dist2 = (pred - matched_out) ** 2
        dist2 = dist2.view(dist2.shape[0], -1) # <-- ???
        dist2 = torch.mean(dist2, dim=1, keepdims=True) # B,
        dist2 /= pcd_radius
        return torch.mean(dist2)

    def get_cd_loss(self, pred, gt, pcd_radius):
        cost_for, cost_bac = self.chamfer_distance(gt, pred)
        cost = 0.8 * cost_for + 0.2 * cost_bac
        cost /= pcd_radius
        cost = torch.mean(cost)
        return cost

    def get_repulsion_loss(self, pred):
        _, idx, _ = knn.knn_points(pred, pred, K=self.nn_size)
        # _, idx = knn_point(self.nn_size, pred, pred, transpose_mode=True)
        # _, idx = pn2_utils.knn
        idx = idx[:, :, 1:].to(torch.int32) # remove first one
        idx = idx.contiguous() # B, N, nn

        pred = pred.transpose(1, 2).contiguous() # B, 3, N
        grouped_points = pn2_utils.grouping_operation(pred, idx) # (B, 3, N), (B, N, nn) => (B, 3, N, nn)

        grouped_points = grouped_points - pred.unsqueeze(-1)
        dist2 = torch.sum(grouped_points ** 2, dim=1)
        dist2 = torch.max(dist2, torch.tensor(self.eps).cuda())
        dist = torch.sqrt(dist2)
        weight = torch.exp(- dist2 / self.h ** 2)

        uniform_loss = torch.mean((self.radius - dist) * weight)
        # uniform_loss = torch.mean(self.radius - dist * weight) # punet
        return uniform_loss

    def forward(self, pred, gt, pcd_radius):
        return self.get_emd_loss(pred, gt, pcd_radius) * 100, \
            self.alpha * self.get_repulsion_loss(pred)


def normalize_point_cloud(input, centroid=None, furthest_distance=None):
    # input: (b, 3, n) tensor

    if centroid is None:
        # (b, 3, 1)
        centroid = torch.mean(input, dim=-1, keepdim=True)
    # (b, 3, n)
    input = input - centroid
    if furthest_distance is None:
        # (b, 3, n) -> (b, 1, n) -> (b, 1, 1)
        furthest_distance = torch.max(torch.norm(input, p=2, dim=1, keepdim=True), dim=-1, keepdim=True)[0]
    input = input / furthest_distance

    return input, centroid, furthest_distance

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()

def index_points(pts, idx):
    """
    Input:
        pts: input points data, [B, C, N]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, C, S, [K]]
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = pts.shape[1]
    reshape = False
    if len(idx.shape) == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)
    # (b, c, (s k))
    res = torch.gather(pts, 2, idx[:, None].repeat(1, fdim, 1))
    if reshape:
        res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)

    return res

def get_knn_pts(k, pts, center_pts, return_idx=False):
    # input: (b, 3, n)
    # 1，GT，sample
    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, m, 3)
    center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()
    # (b, m, k)
    knn_idx = knnquery_heap(k, pts_trans, center_pts_trans).long()
    # (b, 3, m, k)
    knn_pts = index_points(pts, knn_idx)

    if return_idx == False:
        return knn_pts
    else:
        return knn_pts, knn_idx

def FPS(pts, fps_pts_num):
    # input: (b, 3, n)

    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, fps_pts_num)
    sample_idx = furthestsampling(pts_trans, fps_pts_num).long()
    # (b, 3, fps_pts_num)
    sample_pts = index_points(pts, sample_idx)

    return sample_pts

def midpoint_interpolate(sparse_pts, up_rate=4, normal=False, ):
    # sparse_pts: (b, 3, 256)

    if (normal):
        sparse_pts, centroid, furthest_distance = normalize_point_cloud(sparse_pts)

    pts_num = sparse_pts.shape[-1]
    up_pts_num = int(pts_num * up_rate)
    k = int(2 * up_rate)
    # (b, 3, n, k)
    knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
    # (b, 3, n, k)
    repeat_pts = repeat(sparse_pts, 'b c n -> b c n k', k=k)
    # (b, 3, n, k)
    mid_pts = (knn_pts + repeat_pts) / 2.0
    # (b, 3, (n k))
    mid_pts = rearrange(mid_pts, 'b c n k -> b c (n k)')
    # note that interpolated_pts already contain sparse_pts
    interpolated_pts = mid_pts
    # fps: (b, 3, up_pts_num)
    interpolated_pts = FPS(interpolated_pts, up_pts_num)

    if (normal):
        interpolated_pts = centroid + interpolated_pts * furthest_distance

    return interpolated_pts


def training_loss(
        net,
        loss_fn,
        x0,
        diffusion_hyperparams,
        label=None,
        condition=None,
        alpha=1.0,
        gamma=None
):
    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
    B, N, D = x0.shape
    diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda()  # t ~ U[T]
    z = std_normal(x0.shape)
    # xt = sqrt(at_) * X0 + sqrt(1-at_) * z ==> q(xt|x0)
    xt = torch.sqrt(Alpha_bar[diffusion_steps]) * x0 + torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z
    i = midpoint_interpolate(condition.permute(0, 2, 1)).permute(0, 2, 1)
    xt = torch.cat([xt, i], dim=-1)
    epsilon_theta = net(
        xt,
        condition,
        ts=diffusion_steps.view(B, ),
        label=label
    )

    if (isinstance(epsilon_theta, tuple)):
        noisy, condition_pre = epsilon_theta
        mse_theta = loss_fn(noisy, z)
        mse_psi = loss_fn(condition_pre, condition)
        loss = mse_theta + alpha * mse_psi
    else:
        loss = loss_fn(epsilon_theta, z)

    return loss

class lpips_loss_fn():
    def __init__(self):
        import lpips
        self.lpips = lpips.LPIPS(net='vgg').cuda()
        self.lpips.eval()
        for param in self.lpips.parameters():
            param.requires_grad = False

    def __call__(self, x, y):
        # x  B,H,W,C [0,1]
        # y  B,H,W,C [0,1]
        loss = self.lpips(x.permute(0,3,1,2), y.permute(0,3,1,2), normalize=True)#.mean()
        return loss