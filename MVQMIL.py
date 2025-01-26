import torch
from torch import nn
import numpy as np
from datten import DAttention
from timm.models.layers import DropPath
from kmeansplusplus import kmeans


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def region_partition(x, region_size):
    """
    Args:
        x: (B, H, W, C)
        region_size (int): region size
    Returns:
        regions: (num_regions*B, region_size, region_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // region_size, region_size, W // region_size, region_size, C)
    regions = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, region_size, region_size, C)
    return regions


def region_reverse(regions, region_size, H, W):
    """
    Args:
        regions: (num_regions*B, region_size, region_size, C)
        region_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(regions.shape[0] / (H * W / region_size / region_size))
    x = regions.view(B, H // region_size, W // region_size, region_size, region_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def padding(x, region_num):
    B, L, C = x.shape
    H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
    _n = -H % region_num
    H, W = H + _n, W + _n
    region_size = int(H // region_num)
    add_length = H * W - L
    if add_length > 0:
        x = torch.cat([x, torch.zeros((B, add_length, C), device=x.device)], dim=1)

    return x, H, W, add_length, region_num, region_size


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionWithVQ(nn.Module):
    def __init__(self, dim, head, qkv_bias, attn_drop, proj_drop, vqe, vqe_k, vqe_bias, vqe_drop):
        super().__init__()
        self.dim = dim
        self.head = head
        self.head_dim = dim // head
        self.scale = self.head_dim ** -0.5
        self.vqe = vqe

        self.qkv = nn.Linear(dim, head * self.head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head * self.head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if vqe:
            self.pe1 = nn.Conv2d(head, head, (vqe_k, 1), padding=(vqe_k // 2, 0), groups=head, bias=vqe_bias)
            self.pe2 = nn.Conv2d(head, head, (vqe_k, 1), padding=(vqe_k // 2, 0), groups=head, bias=vqe_bias)
        else:
            self.pe1 = nn.Identity()
            self.pe2 = nn.Identity()
        self.vqe_drop = nn.Dropout(vqe_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, m=None, s=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale

        if self.vqe:
            attn = (q @ k.transpose(-2, -1)) * 0.5
            m = m.reshape(B, N, self.head, self.head_dim).permute(0, 2, 1, 3)
            s = s.reshape(B, N, self.head, self.head_dim).permute(0, 2, 1, 3)
            pe = self.pe1(m) * self.scale @ self.pe2(s).transpose(-2, -1) * 0.5
            pe = self.vqe_drop(pe)
            attn = attn + pe
        else:
            attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.head * self.head_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class RegionAttention(nn.Module):
    def __init__(self, dim, head, qkv_bias, attn_drop, proj_drop, vqe, vqe_k, vqe_bias, vqe_drop, region_num):
        super().__init__()

        self.dim = dim
        self.head = head
        self.region_num = region_num
        self.vqe = vqe
        self.attn = AttentionWithVQ(dim, head, qkv_bias, attn_drop, proj_drop, vqe, vqe_k, vqe_bias, vqe_drop)

    def forward(self, x, m=None, s=None, return_attn=False):
        B, L, C = x.shape

        # padding
        x, H, W, add_length, region_num, region_size = padding(x, self.region_num)

        x = x.view(B, H, W, C)

        # partition regions
        x_regions = region_partition(x, region_size)  # nW*B, region_size, region_size, C

        x_regions = x_regions.view(-1, region_size * region_size, C)  # nW*B, region_size*region_size, C

        if self.vqe:
            # padding
            m, _, _, _, _, _ = padding(m, self.region_num)
            s, _, _, _, _, _ = padding(s, self.region_num)
            # partition regions
            m_regions = region_partition(m.view(B, H, W, C), region_size).view(-1, region_size * region_size, C)
            s_regions = region_partition(s.view(B, H, W, C), region_size).view(-1, region_size * region_size, C)

        else:
            m_regions, s_regions = None, None

        # R-MSA
        attn_regions = self.attn(x_regions, m_regions, s_regions)  # nW*B, region_size*region_size, C

        # merge regions
        attn_regions = attn_regions.view(-1, region_size, region_size, C)
        x = region_reverse(attn_regions, region_size, H, W)  # B H' W' C
        x = x.view(B, H * W, C)

        if add_length > 0:
            x = x[:, :-add_length]

        return x


class TransLayer(nn.Module):
    def __init__(self, dim, head, qkv_bias, attn_drop, proj_drop, vqe, vqe_k, vqe_bias, vqe_drop, region_num,
                 ffn, ffn_act, mlp_ratio, drop_path):
        super().__init__()
        self.vqe = vqe

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim) if vqe else nn.Identity()
        self.norm3 = nn.LayerNorm(dim) if vqe else nn.Identity()
        self.norm4 = nn.LayerNorm(dim) if ffn else nn.Identity()

        self.attn = RegionAttention(dim, head, qkv_bias, attn_drop, proj_drop, vqe, vqe_k, vqe_bias, vqe_drop,
                                    region_num)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = ffn
        act_layer = nn.GELU if ffn_act == 'gelu' else nn.ReLU
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=proj_drop) if ffn else nn.Identity()

    def forward_trans(self, x, m=None, s=None, need_attn=False):
        attn = None

        if self.vqe:
            m = self.norm2(m)
            s = self.norm3(s)

        if need_attn:
            z, attn = self.attn(self.norm1(x), m, s, return_attn=need_attn)
        else:
            z = self.attn(self.norm1(x), m, s)

        x = x + self.drop_path(z)

        # FFN
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm4(x)))

        return x, attn

    def forward(self, x, m=None, s=None, need_attn=False):

        x, attn = self.forward_trans(x, m, s, need_attn=need_attn)

        if need_attn:
            return x, attn
        else:
            return x

class MultiScaleCodebook(nn.Module):
    def __init__(self, d_k, n_code, beta, k, loss_mode, n_smat):
        super(MultiScaleCodebook, self).__init__()
        self.c = nn.Parameter(torch.zeros((n_code, d_k)))
        self.beta = beta
        self.k = k
        self.cos = nn.CosineEmbeddingLoss(reduction='mean') if loss_mode == 'cos' else nn.Identity()
        self.mse = nn.MSELoss(reduction='mean') if loss_mode == 'mse' else nn.Identity()
        self.n_code = n_code
        self.loss_mode = loss_mode
        self.n_smat = n_smat
        trans_mat = []
        for i in range(1, 1 + n_smat):
            trans_mat += [nn.Linear(n_code, n_code // (2 ** i))]
        self.trans = nn.Sequential(*trans_mat)

    def vq_loss(self, z_q, x):
        if self.loss_mode == 'mse':
            loss = self.beta[0] * self.mse(z_q, x.detach()) + self.beta[1] * self.mse(x, z_q.detach())
        elif self.loss_mode == 'cos':
            loss = self.beta[0] * self.cos(z_q, x.detach(), torch.ones([x.shape[0]]).to(x.device)) \
                   + self.beta[1] * self.cos(x, z_q.detach(), torch.ones([x.shape[0]]).to(x.device))
        else:
            e_latent_loss = torch.nn.functional.mse_loss(z_q, x.detach())
            q_latent_loss = torch.nn.functional.mse_loss(x, z_q.detach())
            loss = self.beta[0] * e_latent_loss + self.beta[1] * q_latent_loss
        return loss

    @staticmethod
    def get_key(z, c, n):
        unique_elements, counts = torch.unique(z, return_counts=True)
        top_counts, top_indices = torch.topk(counts, k=min(n, len(unique_elements)), largest=True)
        top_keys = unique_elements[top_indices]
        if len(top_keys) < n:
            repeats = n // len(top_keys) + 1
            top_keys = top_keys.repeat(repeats)[:n]
        key_instances = c[top_keys]
        return key_instances.detach()

    def forward(self, x, training, get_k):
        x = x.squeeze(0)

        main_c = self.c
        main_z = torch.argmin(torch.cdist(x, main_c), dim=1)
        main_z_q = main_c[main_z]
        if get_k:
            main_k_i = self.get_key(main_z, main_c, self.k)
        else:
            main_k_i = torch.zeros(self.k, x.shape[1]).to(x.device)

        sub_z_q = torch.zeros_like(main_z_q).to(x.device)
        sub_k_i = torch.zeros_like(main_k_i).to(x.device)
        ms_k_i = torch.zeros_like(main_k_i).to(x.device)
        if self.n_smat > 0:
            for i, L in enumerate(self.trans.children()):
                subsig_c = L(main_c.clone().T).T
                subsig_z = torch.argmin(torch.cdist(x, subsig_c), dim=1)
                subsig_z_q = subsig_c[subsig_z]
                sub_z_q += subsig_z_q
                if get_k:
                    sub_k_i += self.get_key(subsig_z, subsig_c, self.k)
            sub_z_q = (self.n_smat ** -1) * sub_z_q
            if get_k:
                sub_k_i = (self.n_smat ** -1) * sub_k_i
                ms_k_i = main_k_i + sub_k_i
            if training:
                total_loss = self.vq_loss(main_z_q, x) + self.vq_loss(sub_z_q, x)
            else:
                total_loss = torch.zeros([])
        else:
            if get_k:
                ms_k_i = main_k_i
            if training:
                total_loss = self.vq_loss(main_z_q, x)
            else:
                total_loss = torch.zeros([])

        flattened_matrix = main_z_q.view(main_z_q.size(0), -1)
        unique_rows = torch.unique(flattened_matrix, dim=0).size(0)

        main_z_q = (x + (main_z_q - x).detach()).unsqueeze(0)
        sub_z_q = (x + (sub_z_q - x).detach()).unsqueeze(0)

        return main_z_q, sub_z_q, total_loss, ms_k_i, unique_rows

    def initialize_with_kmeans(self, data, random_state):
        # 使用KMeans聚类对数据进行聚类
        data = data.squeeze(0).detach()

        _, _, centroids = kmeans(data, n_clusters=self.n_code, random_state=random_state, device=data.device)
        self.c.data = centroids


class RankBoostModule(nn.Module):
    def __init__(self, dimensions, region_num):
        super(RankBoostModule, self).__init__()
        self.dimensions = dimensions
        self.region_num = region_num
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Tanh()
        self.spatial1 = nn.Conv2d(dimensions, dimensions, (1, 1), (1, 1), 1 // 2, groups=dimensions)
        self.spatial2 = nn.Conv2d(dimensions, dimensions, (3, 3), (1, 1), 3 // 2, groups=dimensions)
        self.spatial3 = nn.Conv2d(dimensions, dimensions, (5, 5), (1, 1), 5 // 2, groups=dimensions)
        self.channel1 = nn.Conv2d(region_num ** 2, region_num ** 2, (1, 1), (1, 1), 1 // 2, groups=region_num ** 2)
        self.channel2 = nn.Conv2d(region_num ** 2, region_num ** 2, (3, 3), (1, 1), 3 // 2, groups=region_num ** 2)
        self.channel3 = nn.Conv2d(region_num ** 2, region_num ** 2, (5, 5), (1, 1), 5 // 2, groups=region_num ** 2)
        self.s_weight = nn.Conv2d(dimensions, dimensions, (1, 1), groups=dimensions)
        self.c_weight = nn.Conv2d(region_num ** 2, region_num ** 2, (1, 1), groups=region_num ** 2)

    def forward(self, x):
        B, L, C = x.shape
        x, H, W, add_length, region_num, region_size = padding(x, self.region_num)
        x = x.view(B, H, W, C)
        os = region_partition(x, region_size).reshape(-1, self.dimensions, region_size, region_size)
        oc = region_partition(x, region_size).reshape(self.dimensions, -1, region_size, region_size)

        s_weight = os * self.s_weight(os)
        s = self.spatial1(os) + self.spatial2(os) + self.spatial3(os)
        s = s * s_weight

        c_weight = oc * self.c_weight(oc)
        c = self.channel1(oc) + self.channel2(oc) + self.channel3(oc)
        c = (c * c_weight).permute(1, 0, 2, 3)

        sc = (self.act1(s) * self.act2(c)).permute(0, 2, 3, 1)
        x = region_reverse(sc, region_size, H, W).view(B, H * W, C)

        if add_length > 0:
            x = x[:, :L, :]
        return x


class Aggregation(nn.Module):
    def __init__(self, dim, n_code, beta, k, loss_mode, n_smat, head, qkv_bias, attn_drop, proj_drop,
                 vqe, vqe_k, vqe_bias, vqe_drop, region_num, ffn, ffn_act, mlp_ratio, drop_path, all_shortcut):
        super(Aggregation, self).__init__()
        self.all_shortcut = all_shortcut

        self.ATTN_VQPE = TransLayer(dim=dim, head=head, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop,
                                    vqe=vqe, vqe_k=vqe_k, vqe_bias=vqe_bias, vqe_drop=vqe_drop, region_num=region_num,
                                    ffn=ffn, ffn_act=ffn_act, mlp_ratio=mlp_ratio, drop_path=drop_path)
        self.RBM = RankBoostModule(dim, region_num)
        self.MSCB = MultiScaleCodebook(dim, n_code, beta, k, loss_mode, n_smat)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, training, key_ins):
        x_shortcut = x

        # vq_loss, k, r = 0., None, 0.
        xm, xs, vq_loss, k, r = self.MSCB(x, training, key_ins)

        x = self.ATTN_VQPE(x, xm, xs) + self.RBM(xm + xs)

        if self.all_shortcut:
            x = x + x_shortcut

        x = self.norm(x)

        return x, vq_loss, k, r


class MIL(nn.Module):
    def __init__(self, input_dim=1024, dim=512, n_classes=2, n_code=256, beta=tuple([1., 0.3]), k=4, loss_mode='mse',
                 n_smat=4, head=8, qkv_bias=True, attn_drop=0.25, proj_drop=0.25, vqe=True, vqe_k=9, vqe_bias=True,
                 vqe_drop=0.1, region_num=8, ffn=False, global_act='relu', mlp_ratio=4., drop_path=0., all_shortcut=True,
                 da_act='tanh', da_gated=True, da_bias=False, da_dropout=False, key_ins=True, key_coe=0.6):
        super(MIL, self).__init__()
        self.key_ins = key_ins
        self.key_coe = key_coe
        self.patch_to_emb = [nn.Linear(input_dim, dim)]

        if global_act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif global_act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]
        self.dp = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()

        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        self.aggregation = Aggregation(dim, n_code, beta, k, loss_mode, n_smat, head, qkv_bias, attn_drop, proj_drop,
                                       vqe, vqe_k, vqe_bias, vqe_drop, region_num, ffn, global_act, mlp_ratio,
                                       drop_path,
                                       all_shortcut)

        self.pool_fn = DAttention(dim, da_act, gated=da_gated, bias=da_bias, dropout=da_dropout)
        self.keyin_to_emb = DAttention(dim, da_act, gated=da_gated, bias=da_bias,
                                       dropout=da_dropout) if key_ins else nn.Identity()

        self.predictor = nn.Linear(dim, n_classes)
        self.kinspreor = nn.Linear(dim, n_classes) if key_ins else nn.Identity()

        self.apply(initialize_weights)

    def forward(self, x, training=False, return_attn=False, no_norm=False):
        rows = x.shape[1]
        x = self.patch_to_emb(x)  # n*512
        x = self.dp(x)

        # feature re-embedding
        x, vq_loss, k, r = self.aggregation(x, training, self.key_ins)

        # feature aggregation
        if return_attn:
            x, a = self.pool_fn(x)
        else:
            x = self.pool_fn(x)

        # prediction
        if self.key_ins and k is not None:
            k = self.keyin_to_emb(k.unsqueeze(0))
            logits = self.predictor(x) + self.key_coe * self.kinspreor(k)
        else:
            logits = self.predictor(x)

        return logits, vq_loss, r


if __name__ == "__main__":
    x = torch.rand(1, 1000, 1024).cuda()

    mil = MIL(input_dim=1024, dim=512, n_classes=2, n_code=256, beta=tuple([1., 0.3]), k=4, loss_mode='mse', n_smat=4,
              head=8, qkv_bias=True, attn_drop=0.25, proj_drop=0.25, vqe=True, vqe_k=9, vqe_bias=True, vqe_drop=0.1,
              region_num=8, ffn=False, global_act='relu', mlp_ratio=4., drop_path=0., all_shortcut=True,
              da_act='tanh', da_gated=True, da_bias=False, da_dropout=False, key_ins=True, key_coe=0.6).cuda()

    agg_input = mil.patch_to_emb(x)
    mil.aggregation.MSCB.initialize_with_kmeans(agg_input, random_state=2021)
    x = mil(x, training=True)

    print(x)