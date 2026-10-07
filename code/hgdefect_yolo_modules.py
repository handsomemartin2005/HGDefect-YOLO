"""HGDefect-YOLO modules: PPAP-EMA, KDE hypergraph fusion, and RGCU-CLAG."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, c1, c2, k=1):
        super().__init__(nn.Conv2d(c1, c2, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(c2), nn.SiLU())


class PPAP(nn.Module):
    def __init__(self, ratio=0.05):
        super().__init__()
        if not 0 < ratio <= 1:
            raise ValueError('ratio must be in (0, 1]')
        self.ratio = ratio

    def forward(self, x):
        k = max(1, math.floor(self.ratio * x.shape[-2] * x.shape[-1]))
        return x.flatten(2).topk(k, dim=-1, sorted=False).values.mean(-1)[..., None, None]


class PPAPEMA(nn.Module):
    def __init__(self, channels, groups=8, ratio=0.05, gn_groups=1):
        super().__init__()
        if channels % groups or (channels // groups) % gn_groups:
            raise ValueError('channels and group counts must be divisible')
        self.groups = groups
        c = channels // groups
        self.ppap = PPAP(ratio)
        self.axial = nn.Conv2d(c, c, 1)
        self.local = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(gn_groups, c)

    def forward(self, x):
        b, c, h, w = x.shape
        grouped = x.reshape(b * self.groups, c // self.groups, h, w)
        directional = torch.cat((grouped.mean(2, keepdim=True),
                                 grouped.mean(3, keepdim=True).transpose(2, 3)), 3)
        aw, ah = self.axial(directional).split((w, h), 3)
        normalized = self.norm(grouped * aw.sigmoid() * ah.transpose(2, 3).sigmoid())
        local = self.local(grouped)
        q1 = self.ppap(normalized).flatten(1).softmax(1).unsqueeze(1)
        q2 = self.ppap(local).flatten(1).softmax(1).unsqueeze(1)
        attention = (q1 @ local.flatten(2) + q2 @ normalized.flatten(2)).reshape(-1, 1, h, w).sigmoid()
        # Recalibrate the grouped input using cross-spatial attention.
        return (grouped * attention).reshape(b, c, h, w)


def soft_prototypes(features, count=48, iterations=3, temperature=0.1):
    count = min(count, features.shape[1])
    indices = torch.linspace(0, features.shape[1] - 1, count, device=features.device).long()
    centers = F.normalize(features[:, indices], dim=-1)
    normalized = F.normalize(features, dim=-1)
    for _ in range(iterations):
        assignment = (normalized @ centers.transpose(1, 2) / temperature).softmax(-1)
        centers = F.normalize(assignment.transpose(1, 2) @ features /
                              assignment.sum(1).unsqueeze(-1).clamp_min(1e-6), dim=-1)
    return assignment


class KDEHyperGraphConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.theta = nn.Linear(channels, channels, bias=False)

    @staticmethod
    def incidence(nodes):
        # H[vertex, edge]; each edge contains its center and higher-density vertices.
        m = nodes.shape[1]
        eye = torch.eye(m, device=nodes.device, dtype=torch.bool)[None]
        distance = torch.cdist(nodes, nodes)
        if m == 1:
            bandwidth = torch.ones_like(distance[:, :1, :1])
        else:
            bandwidth = (distance.masked_fill(eye, 0).sum((1, 2)) / (m * (m - 1))).clamp_min(1e-6)[:, None, None]
        density = torch.exp(-distance.square() / (2 * bandwidth.square())).sum(-1)
        return ((density[:, :, None] > density[:, None, :]) | eye).to(nodes.dtype)

    def forward(self, nodes):
        h = self.incidence(nodes)
        dv = h.sum(2, keepdim=True).clamp_min(1)
        de = h.sum(1).unsqueeze(-1).clamp_min(1)
        edges = (h.transpose(1, 2) @ self.theta(nodes)) / de
        return nodes + (h @ edges) / dv


class KDEHyperGraphFusion(nn.Module):
    def __init__(self, in_channels, out_channels, target_pos, num_nodes=48, temperature=0.1):
        super().__init__()
        self.target_pos, self.num_nodes, self.temperature = target_pos, num_nodes, temperature
        branch = max(out_channels // 2, 16)
        self.project = nn.ModuleList(ConvBNAct(c, branch) for c in in_channels)
        self.mix = ConvBNAct(branch * len(in_channels), out_channels)
        self.hyper = KDEHyperGraphConv(out_channels)
        self.refine = ConvBNAct(out_channels, out_channels, 3)

    def forward(self, features):
        size = features[self.target_pos].shape[-2:]
        aligned = []
        for x, project in zip(features, self.project):
            x = project(x)
            if x.shape[-2:] != size:
                x = F.adaptive_avg_pool2d(x, size) if any(a > b for a, b in zip(x.shape[-2:], size)) else F.interpolate(x, size=size, mode='nearest')
            aligned.append(x)
        mixed = self.mix(torch.cat(aligned, 1))
        b, c, h, w = mixed.shape
        tokens = mixed.flatten(2).transpose(1, 2)
        assignment = soft_prototypes(tokens, self.num_nodes, temperature=self.temperature)
        nodes = (assignment.transpose(1, 2) @ tokens) / assignment.sum(1).unsqueeze(-1).clamp_min(1e-6)
        updated = self.hyper(nodes)
        scattered = (assignment @ updated).transpose(1, 2).reshape(b, c, h, w)
        return self.refine(scattered)


class RGCU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.baseline = nn.Conv2d(in_channels, out_channels, 1)
        # C2D: stride-2 transposed convolution.
        self.c2d = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
        self.reconstruct = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.spatial = nn.Conv2d(2, 1, 3, padding=1)
        hidden = max(1, out_channels // 4)
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channels * 2, hidden, 1),
                                     nn.SiLU(), nn.Conv2d(hidden, out_channels, 1), nn.Sigmoid())

    @staticmethod
    def blend(baseline, reconstruction, gate):
        return (1 - gate) * baseline + gate * reconstruction

    def components(self, deep, size):
        baseline = self.baseline(F.interpolate(deep, size=size, mode='nearest'))
        # A stride-2 transposed convolution supports output 2*n-1 or 2*n.
        if any(s not in (2 * n - 1, 2 * n) for n, s in zip(deep.shape[-2:], size)):
            raise ValueError('RGCU expects adjacent feature levels with scale factor two')
        reconstructed = self.reconstruct(self.c2d(deep, output_size=(deep.shape[0], self.c2d.out_channels, *size)))
        gs = self.spatial(torch.cat((baseline.mean(1, keepdim=True), reconstructed.mean(1, keepdim=True)), 1)).sigmoid()
        gc = self.channel(torch.cat((baseline, reconstructed), 1))
        discrepancy = (baseline - reconstructed).abs().mean(1, keepdim=True)
        scale = (baseline.abs() + reconstructed.abs()).mean(1, keepdim=True).clamp_min(1e-6)
        gm = torch.exp(-discrepancy / scale)
        return baseline, reconstructed, gs, gc, gm

    def forward(self, deep, size):
        baseline, reconstructed, gs, gc, gm = self.components(deep, size)
        return self.blend(baseline, reconstructed, gs * gc * gm)


class CLAG(nn.Module):
    def __init__(self, shallow_channels, hidden_channels):
        super().__init__()
        self.attention = nn.Sequential(nn.Conv2d(shallow_channels, hidden_channels, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(hidden_channels), nn.SiLU(),
                                       nn.Conv2d(hidden_channels, 1, 1), nn.Sigmoid())

    def forward(self, shallow, upsampled):
        return self.attention(shallow) * upsampled


class CLAGRGCUFuse(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        deep_channels, shallow_channels = in_channels
        self.rgcu = RGCU(deep_channels, out_channels)
        self.clag = CLAG(shallow_channels, out_channels)
        self.fuse = ConvBNAct(shallow_channels + out_channels, out_channels)

    def forward(self, features):
        deep, shallow = features
        upsampled = self.rgcu(deep, shallow.shape[-2:])
        guided = self.clag(shallow, upsampled)
        # The model YAML supplies the following C2f refinement block (Phi).
        return self.fuse(torch.cat((shallow, guided), 1))
