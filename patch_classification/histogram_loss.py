import torch
from torch import nn, sigmoid
from autoencoder_networks import rgb2hsv


def rgb2yuv(rgb, device):
    yuv_from_rgb = torch.transpose(torch.Tensor([[0.299, 0.587, 0.114],
                                [-0.14714119, -0.28886916, 0.43601035],
                                [0.61497538, -0.51496512, -0.10001026]]).to(device), 0, 1)
    rgb = rgb.permute(0, 2, 3, 1)
    yuv = torch.matmul(rgb, yuv_from_rgb)
    yuv = yuv.permute(0, 3, 1, 2)
    return yuv

def yuv2rgb(yuv, device):
    rgb_from_yuv = torch.transpose(torch.Tensor([[1, 0, 1.1398],
                                [1, -0.39465, -0.5806],
                                [1, 2.03211, 0]]).to(device), 0, 1)
    yuv = yuv.permute(0, 2, 3, 1)
    rgb = torch.matmul(yuv, rgb_from_yuv)
    rgb = rgb.permute(0, 3, 1, 2)
    rgb[rgb < 0] = 0
    return rgb


def compute_histogram_loss(output_ae, target_ae, device, color_space="yuv", g_act="sigmoid"):
    # Parameters
    N = output_ae.size(2) * output_ae.size(3)
    K = 256
    L = 1 / K  # 2 / K -> if values in [-1,1] (Paper)
    W = L / 2.5
    mu_k = (L * (torch.arange(K) + 0.5)).view(-1, 1)
    mu_k = mu_k.to(device)

    # convert to range (0, 1)
    if g_act == "tanh":
        output_ae = (output_ae / 2) + 0.5
        target_ae = (target_ae / 2) + 0.5

    # convert to hsv / yuv
    if color_space == "hsv" or color_space == "hs":
        output_ae = rgb2hsv(output_ae)
        target_ae = rgb2hsv(target_ae)
        # rescale h from range (0, 360) to (0, 1)
        output_ae[:, 0, ::] = output_ae[:, 0, ::] / 360
        target_ae[:, 0, ::] = output_ae[:, 0, ::] / 360
    elif color_space == "yuv" or color_space == "uv":
        output_ae = rgb2yuv(output_ae, device)
        target_ae = rgb2yuv(target_ae, device)
        # rescale u from range (-u_max, u_max) to (0, 1)
        u_max = 0.436
        output_ae[:, 1, ::] = (output_ae[:, 1, ::] / u_max / 2) + 0.5
        target_ae[:, 1, ::] = (target_ae[:, 1, ::] / u_max / 2) + 0.5
        # rescale v from range (-v_max, v_max) to (0, 1)
        v_max = 0.615
        output_ae[:, 2, ::] = (output_ae[:, 2, ::] / v_max / 2) + 0.5
        target_ae[:, 2, ::] = (target_ae[:, 2, ::] / v_max / 2) + 0.5

    # output histograms
    output_act_1 = compute_pj(output_ae[:, 0, ::], mu_k, K, L, W) # B x 256 x N
    output_hist_1 = output_act_1.sum(dim=2)/N # B x 256
    output_act_2 = compute_pj(output_ae[:, 1, ::], mu_k, K, L, W) # B x 256 x N
    output_hist_2 = output_act_2.sum(dim=2)/N # B x 256
    output_act_3 = compute_pj(output_ae[:, 2, ::], mu_k, K, L, W) # B x 256 x N
    output_hist_3 = output_act_3.sum(dim=2)/N # B x 256

    # target histograms
    target_act_1 = compute_pj(target_ae[:, 0, ::], mu_k, K, L, W)  # B x 256 x N
    target_hist_1 = target_act_1.sum(dim=2) / N  # B x 256
    target_act_2 = compute_pj(target_ae[:, 1, ::], mu_k, K, L, W)  # B x 256 x N
    target_hist_2 = target_act_2.sum(dim=2) / N  # B x 256
    target_act_3 = compute_pj(target_ae[:, 2, ::], mu_k, K, L, W)  # B x 256 x N
    target_hist_3 = target_act_3.sum(dim=2) / N  # B x 256

    # joint histograms
    joint_hist_1 = torch.matmul(output_act_1, torch.transpose(target_act_1, 1, 2)) / N # B x 256 x 256
    joint_hist_2 = torch.matmul(output_act_2, torch.transpose(target_act_2, 1, 2)) / N  # B x 256 x 256
    joint_hist_3 = torch.matmul(output_act_3, torch.transpose(target_act_3, 1, 2)) / N  # B x 256 x 256

    # Earth Movers Distance Loss
    emd_1 = EarthMoversDistanceLoss(device=device)(output_hist_1, target_hist_1)
    emd_2 = EarthMoversDistanceLoss(device=device)(output_hist_2, target_hist_2)
    emd_3 = EarthMoversDistanceLoss(device=device)(output_hist_3, target_hist_3)

    if color_space == "uv":
        emd_cat = torch.stack((emd_2, emd_3), dim=1)
    elif color_space == "hs":
        emd_cat = torch.stack((emd_1, emd_2), dim=1)
    else:
        emd_cat = torch.stack((emd_1, emd_2, emd_3), dim=1)

    emd = torch.mean(emd_cat)


    # Mutual Information Loss
    mi_1 = MutualInformationLoss()(output_hist_1, target_hist_1, joint_hist_1)
    mi_2 = MutualInformationLoss()(output_hist_2, target_hist_2, joint_hist_2)
    mi_3 = MutualInformationLoss()(output_hist_3, target_hist_3, joint_hist_3)

    if color_space == "uv":
        mi_cat = torch.stack((mi_2, mi_3), dim=1)
    elif color_space == "hs":
        mi_cat = torch.stack((mi_1, mi_2), dim=1)
    else:
        mi_cat = torch.stack((mi_1, mi_2, mi_3), dim=1)

    mi = torch.mean(mi_cat)

    return emd, mi




class EarthMoversDistanceLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, x, y):
        # input has dims: (Batch x Bins)
        bins = x.size(1)
        r = torch.arange(bins)
        s, t = torch.meshgrid(r, r)
        tt = t >= s
        tt = tt.to(self.device)

        cdf_x = torch.matmul(x, tt.float())
        cdf_y = torch.matmul(y, tt.float())

        return torch.sum(torch.square(cdf_x - cdf_y), dim=1)


class MutualInformationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p1, p2, p12):
        # input p12 has dims: (Batch x Bins x Bins)
        # input p1 & p2 has dims: (Batch x Bins)

        product_p = torch.matmul(torch.transpose(p1.unsqueeze(1), 1, 2), p2.unsqueeze(1)) + torch.finfo(p1.dtype).eps
        mi = torch.sum(p12 * torch.log(p12 / product_p + torch.finfo(p1.dtype).eps), dim=(1, 2))
        h = -torch.sum(p12 * torch.log(p12 + torch.finfo(p1.dtype).eps), dim=(1, 2))

        return 1 - (mi / h)

def phi_k(x, L, W):
    return sigmoid((x + (L / 2)) / W) - sigmoid((x - (L / 2)) / W)


def compute_pj(x, mu_k, K, L, W):
    # we assume that x has only one channel already
    # flatten spatial dims
    x = x.reshape(x.size(0), 1, -1)
    x = x.repeat(1, K, 1)  # construct K channels

    # apply activation functions
    return phi_k(x - mu_k, L, W)
