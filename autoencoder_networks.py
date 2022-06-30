import torch
from torch import nn, sigmoid
from collections import OrderedDict
from torch.nn import functional as F


class AeSegParam02(nn.Module):
    def __init__(self, c_seg=8, c_ae=2, c_param=1, mode="none", ratio=0.1, act="sigmoid"):
        super(AeSegParam02, self).__init__()
        self.act = act
        self.mode=mode
        self.ratio=ratio
        self.C_seg= c_seg
        self.C_ae = c_ae
        self.C_param = c_param
        # Encoder
        self.encoder = get_encoder(hidden_channels=self.C_seg+self.C_ae+self.C_param)
        # Decoder
        self.ae_decoder = get_decoder(hidden_channels=self.C_seg+self.C_ae, output_channels=3, act=self.act)
        self.seg_decoder = get_decoder(hidden_channels=self.C_seg, output_channels=2, act="sigmoid")
        # Param Extractor
        self.param_network_1 = get_linear(7*7*self.C_param, 18)

    def extract_color_sequence(self, features_param, ratio=0.1):
        params_dict = dict()
        features_param = torch.flatten(features_param, start_dim=1)
        total_params = self.param_network_1(features_param) # Bx18
        # relative interval lengths, adding up to one
        for idx, channel in enumerate(["h", "s", "v"]):
            params = total_params[:, idx*6:idx*6+6] # Bx6
            lengths = F.softmax(params[:, 0:3], dim=1) # Bx3
            l1 = lengths[:, 0]
            l2 = lengths[:, 1]
            l3 = lengths[:, 2]
            # interval starting points
            s1 = params[:, 3] # between 0 and 1
            s1 = s1*(1-ratio) # linear map to [0, 1-ratio]
            s2 = params[:, 4] # between 0 and 1
            s2 = s1 + l1*ratio + s2 * (1 - (l2+l3)*ratio - s1 - l1*ratio) # linear map to [s1+l1*ratio, 1-(l2+l3)*ratio]
            s3 = params[:, 5] # between 0 and 1
            s3 = s2 + l2*ratio + s3 * (1 - l3*ratio - s2 - l2*ratio) # linear map to [s2+l2*ratio, 1-l3*ratio]

            params_dict[f"{channel}_l1"] = l1
            params_dict[f"{channel}_l2"] = l2
            params_dict[f"{channel}_l3"] = l3
            params_dict[f"{channel}_s1"] = s1
            params_dict[f"{channel}_s2"] = s2
            params_dict[f"{channel}_s3"] = s3
            params_dict["ratio"] = ratio
        return params_dict


    def forward(self, x):
        # from https://github.com/pytorch/vision/blob/729178c7cb3984f1ac04dd89b1610e95abf5cb4a/torchvision/models/segmentation/_utils.py#L11
        result = OrderedDict()

        features = self.encoder(x) # Bx(C_seg+C_ae+C_param)x7x7

        # split features (some for ae only, some for segmentation, some for parameters)
        features_seg = features[:, 0:self.C_seg, :, :] # BxC_segx7x7
        if self.C_ae == 1:
            features_ae_only = features[:, self.C_seg, :, :].unsqueeze(dim=1) # Bx1x7x7
        else:
            features_ae_only = features[:, self.C_seg:self.C_seg+self.C_ae, :, :] # BxC_aex7x7
        features_param = features[:, self.C_seg+self.C_ae:self.C_seg+self.C_ae+self.C_param, :, :] # BxC_paramx7x7

        # re-concatenate s.t. seg features are only optimized from segmentation branch
        features_seg_clone = features_seg.clone().detach()
        features_ae = torch.cat((features_ae_only, features_seg_clone), dim=1) # BxC_seg+C_aex7x7

        # Decoders
        out_ae = self.ae_decoder(features_ae) # Bx3x224x224
        out_seg = self.seg_decoder(features_seg) # Bx2x224x224

        # apply color sequence parameters to output
        # extract color sequence parameters
        if not self.mode == "none":
            params = self.extract_color_sequence(features_param, ratio=self.ratio)
            out_seg_mask = out_seg.argmax(1).clone().detach()
            if self.act == "tanh":
                out_ae = (out_ae / 2) + 0.5 # rescale to range (0, 1)
            out_ae_hsv = rgb2hsv(out_ae)
            out_ae_hsv = limit_hsv_to_sequence2(out_ae_hsv, out_seg_mask, self.mode, params)
            out_ae = hsv2rgb(out_ae_hsv)
            if self.act == "tanh":
                out_ae = (out_ae - 0.5) * 2 # rescale to range (-1, 1)

        result["features"] = features
        result["features_seg"] = features_seg
        result["features_ae_only"] = features_ae_only
        result["features_ae"] = features_ae
        result["features_param"] = features_param
        result["out_aa"] = out_ae
        result["out_seg"] = out_seg
        return result

def rgb2hsv(input, epsilon=1e-10):
    # from https://www.linuxtut.com/en/20819a90872275811439/
    assert(input.shape[1] == 3)

    r, g, b = input[:, 0], input[:, 1], input[:, 2]
    max_rgb, argmax_rgb = input.max(1)
    min_rgb, argmin_rgb = input.min(1)

    max_min = max_rgb - min_rgb + epsilon

    h1 = 60.0 * (g - r) / max_min + 60.0
    h2 = 60.0 * (b - g) / max_min + 180.0
    h3 = 60.0 * (r - b) / max_min + 300.0

    h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
    # h[h==360] = 0 # added this for stability later on
    s = max_min / (max_rgb + epsilon)
    v = max_rgb

    return torch.stack((h, s, v), dim=1)

def hsv2rgb(input):
    # https://cs.stackexchange.com/questions/64549/convert-hsv-to-rgb-colors
    assert(input.shape[1] == 3)

    h, s, v = input[:, 0], input[:, 1], input[:, 2]
    h_ = (h - torch.floor(h / 360) * 360) / 60 # should be between [0, 5.9999], but can be 6.0 sometimes
    h_[h_>=6.0] = 5.999999 # therefore, clip to maximum of 5.999999
    c = s * v
    x = c * (1 - torch.abs(torch.fmod(h_, 2) - 1))

    zero = torch.zeros_like(c)
    y = torch.stack((
        torch.stack((c, x, zero), dim=1),
        torch.stack((x, c, zero), dim=1),
        torch.stack((zero, c, x), dim=1),
        torch.stack((zero, x, c), dim=1),
        torch.stack((x, zero, c), dim=1),
        torch.stack((c, zero, x), dim=1),
    ), dim=0)
    index = torch.repeat_interleave(torch.floor(h_).unsqueeze(1), 3, dim=1).unsqueeze(0).to(torch.long)
    g = y.gather(dim=0, index=index)
    vc = (v-c).unsqueeze(1).unsqueeze(0).repeat(1, 1, 3, 1, 1) # this was changed from original version
    rgb = (g+vc).squeeze(0)
    #rgb = (y.gather(dim=0, index=index) + (v - c)).squeeze(0) # old version, buggy
    return rgb

def limit_hsv_to_sequence(input_hsv, input_mask, mode, ratio, l1, l2, l3, s1, s2, s3):
    h = input_hsv[:, 0, ::] # Bx224x224
    l1 = l1.view(-1, 1, 1) * 360.0
    l1 = l1.repeat(1, 224, 224)
    l2 = l2.view(-1, 1, 1) * 360.0
    l2 = l2.repeat(1, 224, 224)
    l3 = l3.view(-1, 1, 1) * 360.0
    l3 = l3.repeat(1, 224, 224)
    s1 = s1.view(-1, 1, 1) * 360.0
    s1 = s1.repeat(1, 224, 224)
    s2 = s2.view(-1, 1, 1) * 360.0
    s2 = s2.repeat(1, 224, 224)
    s3 = s3.view(-1, 1, 1) * 360.0
    s3 = s3.repeat(1, 224, 224)

    if mode == "remap":
        # Remap colors inside mask to defined colorspace (three intervals)
        inter1 = torch.logical_and(torch.logical_and(h>=0, l1-h>0), input_mask>0)
        inter2 = torch.logical_and(torch.logical_and(h-l1>=0, l2+l1-h>0), input_mask>0)
        inter3 = torch.logical_and(torch.logical_and(h-l1-l2>=0, l3+l2+l1-h>0), input_mask>0)
        inter_all = torch.logical_or(torch.logical_or(inter1, inter2), inter3)
        h = torch.where(inter1, 1, 0) * (s1 + ratio * h) + \
            torch.where(inter2, 1, 0) * (s2 + ratio * (h - l1)) + \
            torch.where(inter3, 1, 0) * (s3 + ratio * (h - l1 - l2)) + \
            torch.where(inter_all, 0, 1) * h
    elif mode == "zero":
        # Set all colors inside mask that are outside of learned color space to zero
        inter1 = torch.logical_and(h-s1 >= 0, s1 + ratio * l1 - h > 0)
        inter2 = torch.logical_and(h-s2 >= 0, s2 + ratio * l2 - h > 0)
        inter3 = torch.logical_and(h-s3 >= 0, s3 + ratio * l3 - h > 0)
        not_inter_all = torch.logical_not(torch.logical_or(torch.logical_or(inter1, inter2), inter3))
        condition = torch.logical_and(not_inter_all, input_mask>0)
        h = torch.where(condition, 0, 1) * h
    else:
        print(f"Mode {mode} unknown!")

    input_hsv[:, 0, ::] = h
    return input_hsv

def limit_hsv_to_sequence2(input_hsv, input_mask, mode, params):
    ratio = params["ratio"]
    for idx, channel in enumerate(["h", "s", "v"]):
        ch = input_hsv[:, idx, ::] # Bx224x224
        scale_factor = 360.0 if channel == "h" else 1.0
        l1 = params[f"{channel}_l1"].view(-1, 1, 1) * scale_factor
        l1 = l1.repeat(1, 224, 224)
        l2 = params[f"{channel}_l2"].view(-1, 1, 1) * scale_factor
        l2 = l2.repeat(1, 224, 224)
        l3 = params[f"{channel}_l3"].view(-1, 1, 1) * scale_factor
        l3 = l3.repeat(1, 224, 224)
        s1 = params[f"{channel}_s1"].view(-1, 1, 1) * scale_factor
        s1 = s1.repeat(1, 224, 224)
        s2 = params[f"{channel}_s2"].view(-1, 1, 1) * scale_factor
        s2 = s2.repeat(1, 224, 224)
        s3 = params[f"{channel}_s3"].view(-1, 1, 1) * scale_factor
        s3 = s3.repeat(1, 224, 224)

        if mode == "remap":
            # Remap colors inside mask to defined colorspace (three intervals)
            inter1 = torch.logical_and(torch.logical_and(ch>=0, l1-ch>0), input_mask>0)
            inter2 = torch.logical_and(torch.logical_and(ch-l1>=0, l2+l1-ch>0), input_mask>0)
            inter3 = torch.logical_and(torch.logical_and(ch-l1-l2>=0, l3+l2+l1-ch>0), input_mask>0)
            inter_all = torch.logical_or(torch.logical_or(inter1, inter2), inter3)
            ch = torch.where(inter1, 1, 0) * (s1 + ratio * ch) + \
                torch.where(inter2, 1, 0) * (s2 + ratio * (ch - l1)) + \
                torch.where(inter3, 1, 0) * (s3 + ratio * (ch - l1 - l2)) + \
                torch.where(inter_all, 0, 1) * ch
        elif mode == "zero":
            # Set all colors inside mask that are outside of learned color space to zero
            inter1 = torch.logical_and(ch-s1 >= 0, s1 + ratio * l1 - ch > 0)
            inter2 = torch.logical_and(ch-s2 >= 0, s2 + ratio * l2 - ch > 0)
            inter3 = torch.logical_and(ch-s3 >= 0, s3 + ratio * l3 - ch > 0)
            not_inter_all = torch.logical_not(torch.logical_or(torch.logical_or(inter1, inter2), inter3))
            condition = torch.logical_and(not_inter_all, input_mask>0)
            ch = torch.where(condition, 0, 1) * ch
        else:
            print(f"Mode {mode} unknown!")
        input_hsv[:, idx, ::] = ch
    return input_hsv


def get_linear(c_in, c_out):
    linear = nn.Sequential(OrderedDict([
        # block 1: 224x3 --> 112x32
        ("param_linear1", nn.Linear(c_in, 16)),
        ("param_relu1", nn.ReLU()),
        ("param_linear2", nn.Linear(16, c_out)),
        ("param_sigmoid", nn.Sigmoid()),
    ]))
    return linear

def get_encoder(hidden_channels=24):
    C = hidden_channels
    encoder = nn.Sequential(OrderedDict([
        # block 1: 224x3 --> 112x32
        ("conv1a", nn.Conv2d(3, 32, 3, stride=1, padding=1)),
        ("batchnorm1a", nn.BatchNorm2d(32, affine=False)),
        ("relu1a", nn.ReLU()),
        ("conv1b", nn.Conv2d(32, 32, 3, stride=1, padding=1)),
        ("batchnorm1b", nn.BatchNorm2d(32, affine=False)),
        ("relu1b", nn.ReLU()),
        ("conv1c", nn.Conv2d(32, 32, 4, stride=2, padding=1)),
        ("batchnorm1c", nn.BatchNorm2d(32, affine=False)),
        ("relu1c", nn.ReLU()),

        # block 2: 112x32 --> 56x64
        ("conv2a", nn.Conv2d(32, 32, 3, stride=1, padding=1)),
        ("batchnorm2a", nn.BatchNorm2d(32, affine=False)),
        ("relu2a", nn.ReLU()),
        ("conv2b", nn.Conv2d(32, 32, 3, stride=1, padding=1)),
        ("batchnorm2b", nn.BatchNorm2d(32, affine=False)),
        ("relu2b", nn.ReLU()),
        ("conv2c", nn.Conv2d(32, 64, 4, stride=2, padding=1)),
        ("batchnorm2c", nn.BatchNorm2d(64, affine=False)),
        ("relu2c", nn.ReLU()),

        # block 3: 56x64 -->  28x64
        ("conv3a", nn.Conv2d(64, 64, 3, stride=1, padding=1)),
        ("batchnorm3a", nn.BatchNorm2d(64, affine=False)),
        ("relu3a", nn.ReLU()),
        ("conv3b", nn.Conv2d(64, 64, 3, stride=1, padding=1)),
        ("batchnorm3b", nn.BatchNorm2d(64, affine=False)),
        ("relu3b", nn.ReLU()),
        ("conv3c", nn.Conv2d(64, 64, 4, stride=2, padding=1)),
        ("batchnorm3c", nn.BatchNorm2d(64, affine=False)),
        ("relu3c", nn.ReLU()),

        # block 4: 28x64 --> 14x128
        ("conv4a", nn.Conv2d(64, 64, 3, stride=1, padding=1)),
        ("batchnorm4a", nn.BatchNorm2d(64, affine=False)),
        ("relu4a", nn.ReLU()),
        ("conv4b", nn.Conv2d(64, 64, 3, stride=1, padding=1)),
        ("batchnorm4b", nn.BatchNorm2d(64, affine=False)),
        ("relu4b", nn.ReLU()),
        ("conv4c", nn.Conv2d(64, 128, 4, stride=2, padding=1)),
        ("batchnorm4c", nn.BatchNorm2d(128, affine=False)),
        ("relu4c", nn.ReLU()),

        # block 5: 14x128 --> 7x256
        ("conv5a", nn.Conv2d(128, 128, 3, stride=1, padding=1)),
        ("batchnorm5a", nn.BatchNorm2d(128, affine=False)),
        ("relu5a", nn.ReLU()),
        ("conv5b", nn.Conv2d(128, 128, 3, stride=1, padding=1)),
        ("batchnorm5b", nn.BatchNorm2d(128, affine=False)),
        ("relu5b", nn.ReLU()),
        ("conv5c", nn.Conv2d(128, 256, 4, stride=2, padding=1)),
        ("batchnorm5c", nn.BatchNorm2d(256, affine=False)),
        ("relu5c", nn.ReLU()),

        # block 6: 7x256 --> 7xC
        ("conv6a", nn.Conv2d(256, 128, 3, stride=1, padding=1)),
        ("batchnorm6a", nn.BatchNorm2d(128, affine=False)),
        ("relu6a", nn.ReLU()),
        ("conv6b", nn.Conv2d(128, 64, 3, stride=1, padding=1)),
        ("batchnorm6b", nn.BatchNorm2d(64, affine=False)),
        ("relu6b", nn.ReLU()),
        ("conv6c", nn.Conv2d(64, 32, 3, stride=1, padding=1)),
        ("batchnorm6c", nn.BatchNorm2d(32, affine=False)),
        ("relu6c", nn.ReLU()),
        ("conv6d", nn.Conv2d(32, 32, 3, stride=1, padding=1)),
        ("batchnorm6d", nn.BatchNorm2d(32, affine=False)),
        ("relu6d", nn.ReLU()),
        ("conv6e", nn.Conv2d(32, C, 3, stride=1, padding=1)),
        ("batchnorm6e", nn.BatchNorm2d(C, affine=False)),
        ("relu6e", nn.ReLU()),
    ]))
    return encoder

def get_decoder(hidden_channels=8, output_channels=3, act="sigmoid"):
    C_in = hidden_channels
    C_out = output_channels
    decoder_dict = OrderedDict([
            # t_block 6: 7xC_in --> 14x128
            ("t_conv6a", nn.ConvTranspose2d(C_in, 16, 3, stride=1, padding=1)),
            ("t_batchnorm6a", nn.BatchNorm2d(16, affine=False)),
            ("t_relu6a", nn.ReLU()),
            ("t_conv6b", nn.ConvTranspose2d(16, 32, 3, stride=1, padding=1)),
            ("t_batchnorm6b", nn.BatchNorm2d(32, affine=False)),
            ("t_relu6b", nn.ReLU()),
            ("t_conv6c", nn.ConvTranspose2d(32, 64, 3, stride=1, padding=1)),
            ("t_batchnorm6c", nn.BatchNorm2d(64, affine=False)),
            ("t_relu6c", nn.ReLU()),
            ("t_conv6d", nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1)),
            ("t_batchnorm6d", nn.BatchNorm2d(128, affine=False)),
            ("t_relu6d", nn.ReLU()),
            ("t_conv6e", nn.ConvTranspose2d(128, 256, 3, stride=1, padding=1)),
            ("t_batchnorm6e", nn.BatchNorm2d(256, affine=False)),
            ("t_relu6e", nn.ReLU()),
            ("t_conv6f", nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)),
            ("t_batchnorm6f", nn.BatchNorm2d(128, affine=False)),
            ("t_relu6f", nn.ReLU()),

            # t_block 5: 14x128 --> 28x64
            ("t_conv5a", nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1)),
            ("t_batchnorm5a", nn.BatchNorm2d(128, affine=False)),
            ("t_relu5a", nn.ReLU()),
            ("t_conv5b", nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1)),
            ("t_batchnorm5b", nn.BatchNorm2d(128, affine=False)),
            ("t_relu5b", nn.ReLU()),
            ("t_conv5c", nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)),
            ("t_batchnorm5c", nn.BatchNorm2d(64, affine=False)),
            ("t_relu5c", nn.ReLU()),

            # t_block 4: 28x64 --> 56x64
            ("t_conv4a", nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)),
            ("t_batchnorm4a", nn.BatchNorm2d(64, affine=False)),
            ("t_relu4a", nn.ReLU()),
            ("t_conv4b", nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)),
            ("t_batchnorm4b", nn.BatchNorm2d(64, affine=False)),
            ("t_relu4b", nn.ReLU()),
            ("t_conv4c", nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)),
            ("t_batchnorm4c", nn.BatchNorm2d(64, affine=False)),
            ("t_relu4c", nn.ReLU()),

            # t_block 3: 56x64 --> 112x32
            ("t_conv3a", nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)),
            ("t_batchnorm3a", nn.BatchNorm2d(64, affine=False)),
            ("t_relu3a", nn.ReLU()),
            ("t_conv3b", nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)),
            ("t_batchnorm3b", nn.BatchNorm2d(64, affine=False)),
            ("t_relu3b", nn.ReLU()),
            ("t_conv3c", nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)),
            ("t_batchnorm3c", nn.BatchNorm2d(32, affine=False)),
            ("t_relu3c", nn.ReLU()),

            # t_block 2: 112x32 --> 224x32
            ("t_conv2a", nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1)),
            ("t_batchnorm2a", nn.BatchNorm2d(32, affine=False)),
            ("t_relu2a", nn.ReLU()),
            ("t_conv2b", nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1)),
            ("t_batchnorm2b", nn.BatchNorm2d(32, affine=False)),
            ("t_relu2b", nn.ReLU()),
            ("t_conv2c", nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)),
            ("t_batchnorm2c", nn.BatchNorm2d(32, affine=False)),
            ("t_relu2c", nn.ReLU()),

            # t_block 1: 224x32 --> 224xC_out
            ("t_conv1a", nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1)),
            ("t_batchnorm1a", nn.BatchNorm2d(32, affine=False)),
            ("t_relu1a", nn.ReLU()),
            ("t_conv1b", nn.ConvTranspose2d(32, C_out, 3, stride=1, padding=1)),
        ])
    if act == "sigmoid":
        decoder_dict.update({"sigmoid": nn.Sigmoid()})
    elif act == "tanh":
        print("Tanh")
        decoder_dict.update({"tanh": nn.Tanh()})
    else:
        print(f"No such activation function: {act}")
        return 0
    decoder = nn.Sequential(decoder_dict)
    return decoder
