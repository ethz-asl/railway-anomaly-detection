import torch
from torch import nn
from collections import OrderedDict


def get_standard_module(in_channels, out_channels, log_idx):
    return [
        (f"conv3x3_{log_idx}", nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)),
        (f"batchnorm3x3_{log_idx}", nn.BatchNorm2d(out_channels, affine=False)),
        (f"relu3x3_{log_idx}", nn.ReLU()),
        (f"conv5x5_{log_idx}", nn.Conv2d(out_channels, out_channels, 5, stride=1, padding=2)),
        (f"batchnorm5x5_{log_idx}", nn.BatchNorm2d(out_channels, affine=False)),
        (f"relu5x5_{log_idx}", nn.ReLU()),
    ]

def get_light_module(in_channels, out_channels, log_idx):
    return [
        (f"conv7x7_{log_idx}", nn.Conv2d(in_channels, out_channels, 7, stride=1, padding=3)),
        (f"batchnorm7x7_{log_idx}", nn.BatchNorm2d(out_channels, affine=False)),
        (f"relu7x7_{log_idx}", nn.ReLU()),
    ]

class PatchClassModel(nn.Module):
    def __init__(self, stages=4, in_channels=6):
        super(PatchClassModel, self).__init__()
        self.stages = stages
        self.in_channels = in_channels

        # Classifier
        self.classifier = nn.Conv2d(32, 2, 1, stride=1, padding=0)

        # Block 1, always full!
        block_1_module_list = get_3x3_module(self.in_channels, 32, 0, stride=1, padding=1, transposed=False)
        block_1_module_list.extend(get_3x3_module(32, 32, 1, stride=1, padding=1, transposed=False))
        t_block_1_module_list = get_3x3_module(64, 32, 1, stride=1, padding=1, transposed=True)
        t_block_1_module_list.extend(get_3x3_module(32, 32, 0, stride=1, padding=1, transposed=True))
        self.block_1 = nn.Sequential(OrderedDict(block_1_module_list))
        self.t_block_1 = nn.Sequential(OrderedDict(t_block_1_module_list))

        # Block 2
        if stages >= 3: # full + skip connection after it!
            block_2_module_list = get_3x3_module(32, 32, 2, stride=2, padding=1, transposed=False)
            block_2_module_list.extend(get_3x3_module(32, 64, 3, stride=1, padding=1, transposed=False))
            block_2_module_list.extend(get_3x3_module(64, 64, 4, stride=1, padding=1, transposed=False))
            t_block_2_module_list = get_3x3_module(128, 64, 4, stride=1, padding=1, transposed=True)
            t_block_2_module_list.extend(get_3x3_module(64, 32, 3, stride=1, padding=1, transposed=True))
            t_block_2_module_list.extend(get_3x3_module(32, 32, 2, stride=2, padding=1, transposed=True))
            self.block_2 = nn.Sequential(OrderedDict(block_2_module_list))
            self.t_block_2 = nn.Sequential(OrderedDict(t_block_2_module_list))
        elif stages == 2: # full + no skip connection after it!
            block_2_module_list = get_3x3_module(32, 32, 2, stride=2, padding=1, transposed=False)
            block_2_module_list.extend(get_3x3_module(32, 64, 3, stride=1, padding=1, transposed=False))
            block_2_module_list.extend(get_3x3_module(64, 64, 4, stride=1, padding=1, transposed=False))
            t_block_2_module_list = get_3x3_module(64, 64, 4, stride=1, padding=1, transposed=True)
            t_block_2_module_list.extend(get_3x3_module(64, 32, 3, stride=1, padding=1, transposed=True))
            t_block_2_module_list.extend(get_3x3_module(32, 32, 2, stride=2, padding=1, transposed=True))
            self.block_2 = nn.Sequential(OrderedDict(block_2_module_list))
            self.t_block_2 = nn.Sequential(OrderedDict(t_block_2_module_list))
        elif stages == 1: # first two layers + no skip connection after it!
            block_2_module_list = get_3x3_module(32, 32, 2, stride=2, padding=1, transposed=False)
            block_2_module_list.extend(get_3x3_module(32, 64, 3, stride=1, padding=1, transposed=False))
            t_block_2_module_list = get_3x3_module(64, 32, 3, stride=1, padding=1, transposed=True)
            t_block_2_module_list.extend(get_3x3_module(32, 32, 2, stride=2, padding=1, transposed=True))
            self.block_2 = nn.Sequential(OrderedDict(block_2_module_list))
            self.t_block_2 = nn.Sequential(OrderedDict(t_block_2_module_list))
        elif stages == 0: # first layer only and no skip connection after it
            block_2_module_list = get_3x3_module(32, 32, 2, stride=2, padding=1, transposed=False)
            t_block_2_module_list = get_3x3_module(32, 32, 2, stride=2, padding=1, transposed=True)
            self.block_2 = nn.Sequential(OrderedDict(block_2_module_list))
            self.t_block_2 = nn.Sequential(OrderedDict(t_block_2_module_list))

        # Block 3
        if stages == 5: # full block 3 but no skip connection after it
            block_3_module_list = get_3x3_module(64, 64, 5, stride=2, padding=1, transposed=False)
            block_3_module_list.extend(get_3x3_module(64, 128, 6, stride=1, padding=1, transposed=False))
            block_3_module_list.extend(get_3x3_module(128, 128, 7, stride=1, padding=1, transposed=False))
            t_block_3_module_list = get_3x3_module(128, 128, 7, stride=1, padding=1, transposed=True)
            t_block_3_module_list.extend(get_3x3_module(128, 64, 6, stride=1, padding=1, transposed=True))
            t_block_3_module_list.extend(get_3x3_module(64, 64, 5, stride=2, padding=1, transposed=True))
            self.block_3 = nn.Sequential(OrderedDict(block_3_module_list))
            self.t_block_3 = nn.Sequential(OrderedDict(t_block_3_module_list))

        elif stages == 4: # only first two layers of block and no skip connection after it
            block_3_module_list = get_3x3_module(64, 64, 5, stride=2, padding=1, transposed=False)
            block_3_module_list.extend(get_3x3_module(64, 128, 6, stride=1, padding=1, transposed=False))
            t_block_3_module_list = get_3x3_module(128, 64, 6, stride=1, padding=1, transposed=True)
            t_block_3_module_list.extend(get_3x3_module(64, 64, 5, stride=2, padding=1, transposed=True))
            self.block_3 = nn.Sequential(OrderedDict(block_3_module_list))
            self.t_block_3 = nn.Sequential(OrderedDict(t_block_3_module_list))

        elif stages == 3: # only first layer of block and no skip connection after it
            block_3_module_list = get_3x3_module(64, 64, 5, stride=2, padding=1, transposed=False)
            t_block_3_module_list = get_3x3_module(64, 64, 5, stride=2, padding=1, transposed=True)
            self.block_3 = nn.Sequential(OrderedDict(block_3_module_list))
            self.t_block_3 = nn.Sequential(OrderedDict(t_block_3_module_list))


    def forward(self, x):
        b1 = self.block_1(x)
        b2 = self.block_2(b1)

        if self.stages < 3:
            t_b2 = self.t_block_2(b2)
        elif self.stages >= 3:
            b3 = self.block_3(b2)
            t_b3 = self.t_block_3(b3)
            t_b2 = self.t_block_2(torch.cat((t_b3, b2), dim=1))
        t_b1 = self.t_block_1(torch.cat((t_b2, b1), dim=1))

        final = self.classifier(t_b1)
        return {"out": final}

def get_3x3_module(in_channels, out_channels, log_idx, stride=1, padding=1, transposed=False):
    if not transposed:
        return [
            (f"conv{log_idx}", nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=padding)),
            (f"batchnorm{log_idx}", nn.BatchNorm2d(out_channels, affine=False)),
            (f"relu{log_idx}", nn.ReLU()),
        ]
    else:
        if stride == 2:
            return [
                    (f"t_conv{log_idx}", nn.ConvTranspose2d(in_channels, out_channels, 3, stride=stride, padding=padding)),
                    (f"t_pad{log_idx}", nn.ZeroPad2d((0, 1, 0, 1))),
                    (f"t_batchnorm{log_idx}", nn.BatchNorm2d(out_channels, affine=False)),
                    (f"t_relu{log_idx}", nn.ReLU()),
                ]
        else:
            return [
                (f"t_conv{log_idx}", nn.ConvTranspose2d(in_channels, out_channels, 3, stride=stride, padding=padding)),
                (f"t_batchnorm{log_idx}", nn.BatchNorm2d(out_channels, affine=False)),
                (f"t_relu{log_idx}", nn.ReLU()),
            ]


class PatchSegModelLight(nn.Module):
    def __init__(self, in_channels=6, out_channels=2, stages=2, patch_only=False):
        super(PatchSegModelLight, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stages = stages
        self.patch_only = patch_only

        # Block 1, always full!
        if self.patch_only:
            if self.stages == 1:
                module_list = [
                    # (f"pad01", nn.ConstantPad2d((2, 2, 2, 2), 0)),
                    (f"conv01", nn.Conv2d(self.in_channels, 128, 5, stride=1, dilation=1, padding=0)),
                    (f"relu01", nn.LeakyReLU(0.005)),
                    #(f"pad02", nn.ConstantPad2d((2, 2, 2, 2), 0)),
                    (f"conv02", nn.Conv2d(128, 256, 5, stride=1, dilation=1, padding=0)),
                    (f"relu02", nn.LeakyReLU(0.005)),
                    #(f"pad03", nn.ConstantPad2d((2, 2, 2, 2), 0)),
                    (f"conv03", nn.Conv2d(256, 256, 5, stride=1, dilation=1, padding=0)),
                    (f"relu03", nn.LeakyReLU(0.005)),
                    #(f"pad04", nn.ConstantPad2d((1, 2, 1, 2), 0)),
                    (f"conv04", nn.Conv2d(256, 128, 4, stride=1, dilation=1, padding=0)),
                    (f"relu04", nn.LeakyReLU(0.005)),
                ]
            elif self.stages == 2:
                module_list = [
                    # (f"pad01", nn.ConstantPad2d((2, 2, 2, 2), 0)),
                    (f"conv01", nn.Conv2d(self.in_channels, 128, 5, stride=1, dilation=1, padding=0)),
                    (f"relu01", nn.LeakyReLU(0.005)),
                    #(f"pad02", nn.ConstantPad2d((0, 1, 0, 1), -99999)),
                    (f"pool02", nn.MaxPool2d(2, stride=2, dilation=1, padding=0)),
                    (f"relu02", nn.LeakyReLU(0.005)),
                    # (f"pad03", nn.ConstantPad2d((4, 4, 4, 4), 0)),
                    (f"conv03", nn.Conv2d(128, 256, 5, stride=1, dilation=1, padding=0)),
                    (f"relu03", nn.LeakyReLU(0.005)),
                    # (f"pad04", nn.ConstantPad2d((1, 1, 1, 1), -99999)),
                    (f"pool04", nn.MaxPool2d(2, stride=2, dilation=1, padding=0)),
                    (f"relu04", nn.LeakyReLU(0.005)),
                    # (f"pad05", nn.ConstantPad2d((2, 2, 2, 2), 0)),
                    (f"conv05", nn.Conv2d(256, 256, 2, stride=1, dilation=1, padding=0)),
                    (f"relu05", nn.LeakyReLU(0.005)),
                    # (f"pad06", nn.ConstantPad2d((6, 6, 6, 6), 0)),
                    (f"conv06", nn.Conv2d(256, 128, 4, stride=1, dilation=1, padding=0)),
                    (f"relu06", nn.LeakyReLU(0.005)),
                ]
            elif self.stages == 3:
                module_list = [
                    # (f"pad01", nn.ConstantPad2d((2, 2, 2, 2), 0)),
                    (f"conv00", nn.Conv2d(self.in_channels, 128, 5, stride=1, dilation=1, padding=0)),
                    (f"relu00", nn.LeakyReLU(0.005)),
                    #(f"pad00", nn.ConstantPad2d((2, 2, 2, 2), 0)),
                    (f"conv01", nn.Conv2d(128, 128, 5, stride=1, dilation=1, padding=0)),
                    (f"relu01", nn.LeakyReLU(0.005)),
                    #(f"pad02", nn.ConstantPad2d((0, 1, 0, 1), -99999)),
                    (f"pool02", nn.MaxPool2d(2, stride=2, dilation=1, padding=0)),
                    (f"relu02", nn.LeakyReLU(0.005)),
                    #(f"pad03", nn.ConstantPad2d((4, 4, 4, 4), 0)),
                    (f"conv03", nn.Conv2d(128, 128, 5, stride=1, dilation=1, padding=0)),
                    (f"relu03", nn.LeakyReLU(0.005)),
                    #(f"pad04", nn.ConstantPad2d((1, 1, 1, 1), -99999)),
                    (f"pool04", nn.MaxPool2d(2, stride=2, dilation=1, padding=0)),
                    (f"relu04", nn.LeakyReLU(0.005)),
                    #(f"pad05", nn.ConstantPad2d((6, 6, 6, 6), 0)),
                    (f"conv05", nn.Conv2d(128, 256, 5, stride=1, dilation=1, padding=0)),
                    (f"relu05", nn.LeakyReLU(0.005)),
                    #(f"pad06", nn.ConstantPad2d((2, 2, 2, 2), -99999)),
                    (f"pool06", nn.MaxPool2d(2, stride=2, dilation=1, padding=0)),
                    (f"relu06", nn.LeakyReLU(0.005)),
                    #(f"pad07", nn.ConstantPad2d((8, 8, 8, 8), 0)),
                    (f"conv07", nn.Conv2d(256, 128, 4, stride=1, dilation=1, padding=0)),
                    (f"relu07", nn.LeakyReLU(0.005)),
                    #(f"pad08", nn.ConstantPad2d((12, 12, 12, 12), 0)),
                    # (f"conv08", nn.Conv2d(256, 128, 4, stride=1, dilation=1, padding=0)),
                    # (f"relu08", nn.LeakyReLU(0.005)),
                ]
        else:
            if self.stages == 1:
                module_list = [
                            (f"pad01", nn.ConstantPad2d((2, 2, 2, 2), 0)),
                            (f"conv01", nn.Conv2d(self.in_channels, 128, 5, stride=1, dilation=1, padding=0)),
                            (f"relu01", nn.LeakyReLU(0.005)),
                            (f"pad02", nn.ConstantPad2d((2, 2, 2, 2), 0)),
                            (f"conv02", nn.Conv2d(128, 256, 5, stride=1, dilation=1, padding=0)),
                            (f"relu02", nn.LeakyReLU(0.005)),
                            (f"pad03", nn.ConstantPad2d((2, 2, 2, 2), 0)),
                            (f"conv03", nn.Conv2d(256, 256, 5, stride=1, dilation=1, padding=0)),
                            (f"relu03", nn.LeakyReLU(0.005)),
                            (f"pad04", nn.ConstantPad2d((1, 2, 1, 2), 0)),
                            (f"conv04", nn.Conv2d(256, 128, 4, stride=1, dilation=1, padding=0)),
                            (f"relu04", nn.LeakyReLU(0.005)),
                        ]
            elif self.stages == 2:
                module_list = [
                    (f"pad01", nn.ConstantPad2d((2, 2, 2, 2), 0)),
                    (f"conv01", nn.Conv2d(self.in_channels, 128, 5, stride=1, dilation=1, padding=0)),
                    (f"relu01", nn.LeakyReLU(0.005)),
                    (f"pad02", nn.ConstantPad2d((0, 1, 0, 1), -99999)),
                    (f"pool02", nn.MaxPool2d(2, stride=1, dilation=1, padding=0)),
                    (f"relu02", nn.LeakyReLU(0.005)),
                    (f"pad03", nn.ConstantPad2d((4, 4, 4, 4), 0)),
                    (f"conv03", nn.Conv2d(128, 256, 5, stride=1, dilation=2, padding=0)),
                    (f"relu03", nn.LeakyReLU(0.005)),
                    (f"pad04", nn.ConstantPad2d((1, 1, 1, 1), -99999)),
                    (f"pool04", nn.MaxPool2d(2, stride=1, dilation=2, padding=0)),
                    (f"relu04", nn.LeakyReLU(0.005)),
                    (f"pad05", nn.ConstantPad2d((2, 2, 2, 2), 0)),
                    (f"conv05", nn.Conv2d(256, 256, 2, stride=1, dilation=4, padding=0)),
                    (f"relu05", nn.LeakyReLU(0.005)),
                    (f"pad06", nn.ConstantPad2d((6, 6, 6, 6), 0)),
                    (f"conv06", nn.Conv2d(256, 128, 4, stride=1, dilation=4, padding=0)),
                    (f"relu06", nn.LeakyReLU(0.005)),
                ]
            elif self.stages == 3:
                module_list = [
                    (f"pad00", nn.ConstantPad2d((2, 2, 2, 2), 0)),
                    (f"conv00", nn.Conv2d(self.in_channels, 128, 5, stride=1, dilation=1, padding=0)),
                    (f"relu00", nn.LeakyReLU(0.005)),
                    (f"pad01", nn.ConstantPad2d((2, 2, 2, 2), 0)),
                    (f"conv01", nn.Conv2d(128, 128, 5, stride=1, dilation=1, padding=0)),
                    (f"relu01", nn.LeakyReLU(0.005)),
                    (f"pad02", nn.ConstantPad2d((0, 1, 0, 1), -99999)),
                    (f"pool02", nn.MaxPool2d(2, stride=1, dilation=1, padding=0)),
                    (f"relu02", nn.LeakyReLU(0.005)),
                    (f"pad03", nn.ConstantPad2d((4, 4, 4, 4), 0)),
                    (f"conv03", nn.Conv2d(128, 128, 5, stride=1, dilation=2, padding=0)),
                    (f"relu03", nn.LeakyReLU(0.005)),
                    (f"pad04", nn.ConstantPad2d((1, 1, 1, 1), -99999)),
                    (f"pool04", nn.MaxPool2d(2, stride=1, dilation=2, padding=0)),
                    (f"relu04", nn.LeakyReLU(0.005)),
                    (f"pad05", nn.ConstantPad2d((8, 8, 8, 8), 0)),
                    (f"conv05", nn.Conv2d(128, 256, 5, stride=1, dilation=4, padding=0)),
                    (f"relu05", nn.LeakyReLU(0.005)),
                    (f"pad06", nn.ConstantPad2d((2, 2, 2, 2), -99999)),
                    (f"pool06", nn.MaxPool2d(2, stride=1, dilation=4, padding=0)),
                    (f"relu06", nn.LeakyReLU(0.005)),
                    (f"pad07", nn.ConstantPad2d((12, 12, 12, 12), 0)),
                    (f"conv07", nn.Conv2d(256, 128, 4, stride=1, dilation=8, padding=0)),
                    (f"relu07", nn.LeakyReLU(0.005)),
                    # (f"pad08", nn.ConstantPad2d((12, 12, 12, 12), 0)),
                    # (f"conv08", nn.Conv2d(256, 128, 4, stride=1, dilation=8, padding=0)),
                    # (f"relu08", nn.LeakyReLU(0.005)),
                ]

        self.block_1 = nn.Sequential(OrderedDict(module_list))

    def forward(self, x):
        b1 = self.block_1(x)
        return {"out": b1, "descriptor": b1}


