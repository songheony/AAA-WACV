import math
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from got10k.trackers import Tracker
from external.SiamMask.utils.load_helper import load_pretrain
from external.SiamMask.utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from external.SiamMask.models.siammask import SiamMask
from external.SiamMask.models.features import Features
from external.SiamMask.models.rpn import RPN, DepthCorr
from external.SiamMask.models.mask import Mask
from external.SiamMask.utils.anchors import Anchors
from external.SiamMask.utils.tracker_config import TrackerConfig

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(Features):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # padding = (2 - stride) + (dilation // 2 - 1)
        padding = 2 - stride
        assert (
            stride == 1 or dilation == 1
        ), "stride and dilation must have one equals to zero at least"
        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if out.size() != residual.size():
            print(out.size(), residual.size())
        out += residual

        out = self.relu(out)

        return out


class Bottleneck_nop(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_nop, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        s = residual.size(3)
        residual = residual[:, :, 1 : s - 1, 1 : s - 1]

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, layer4=False, layer3=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=0, bias=False  # 3
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 31x31, 15x15

        self.feature_size = 128 * block.expansion

        if layer3:
            self.layer3 = self._make_layer(
                block, 256, layers[2], stride=1, dilation=2
            )  # 15x15, 7x7
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x  # identity

        if layer4:
            self.layer4 = self._make_layer(
                block, 512, layers[3], stride=1, dilation=4
            )  # 7x7, 3x3
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x  # identity

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=3,
                        stride=stride,
                        bias=False,
                        padding=padding,
                        dilation=dd,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        # layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dd))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        p0 = self.relu(x)
        x = self.maxpool(p0)

        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)

        return p0, p1, p2, p3


class ResAdjust(nn.Module):
    def __init__(
        self, block=Bottleneck, out_channels=256, adjust_number=1, fuse_layers=[2, 3, 4]
    ):
        super(ResAdjust, self).__init__()
        self.fuse_layers = set(fuse_layers)

        if 2 in self.fuse_layers:
            self.layer2 = self._make_layer(block, 128, 1, out_channels, adjust_number)
        if 3 in self.fuse_layers:
            self.layer3 = self._make_layer(block, 256, 2, out_channels, adjust_number)
        if 4 in self.fuse_layers:
            self.layer4 = self._make_layer(block, 512, 4, out_channels, adjust_number)

        self.feature_size = out_channels * len(self.fuse_layers)

    def _make_layer(self, block, plances, dilation, out, number=1):

        layers = []

        for _ in range(number):
            layer = block(plances * block.expansion, plances, dilation=dilation)
            layers.append(layer)

        downsample = nn.Sequential(
            nn.Conv2d(
                plances * block.expansion, out, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out),
        )
        layers.append(downsample)

        return nn.Sequential(*layers)

    def forward(self, p2, p3, p4):

        outputs = []

        if 2 in self.fuse_layers:
            outputs.append(self.layer2(p2))
        if 3 in self.fuse_layers:
            outputs.append(self.layer3(p3))
        if 4 in self.fuse_layers:
            outputs.append(self.layer4(p4))
        # return torch.cat(outputs, 1)
        return outputs


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model


if __name__ == "__main__":
    net = resnet50()
    print(net)
    net = net.cuda()

    var = torch.FloatTensor(1, 3, 127, 127).cuda()

    net(var)
    print("*************")
    var = torch.FloatTensor(1, 3, 255, 255).cuda()

    net(var)


class ResDownS(nn.Module):
    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(inplane, outplane, kernel_size=1, bias=False),
            nn.BatchNorm2d(outplane),
        )

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l, r = 4, -4
            x = x[:, :, l:r, l:r]
        return x


class ResDown(Features):
    def __init__(self, pretrain=False):
        super(ResDown, self).__init__()
        self.features = resnet50(layer3=True, layer4=False)
        if pretrain:
            load_pretrain(self.features, "resnet.model")

        self.downsample = ResDownS(1024, 256)

    def forward(self, x):
        output = self.features(x)
        p3 = self.downsample(output[-1])
        return p3

    def forward_all(self, x):
        output = self.features(x)
        p3 = self.downsample(output[-1])
        return output, p3


class UP(RPN):
    def __init__(self, anchor_num=5, feature_in=256, feature_out=256):
        super(UP, self).__init__()

        self.anchor_num = anchor_num
        self.feature_in = feature_in
        self.feature_out = feature_out

        self.cls_output = 2 * self.anchor_num
        self.loc_output = 4 * self.anchor_num

        self.cls = DepthCorr(feature_in, feature_out, self.cls_output)
        self.loc = DepthCorr(feature_in, feature_out, self.loc_output)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class MaskCorr(Mask):
    def __init__(self, oSz=63):
        super(MaskCorr, self).__init__()
        self.oSz = oSz
        self.mask = DepthCorr(256, 256, self.oSz ** 2)

    def forward(self, z, x):
        return self.mask(z, x)


class Refine(nn.Module):
    def __init__(self):
        """
        Mask refinement module
        Please refer SiamMask (Appendix A)
        https://arxiv.org/abs/1812.05050
        """
        super(Refine, self).__init__()
        self.v0 = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(),
        )

        self.v1 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.ReLU(),
        )

        self.v2 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.h2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.h1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )

        self.h0 = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(),
        )

        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)

        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)

    def forward(self, f, corr_feature, pos=None):
        p0 = torch.nn.functional.pad(f[0], [16, 16, 16, 16])[
            :, :, 4 * pos[0] : 4 * pos[0] + 61, 4 * pos[1] : 4 * pos[1] + 61
        ]
        p1 = torch.nn.functional.pad(f[1], [8, 8, 8, 8])[
            :, :, 2 * pos[0] : 2 * pos[0] + 31, 2 * pos[1] : 2 * pos[1] + 31
        ]
        p2 = torch.nn.functional.pad(f[2], [4, 4, 4, 4])[
            :, :, pos[0] : pos[0] + 15, pos[1] : pos[1] + 15
        ]

        p3 = corr_feature[:, :, pos[0], pos[1]].view(-1, 256, 1, 1)

        out = self.deconv(p3)
        out = self.post0(F.upsample(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.upsample(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.upsample(self.h0(out) + self.v0(p0), size=(127, 127)))
        out = out.view(-1, 127 * 127)
        return out


class Custom(SiamMask):
    def __init__(self, pretrain=False, **kwargs):
        super(Custom, self).__init__(**kwargs)
        self.features = ResDown(pretrain=pretrain)
        self.rpn_model = UP(anchor_num=self.anchor_num, feature_in=256, feature_out=256)
        self.mask_model = MaskCorr()
        self.refine_model = Refine()

    def refine(self, f, pos=None):
        return self.refine_model(f, pos)

    def template(self, template):
        self.zf = self.features(template)

    def track(self, search):
        search = self.features(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, search)
        return rpn_pred_cls, rpn_pred_loc

    def track_mask(self, search):
        self.feature, self.search = self.features.forward_all(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, self.search)
        self.corr_feature = self.mask_model.mask.forward_corr(self.zf, self.search)
        pred_mask = self.mask_model.mask.head(self.corr_feature)
        return rpn_pred_cls, rpn_pred_loc, pred_mask

    def track_refine(self, pos):
        pred_mask = self.refine_model(self.feature, self.corr_feature, pos=pos)
        return pred_mask


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode="torch"):
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0.0, -context_xmin))
    top_pad = int(max(0.0, -context_ymin))
    right_pad = int(max(0.0, context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0.0, context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros(
            (r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8
        )
        te_im[top_pad : top_pad + r, left_pad : left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad : left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad :, left_pad : left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad :, :] = avg_chans
        im_patch_original = te_im[
            int(context_ymin) : int(context_ymax + 1),
            int(context_xmin) : int(context_xmax + 1),
            :,
        ]
    else:
        im_patch_original = im[
            int(context_ymin) : int(context_ymax + 1),
            int(context_xmin) : int(context_xmax + 1),
            :,
        ]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original
    # cv2.imshow('crop', im_patch)
    # cv2.waitKey(0)
    return im_to_torch(im_patch) if out_mode in "torch" else im_patch


def generate_anchor(cfg, score_size):
    anchors = Anchors(cfg)
    anchor = anchors.anchors
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)

    total_stride = anchors.stride
    anchor_num = anchor.shape[0]

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = -(score_size // 2) * total_stride
    xx, yy = np.meshgrid(
        [ori + total_stride * dx for dx in range(score_size)],
        [ori + total_stride * dy for dy in range(score_size)],
    )
    xx, yy = (
        np.tile(xx.flatten(), (anchor_num, 1)).flatten(),
        np.tile(yy.flatten(), (anchor_num, 1)).flatten(),
    )
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def siamese_init(im, target_pos, target_sz, model, hp=None):
    state = dict()
    state["im_h"] = im.shape[0]
    state["im_w"] = im.shape[1]
    p = TrackerConfig()
    p.update(hp, model.anchors)

    p.renew()

    net = model
    p.scales = model.anchors["scales"]
    p.ratios = model.anchors["ratios"]
    p.anchor_num = len(p.ratios) * len(p.scales)
    p.anchor = generate_anchor(model.anchors, p.score_size)

    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    z = z_crop.unsqueeze(0)
    net.template(z.cuda())

    if p.windowing == "cosine":
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == "uniform":
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state["p"] = p
    state["net"] = net
    state["avg_chans"] = avg_chans
    state["window"] = window
    state["target_pos"] = target_pos
    state["target_sz"] = target_sz
    return state


def siamese_track(state, im, mask_enable=False, refine_enable=False):
    p = state["p"]
    net = state["net"]
    avg_chans = state["avg_chans"]
    window = state["window"]
    target_pos = state["target_pos"]
    target_sz = state["target_sz"]

    wc_x = target_sz[1] + p.context_amount * sum(target_sz)
    hc_x = target_sz[0] + p.context_amount * sum(target_sz)
    s_x = np.sqrt(wc_x * hc_x)
    scale_x = p.exemplar_size / s_x
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_x
    s_x = s_x + 2 * pad
    crop_box = [
        target_pos[0] - round(s_x) / 2,
        target_pos[1] - round(s_x) / 2,
        round(s_x),
        round(s_x),
    ]

    # extract scaled crops for search region x at previous target position
    x_crop = get_subwindow_tracking(
        im, target_pos, p.instance_size, round(s_x), avg_chans
    ).unsqueeze(0)

    if mask_enable:
        score, delta, mask = net.track_mask(x_crop.cuda())
    else:
        score, delta = net.track(x_crop.cuda())

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = (
        F.softmax(
            score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0), dim=1
        )
        .data[:, 1]
        .cpu()
        .numpy()
    )

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    def change(r):
        return np.maximum(r, 1.0 / r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    target_sz_in_crop = target_sz * scale_x
    s_c = change(
        sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop))
    )  # scale penalty
    r_c = change(
        (target_sz_in_crop[0] / target_sz_in_crop[1]) / (delta[2, :] / delta[3, :])
    )  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
    pscore = penalty * score

    # cos window (motion model)
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    pred_in_crop = delta[:, best_pscore_id] / scale_x
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr  # lr for OTB

    res_x = pred_in_crop[0] + target_pos[0]
    res_y = pred_in_crop[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
    res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])

    # for Mask Branch
    if mask_enable:
        best_pscore_id_mask = np.unravel_index(
            best_pscore_id, (5, p.score_size, p.score_size)
        )
        delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[1]

        if refine_enable:
            mask = (
                net.track_refine((delta_y, delta_x))
                .cuda()
                .sigmoid()
                .squeeze()
                .view(p.out_size, p.out_size)
                .cpu()
                .data.numpy()
            )
        else:
            mask = (
                mask[0, :, delta_y, delta_x]
                .sigmoid()
                .squeeze()
                .view(p.out_size, p.out_size)
                .cpu()
                .data.numpy()
            )

        def crop_back(image, bbox, out_sz, padding=-1):
            a = (out_sz[0] - 1) / bbox[2]
            b = (out_sz[1] - 1) / bbox[3]
            c = -a * bbox[0]
            d = -b * bbox[1]
            mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float)
            crop = cv2.warpAffine(
                image,
                mapping,
                (out_sz[0], out_sz[1]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=padding,
            )
            return crop

        s = crop_box[2] / p.instance_size
        sub_box = [
            crop_box[0] + (delta_x - p.base_size / 2) * p.total_stride * s,
            crop_box[1] + (delta_y - p.base_size / 2) * p.total_stride * s,
            s * p.exemplar_size,
            s * p.exemplar_size,
        ]
        s = p.out_size / sub_box[2]
        back_box = [
            -sub_box[0] * s,
            -sub_box[1] * s,
            state["im_w"] * s,
            state["im_h"] * s,
        ]
        mask_in_img = crop_back(mask, back_box, (state["im_w"], state["im_h"]))

        target_mask = (mask_in_img > p.seg_thr).astype(np.uint8)
        if cv2.__version__[-5] == "4":
            contours, _ = cv2.findContours(
                target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
        else:
            _, contours, _ = cv2.findContours(
                target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]  # use max area polygon
            polygon = contour.reshape(-1, 2)
            # pbox = cv2.boundingRect(polygon)  # Min Max Rectangle
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # Rotated Rectangle

            # box_in_img = pbox
            rbox_in_img = prbox
        else:  # empty mask
            location = cxy_wh_2_rect(target_pos, target_sz)
            rbox_in_img = np.array(
                [
                    [location[0], location[1]],
                    [location[0] + location[2], location[1]],
                    [location[0] + location[2], location[1] + location[3]],
                    [location[0], location[1] + location[3]],
                ]
            )

    target_pos[0] = max(0, min(state["im_w"], target_pos[0]))
    target_pos[1] = max(0, min(state["im_h"], target_pos[1]))
    target_sz[0] = max(10, min(state["im_w"], target_sz[0]))
    target_sz[1] = max(10, min(state["im_h"], target_sz[1]))

    state["target_pos"] = target_pos
    state["target_sz"] = target_sz
    state["score"] = score
    state["mask"] = mask_in_img if mask_enable else []
    state["ploygon"] = rbox_in_img if mask_enable else []
    return state


class SiamMaskTracking(Tracker):
    def __init__(self, root):
        super(SiamMaskTracking, self).__init__(name="SiamMask")
        config = json.load(
            open(root + "/external/SiamMask/experiments/siammask/config_vot.json")
        )
        self.model = Custom(anchors=config["anchors"])
        self.model = load_pretrain(
            self.model, root + "/external/SiamMask/SiamMask_VOT.pth"
        )
        self.model.eval()
        self.model = self.model.cuda()
        self.hp = config["hp"]

    def init(self, image, box):
        im = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cx, cy, w, h = get_axis_aligned_bbox(box)
        target_pos = np.array([cx, cy])
        target_sz = np.array([w, h])
        self.state = siamese_init(im, target_pos, target_sz, self.model, self.hp)

    def update(self, image):
        im = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.state = siamese_track(
            self.state, im, mask_enable=True, refine_enable=True
        )  # track
        location = cxy_wh_2_rect(self.state["target_pos"], self.state["target_sz"])
        return location
