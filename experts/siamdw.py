import os
import json
import cv2
import numpy as np
import torch
from got10k.trackers import Tracker
import external.deeper_wider_siamese_trackers.net.models as models
from external.deeper_wider_siamese_trackers.utils.utils import (
    load_pretrain,
    cxy_wh_2_rect,
)


def load_json(json_path):
    assert os.path.exists(json_path)
    cfg = json.load(open(json_path, "r"))
    return cfg


def to_torch(ndarray):
    return torch.from_numpy(ndarray)


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
    return im_to_torch(im_patch.copy()) if out_mode in "torch" else im_patch


def make_scale_pyramid(im, pos, in_side_scaled, out_side, avg_chans):
    in_side_scaled = [round(x) for x in in_side_scaled]
    num_scale = len(in_side_scaled)
    pyramid = torch.zeros(num_scale, 3, out_side, out_side)
    max_target_side = in_side_scaled[-1]
    min_target_side = in_side_scaled[0]
    beta = out_side / min_target_side

    search_side = round(beta * max_target_side)
    search_region = get_subwindow_tracking(
        im, pos, int(search_side), int(max_target_side), avg_chans, out_mode="np"
    )

    for s, temp in enumerate(in_side_scaled):
        target_side = round(beta * temp)
        pyramid[s, :] = get_subwindow_tracking(
            search_region, (1 + search_side) / 2, out_side, target_side, avg_chans
        )

    return pyramid


class TrackerConfig(object):
    # These are the default hyper-params for CIResNet22 based SiamFC+
    num_scale = 3
    scale_step = 1.0375
    scale_penalty = 0.9745
    scale_lr = 0.590
    response_up = 16

    windowing = "cosine"
    w_influence = 0.350

    exemplar_size = 127
    instance_size = 255
    score_size = 17
    total_stride = 8
    context_amount = 0.5

    def update(self, newparam=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()

    def renew(self):
        self.score_size = (
            self.instance_size - self.exemplar_size
        ) // self.total_stride + 1


def tracker_eval(net, s_x, x_crops, target_pos, window, p):
    # refer to original SiamFC code
    response_map = net.track(x_crops).squeeze().permute(1, 2, 0).cpu().data.numpy()
    up_size = p.response_up * response_map.shape[0]
    response_map_up = cv2.resize(
        response_map, (up_size, up_size), interpolation=cv2.INTER_CUBIC
    )
    temp_max = np.max(response_map_up, axis=(0, 1))
    s_penaltys = np.array([p.scale_penalty, 1.0, p.scale_penalty])
    temp_max *= s_penaltys
    best_scale = np.argmax(temp_max)

    response_map = response_map_up[..., best_scale]
    response_map = response_map - response_map.min()
    response_map = response_map / response_map.sum()

    # apply windowing
    response_map = (1 - p.w_influence) * response_map + p.w_influence * window
    r_max, c_max = np.unravel_index(response_map.argmax(), response_map.shape)
    p_corr = [c_max, r_max]

    disp_instance_final = p_corr - np.ceil(p.score_size * p.response_up / 2)
    disp_instance_input = disp_instance_final * p.total_stride / p.response_up
    disp_instance_frame = disp_instance_input * s_x / p.instance_size
    new_target_pos = target_pos + disp_instance_frame

    return new_target_pos, best_scale


def SiamFC_init(root, im, target_pos, target_sz, model):
    state = dict()
    p = TrackerConfig()
    cfg = load_json(root + "/external/deeper_wider_siamese_trackers/utils/config.json")
    config = cfg["SiamFC_Res22"]["OTB2013"]
    p.update(config)

    net = model

    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    scale_z = p.exemplar_size / s_z

    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad
    min_s_x = 0.2 * s_x
    max_s_x = 5 * s_x

    s_x_serise = {"s_x": s_x, "min_s_x": min_s_x, "max_s_x": max_s_x}
    p.update(s_x_serise)

    z = z_crop.unsqueeze(0)

    net.template(z.cuda())

    if p.windowing == "cosine":
        window = np.outer(
            np.hanning(int(p.score_size) * int(p.response_up)),
            np.hanning(int(p.score_size) * int(p.response_up)),
        )
    elif p.windowing == "uniform":
        window = np.ones(
            int(p.score_size) * int(p.response_up),
            int(p.score_size) * int(p.response_up),
        )
    window /= window.sum()

    p.scales = p.scale_step ** (range(p.num_scale) - np.ceil(p.num_scale // 2))

    state["p"] = p
    state["net"] = net
    state["avg_chans"] = avg_chans
    state["window"] = window
    state["target_pos"] = target_pos
    state["target_sz"] = target_sz
    state["im_h"] = im.shape[0]
    state["im_w"] = im.shape[1]
    return state


def SiamFC_track(state, im):
    p = state["p"]
    net = state["net"]
    avg_chans = state["avg_chans"]
    window = state["window"]
    target_pos = state["target_pos"]
    target_sz = state["target_sz"]

    scaled_instance = p.s_x * p.scales
    scaled_target = [[target_sz[0] * p.scales], [target_sz[1] * p.scales]]

    x_crops = make_scale_pyramid(
        im, target_pos, scaled_instance, p.instance_size, avg_chans
    )

    target_pos, new_scale = tracker_eval(
        net, p.s_x, x_crops.cuda(), target_pos, window, p
    )

    # scale damping and saturation
    p.s_x = max(
        p.min_s_x,
        min(
            p.max_s_x,
            (1 - p.scale_lr) * p.s_x + p.scale_lr * scaled_instance[new_scale],
        ),
    )

    target_sz = [
        (1 - p.scale_lr) * target_sz[0] + p.scale_lr * scaled_target[0][0][new_scale],
        (1 - p.scale_lr) * target_sz[1] + p.scale_lr * scaled_target[1][0][new_scale],
    ]

    target_pos[0] = max(0, min(state["im_w"], target_pos[0]))
    target_pos[1] = max(0, min(state["im_h"], target_pos[1]))
    target_sz[0] = max(10, min(state["im_w"], target_sz[0]))
    target_sz[1] = max(10, min(state["im_h"], target_sz[1]))
    state["target_pos"] = target_pos
    state["target_sz"] = target_sz
    state["p"] = p

    return state


class SiamDW(Tracker):
    def __init__(self, root):
        super(SiamDW, self).__init__(name="SiamDW")
        self.root = root
        self.model = models.__dict__["SiamFC_Res22"]()
        self.model = load_pretrain(
            self.model,
            self.root
            + "/external/deeper_wider_siamese_trackers/pretrain/CIResNet22.pth",
        )
        self.model.eval()
        self.model = self.model.cuda()

    def init(self, image, box):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.state = SiamFC_init(
            self.root, image, box[:2] + box[2:] / 2.0, box[2:], self.model
        )

    def update(self, image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.state = SiamFC_track(self.state, image)  # track
        center = self.state["target_pos"]
        target_sz = self.state["target_sz"]
        box = cxy_wh_2_rect(center, target_sz)
        return box
