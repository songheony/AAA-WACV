import cv2
import torch
import numpy as np
from got10k.trackers import Tracker
from external.DaSiamRPN.code.net import SiamRPNBIG
from external.DaSiamRPN.code.run_SiamRPN import SiamRPN_init, SiamRPN_track
from external.DaSiamRPN.code.utils import cxy_wh_2_rect


class DaSiamRPN(Tracker):
    def __init__(self, root):
        super(DaSiamRPN, self).__init__(name="DaSiamRPN")
        net_file = root + "/external/DaSiamRPN/SiamRPNBIG.model"
        self.net = SiamRPNBIG()
        self.net.load_state_dict(torch.load(net_file))
        self.net.eval().cuda()

        # warm up
        for i in range(10):
            self.net.temple(
                torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda()
            )
            self.net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

    def init(self, image, box):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.state = SiamRPN_init(
            image, box[:2] + box[2:] / 2.0, box[2:], self.net
        )  # init tracker

    def update(self, image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.state = SiamRPN_track(self.state, image)  # track
        center = self.state["target_pos"]
        target_sz = self.state["target_sz"]
        box = cxy_wh_2_rect(center, target_sz)
        return box
