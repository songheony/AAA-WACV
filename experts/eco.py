import numpy as np
import transplant
from got10k.trackers import Tracker


class ECO_H(Tracker):
    def __init__(self, root):
        super(ECO_H, self).__init__(name="ECO-H")
        self.root_path = root + "/external/ECO"
        self.eng = transplant.Matlab()
        genpath = self.eng.genpath(self.root_path)
        self.eng.addpath(genpath)
        self.eng.vl_setupnn()

    def init(self, image, box):
        center = box[:2] + box[2:] / 2.0
        self.eng.init_pos = center[::-1]
        self.eng.init_sz = box[2:][::-1]
        self.eng.img = np.array(image, dtype=float)
        self.eng.frame = 0
        self.eng.evalin("base", "tracker_init", nargout=0)
        self.eng.frame += 1
        self.eng.evalin("base", "tracker_track", nargout=0)

    def update(self, image):
        self.eng.frame += 1
        self.eng.img = np.array(image, dtype=float)
        self.eng.evalin("base", "tracker_track", nargout=0)
        center = self.eng.center_pos[0]
        target_sz = self.eng.target_size[0]
        box = [
            center[1] - target_sz[1] / 2.0,
            center[0] - target_sz[0] / 2.0,
            target_sz[1],
            target_sz[0],
        ]
        return box
