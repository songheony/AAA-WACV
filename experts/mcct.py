import numpy as np
import transplant
from got10k.trackers import Tracker


class MCCT_H(Tracker):
    def __init__(self, root):
        super(MCCT_H, self).__init__(name="MCCT-H")
        self.root_path = root + "/external/MCCT/TrackerMCCT-H"
        self.eng = transplant.Matlab()
        genpath = self.eng.genpath(self.root_path)
        self.eng.addpath(genpath)

    def init(self, image, box):
        center = box[:2] + box[2:] / 2.0
        self.eng.pos = center[::-1]
        self.eng.target_sz = box[2:][::-1]
        self.eng.img = np.array(image, dtype=float)
        self.eng.frame = 0
        self.eng.evalin("base", "tracker_init", nargout=0)
        self.eng.frame += 1
        self.eng.evalin("base", "tracker_track", nargout=0)

    def update(self, image):
        self.eng.frame += 1
        self.eng.img = np.array(image, dtype=float)
        self.eng.evalin("base", "tracker_track", nargout=0)
        box = np.array(self.eng.Final_rect_position[0])
        return box
