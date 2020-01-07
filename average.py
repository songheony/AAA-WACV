import numpy as np
from got10k.trackers import Tracker


class Average(Tracker):
    def __init__(self, n_experts):
        super(Average, self).__init__(name="Average")
        self.n_experts = n_experts

    def init(self, image, box):
        pass

    def update(self, image, bboxes):
        return np.mean(bboxes, axis=0)
