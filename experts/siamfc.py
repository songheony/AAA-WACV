from got10k.trackers import Tracker
from external.siamfc.siamfc import TrackerSiamFC


class SiamFC(Tracker):
    def __init__(self, root):
        super(SiamFC, self).__init__(name="SiamFC")
        self.tracker = TrackerSiamFC(net_path=root + "/external/siamfc/model.pth")

    def init(self, image, box):
        self.tracker.init(image, box)

    def update(self, image):
        box = self.tracker.update(image)
        return box
