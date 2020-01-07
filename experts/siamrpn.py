from got10k.trackers import Tracker
from external.siamrpn_pytorch.siamrpn import TrackerSiamRPN


class SiamRPN(Tracker):
    def __init__(self, root):
        super(SiamRPN, self).__init__(name="SiamRPN")
        self.tracker = TrackerSiamRPN(
            net_path=root + "/external/siamrpn_pytorch/model.pth"
        )

    def init(self, image, box):
        self.tracker.init(image, box)

    def update(self, image):
        box = self.tracker.update(image)
        return box
