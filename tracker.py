import argparse
import time
import numpy as np
import zmq
from message import MessageType, Message
from options import root_path


def track(name, tracker):
    context = zmq.Context()

    ventilator = context.socket(zmq.PULL)
    ventilator.bind("tcp://*:8888")

    sink = context.socket(zmq.PUSH)
    sink.bind("tcp://*:6006")

    while True:
        print("Waiting")
        msg = ventilator.recv_pyobj()
        print("Recived")
        target = msg.data["target"]
        if target == "all" or target == name:
            if msg.messageType == MessageType["init"]:
                image = msg.data["image"]
                box = np.array(msg.data["box"])
                box[box < 1] = 1
                box[2] = min(image.size[0], box[0] + box[2]) - box[0]
                box[3] = min(image.size[1], box[1] + box[3]) - box[1]
                tracker.init(image, box)
            elif msg.messageType == MessageType["track"]:
                image = msg.data["image"]
                start_time = time.time()
                box = tracker.update(image)
                duration = time.time() - start_time
                data = {"name": name, "box": box, "time": duration}
                message = Message(MessageType["result"], data)
                sink.send_pyobj(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--expert", type=str, help="expert")
    args = parser.parse_args()

    if args.expert == "DaSiamRPN":
        from experts.dasiamrpn import DaSiamRPN as Tracker
    elif args.expert == "ECO-H":
        from experts.eco import ECO_H as Tracker
    elif args.expert == "MCCT-H":
        from experts.mcct import MCCT_H as Tracker
    elif args.expert == "SiamDW":
        from experts.siamdw import SiamFC_Plus as Tracker
    elif args.expert == "SiamFC":
        from experts.siamfc import SiamFC as Tracker
    elif args.expert == "SiamMask":
        from experts.siammask import SiamMaskTracking as Tracker
    elif args.expert == "SiamRPN":
        from experts.siamrpn import SiamRPN as Tracker
    tracker = Tracker(root_path)
    track(args.expert, tracker)
