import os
import pickle
import time
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import zmq
from got10k.experiments import (
    ExperimentOTB,
    ExperimentVOT,
    ExperimentTColor128,
)
from got10k.utils.metrics import poly_iou
from message import MessageType, Message
from aaa import AAA
from options import root_path


def run_otb(root, otb, ours, trackers, ex):
    context = zmq.Context()

    ventilators = [context.socket(zmq.PUSH) for _ in range(len(trackers))]
    for ventilator, tracker in zip(ventilators, trackers):
        ventilator.connect("tcp://%s:8888" % tracker)

    sinks = [context.socket(zmq.PULL) for _ in range(len(trackers))]
    for sink, tracker in zip(sinks, trackers):
        sink.connect("tcp://%s:6006" % tracker)

    for s, (img_files, anno) in enumerate(otb.dataset):
        seq_name = otb.dataset.seq_names[s]
        print("--Ex : %s, Sequence %d/%d: %s" % (ex, s + 1, len(otb.dataset), seq_name))

        record_files = [
            os.path.join(otb.result_dir, tracker, "%s.txt" % seq_name)
            for tracker in trackers
        ]

        record_file = os.path.join(otb.result_dir, ours.name, "%s.txt" % seq_name)

        frame_num = len(img_files)
        results = dict()

        results[ours.name] = {
            "boxes": np.zeros((frame_num, 4)),
            "times": np.zeros((frame_num)),
            "anchor_frames": [],
            "weights": np.zeros((frame_num, len(trackers))),
        }

        for tracker in trackers:
            results[tracker] = {
                "boxes": np.zeros((frame_num, 4)),
                "times": np.zeros((frame_num)),
            }

        # tracking loop
        for f, img_file in tqdm(enumerate(img_files)):
            image = Image.open(img_file)
            image = image.convert("RGB")
            if f == 0:
                data = {"image": image, "box": anno[0, :], "target": "all"}
                message = Message(MessageType["init"], data)
                for ventilator in ventilators:
                    ventilator.send_pyobj(message)

                for i in range(len(trackers)):
                    results[trackers[i]]["boxes"][f, :] = anno[0, :]
                    results[trackers[i]]["times"][f] = 0
            else:
                data = {"image": image, "target": "all"}
                message = Message(MessageType["track"], data)
                for ventilator in ventilators:
                    ventilator.send_pyobj(message)

                for sink in sinks:
                    result = sink.recv_pyobj()
                    result = result.data
                    name = result["name"]
                    duration = result["time"]
                    box = result["box"]
                    results[name]["boxes"][f, :] = box
                    results[name]["times"][f] = duration

            expert_boxes = [
                results[trackers[i]]["boxes"][f] for i in range(len(trackers))
            ]
            if f == 0:
                ours.init(image, anno[0, :])
                box = anno[0, :]
                offline_results = None
                duration = 0
            else:
                start_time = time.time()
                box, offline_results = ours.update(image, expert_boxes)
                duration = time.time() - start_time
            results[ours.name]["boxes"][f] = box
            results[ours.name]["times"][f] = duration
            results[ours.name]["anchor_frames"].append(offline_results)
            results[ours.name]["weights"][f] = ours.algorithm.w

            if f != 0 and offline_results is not None and ours.reset_tracker:
                data = {"image": image, "box": box, "target": "all"}
                message = Message(MessageType["init"], data)
                for ventilator in ventilators:
                    ventilator.send_pyobj(message)

        for i in range(len(trackers)):
            otb._record(
                record_files[i],
                results[trackers[i]]["boxes"],
                results[trackers[i]]["times"],
            )

        otb._record(
            record_file, results[ours.name]["boxes"], results[ours.name]["times"]
        )

        # record anchor frames
        anchor_dir = os.path.join(os.path.dirname(record_file), "anchor")
        if not os.path.isdir(anchor_dir):
            os.makedirs(anchor_dir)
        anchor_file = os.path.join(
            anchor_dir, os.path.basename(record_file).replace(".txt", "_anchor.pkl")
        )
        with open(anchor_file, "wb") as fp:
            pickle.dump(results[ours.name]["anchor_frames"], fp)

        weights_dir = os.path.join(os.path.dirname(record_file), "weights")
        if not os.path.isdir(weights_dir):
            os.makedirs(weights_dir)
        weights_file = os.path.join(
            weights_dir, os.path.basename(record_file).replace(".txt", "_weights.txt")
        )
        np.savetxt(weights_file, results[ours.name]["weights"], fmt="%f")


def run_supervised(root, vot, ours, trackers, ex):
    context = zmq.Context()

    ventilators = [context.socket(zmq.PUSH) for _ in range(len(trackers))]
    for ventilator, tracker in zip(ventilators, trackers):
        ventilator.connect("tcp://%s:8888" % tracker)

    sinks = [context.socket(zmq.PULL) for _ in range(len(trackers))]
    for sink, tracker in zip(sinks, trackers):
        sink.connect("tcp://%s:6006" % tracker)

    for s, (img_files, anno, _) in enumerate(vot.dataset):
        seq_name = vot.dataset.seq_names[s]
        print("--Ex : %s, Sequence %d/%d: %s" % (ex, s + 1, len(vot.dataset), seq_name))

        anno_rects = anno.copy()
        if anno_rects.shape[1] == 8:
            anno_rects = vot.dataset._corner2rect(anno_rects)

        for r in tqdm(range(vot.repetitions)):
            record_files = [
                os.path.join(
                    vot.result_dir,
                    tracker,
                    "baseline",
                    seq_name,
                    "%s_%03d.txt" % (seq_name, r + 1),
                )
                for tracker in trackers
            ]

            origin_files = [
                os.path.join(
                    vot.result_dir,
                    tracker,
                    "baseline",
                    seq_name,
                    "%s_%03d_origin.txt" % (seq_name, r + 1),
                )
                for tracker in trackers
            ]

            record_file = os.path.join(
                vot.result_dir,
                ours.name,
                "baseline",
                seq_name,
                "%s_%03d.txt" % (seq_name, r + 1),
            )

            results = dict()

            results[ours.name] = {
                "boxes": [],
                "times": [],
                "anchor_frames": [],
                "weights": [],
                "failure": False,
                "next_start": -1,
            }

            for tracker in trackers:
                results[tracker] = {
                    "boxes": [],
                    "origin_boxes": [],
                    "times": [],
                    "failure": False,
                    "next_start": -1,
                }

            # tracking loop
            for f, img_file in tqdm(enumerate(img_files)):
                image = Image.open(img_file)
                image = image.convert("RGB")
                if f == 0:
                    data = {"image": image, "box": anno_rects[0, :], "target": "all"}
                    message = Message(MessageType["init"], data)
                    for ventilator in ventilators:
                        ventilator.send_pyobj(message)

                    for i in range(len(trackers)):
                        results[trackers[i]]["boxes"].append([1])
                        results[trackers[i]]["origin_boxes"].append([1])
                        results[trackers[i]]["times"].append(0)
                else:
                    data = {"image": image, "target": "all"}
                    message = Message(MessageType["track"], data)
                    for ventilator in ventilators:
                        ventilator.send_pyobj(message)

                    for sink in sinks:
                        result = sink.recv_pyobj()
                        result = result.data
                        name = result["name"]
                        duration = result["time"]
                        box = result["box"]
                        results[name]["boxes"].append(box)
                        results[name]["origin_boxes"].append(box)
                        results[name]["times"].append(duration)

                    for i in range(len(trackers)):
                        key = trackers[i]
                        value = results[key]
                        box = value["boxes"][-1]
                        if value["failure"]:
                            # during failure frames
                            if f == value["next_start"]:
                                value["failure"] = False

                                data = {
                                    "image": image,
                                    "box": anno_rects[f],
                                    "target": key,
                                }
                                message = Message(MessageType["init"], data)
                                for ventilator in ventilators:
                                    ventilator.send_pyobj(message)

                                value["boxes"][-1] = [1]
                                # value['origin_boxes'][-1] = [1]
                            else:
                                results[name]["times"][-1] = np.NaN
                                value["boxes"][-1] = [0]
                                # value['origin_boxes'][-1] = [0]
                        else:
                            # during success frames
                            iou = poly_iou(anno[f], box, bound=image.size)
                            if iou <= 0.0:
                                # tracking failure
                                value["failure"] = True
                                value["next_start"] = f + vot.skip_initialize
                                value["boxes"][-1] = [2]

                expert_boxes = [
                    results[trackers[i]]["origin_boxes"][f]
                    for i in range(len(trackers))
                ]
                if f == 0:
                    ours.init(image, anno_rects[0, :])
                    box = [1]
                    offline_results = None
                    duration = 0
                else:
                    if results[ours.name]["failure"]:
                        if f == results[ours.name]["next_start"]:
                            results[ours.name]["failure"] = False
                            box = [1]
                            duration = 0
                            ours.init(image, anno_rects[f])
                        else:
                            duration = np.NaN
                            box = [0]
                        offline_results = None
                    else:
                        start_time = time.time()
                        box, offline_results = ours.update(image, expert_boxes)
                        duration = time.time() - start_time
                        iou = poly_iou(anno[f], box, bound=image.size)
                        if iou <= 0.0:
                            results[ours.name]["failure"] = True
                            results[ours.name]["next_start"] = f + vot.skip_initialize
                            box = [2]
                results[ours.name]["boxes"].append(box)
                results[ours.name]["times"].append(duration)
                results[ours.name]["anchor_frames"].append(offline_results)
                results[ours.name]["weights"].append(ours.algorithm.w)

                if f != 0 and offline_results is not None and ours.reset_tracker:
                    data = {"image": image, "box": box, "target": "all"}
                    message = Message(MessageType["init"], data)
                    for ventilator in ventilators:
                        ventilator.send_pyobj(message)

            for i in range(len(trackers)):
                vot._record(
                    record_files[i],
                    results[trackers[i]]["boxes"],
                    results[trackers[i]]["times"],
                )

            vot._record(
                record_file, results[ours.name]["boxes"], results[ours.name]["times"]
            )

            for i in range(len(trackers)):
                lines = []
                for box in results[trackers[i]]["origin_boxes"]:
                    if len(box) == 1:
                        lines.append("%d" % box[0])
                    else:
                        lines.append(str.join(",", ["%.4f" % t for t in box]))

                # record bounding boxes
                record_dir = os.path.dirname(origin_files[i])
                if not os.path.isdir(record_dir):
                    os.makedirs(record_dir)
                with open(origin_files[i], "w") as f:
                    f.write(str.join("\n", lines))

            anchor_file = record_file[: record_file.rfind("_")] + "_anchor.pkl"
            with open(anchor_file, "wb") as fp:
                pickle.dump(results[ours.name]["anchor_frames"], fp)

            weights_file = record_file[: record_file.rfind("_")] + "_weights.txt"
            np.savetxt(weights_file, results[ours.name]["weights"], fmt="%f")


def run_unsupervised(root, vot, ours, trackers, ex):
    context = zmq.Context()

    ventilators = [context.socket(zmq.PUSH) for _ in range(len(trackers))]
    for ventilator, tracker in zip(ventilators, trackers):
        ventilator.connect("tcp://%s:8888" % tracker)

    sinks = [context.socket(zmq.PULL) for _ in range(len(trackers))]
    for sink, tracker in zip(sinks, trackers):
        sink.connect("tcp://%s:6006" % tracker)

    for s, (img_files, anno, _) in enumerate(vot.dataset):
        seq_name = vot.dataset.seq_names[s]
        print("--Ex : %s, Sequence %d/%d: %s" % (ex, s + 1, len(vot.dataset), seq_name))

        record_files = [
            os.path.join(
                vot.result_dir,
                tracker,
                "unsupervised",
                seq_name,
                "%s_001.txt" % seq_name,
            )
            for tracker in trackers
        ]

        record_file = os.path.join(
            vot.result_dir, ours.name, "unsupervised", seq_name, "%s_001.txt" % seq_name
        )

        results = dict()

        results[ours.name] = {
            "boxes": [],
            "times": [],
            "anchor_frames": [],
            "weights": [],
        }

        for tracker in trackers:
            results[tracker] = {"boxes": [], "times": []}

        anno_rects = anno.copy()
        if anno_rects.shape[1] == 8:
            anno_rects = vot.dataset._corner2rect(anno_rects)

        # tracking loop
        for f, img_file in tqdm(enumerate(img_files)):
            image = Image.open(img_file)
            image = image.convert("RGB")
            if f == 0:
                data = {"image": image, "box": anno_rects[0, :], "target": "all"}
                message = Message(MessageType["init"], data)
                for ventilator in ventilators:
                    ventilator.send_pyobj(message)

                for i in range(len(trackers)):
                    results[trackers[i]]["boxes"].append([1])
                    results[trackers[i]]["times"].append(0)
            else:
                data = {"image": image, "target": "all"}
                message = Message(MessageType["track"], data)
                for ventilator in ventilators:
                    ventilator.send_pyobj(message)

                for sink in sinks:
                    result = sink.recv_pyobj()
                    result = result.data
                    name = result["name"]
                    duration = result["time"]
                    box = result["box"]
                    results[name]["boxes"].append(box)
                    results[name]["times"].append(duration)

            expert_boxes = [
                results[trackers[i]]["boxes"][f] for i in range(len(trackers))
            ]
            if f == 0:
                ours.init(image, anno_rects[0, :])
                box = [1]
                offline_results = None
                duration = 0
            else:
                start_time = time.time()
                box, offline_results = ours.update(image, expert_boxes)
                duration = time.time() - start_time
            results[ours.name]["boxes"].append(box)
            results[ours.name]["times"].append(duration)
            results[ours.name]["anchor_frames"].append(offline_results)
            results[ours.name]["weights"].append(ours.algorithm.w)

            if f != 0 and offline_results is not None and ours.reset_tracker:
                data = {"image": image, "box": box, "target": "all"}
                message = Message(MessageType["init"], data)
                for ventilator in ventilators:
                    ventilator.send_pyobj(message)

        for i in range(len(trackers)):
            vot._record(
                record_files[i],
                results[trackers[i]]["boxes"],
                results[trackers[i]]["times"],
            )

        vot._record(
            record_file, results[ours.name]["boxes"], results[ours.name]["times"]
        )

        anchor_file = record_file[: record_file.rfind("_")] + "_anchor.pkl"
        with open(anchor_file, "wb") as fp:
            pickle.dump(results[ours.name]["anchor_frames"], fp)

        weights_file = record_file[: record_file.rfind("_")] + "_weights.txt"
        np.savetxt(weights_file, results[ours.name]["weights"], fmt="%f")


def run_realtime(root, vot, ours, trackers, ex):
    context = zmq.Context()

    ventilators = [context.socket(zmq.PUSH) for _ in range(len(trackers))]
    for ventilator, tracker in zip(ventilators, trackers):
        ventilator.connect("tcp://%s:8888" % tracker)

    sinks = [context.socket(zmq.PULL) for _ in range(len(trackers))]
    for sink, tracker in zip(sinks, trackers):
        sink.connect("tcp://%s:6006" % tracker)

    for s, (img_files, anno, _) in enumerate(vot.dataset):
        seq_name = vot.dataset.seq_names[s]
        print("--Ex : %s, Sequence %d/%d: %s" % (ex, s + 1, len(vot.dataset), seq_name))

        record_files = [
            os.path.join(
                vot.result_dir, tracker, "realtime", seq_name, "%s_001.txt" % seq_name
            )
            for tracker in trackers
        ]

        origin_files = [
            os.path.join(
                vot.result_dir,
                tracker,
                "baseline",
                seq_name,
                "%s_001_origin.txt" % seq_name,
            )
            for tracker in trackers
        ]

        record_file = os.path.join(
            vot.result_dir, ours.name, "realtime", seq_name, "%s_001.txt" % seq_name
        )

        results = dict()

        results[ours.name] = {
            "boxes": [],
            "times": [],
            "anchor_frames": [],
            "weights": [],
        }

        for tracker in trackers:
            results[tracker] = {
                "boxes": [],
                "origin_boxes": [],
                "times": [],
                "failure": False,
                "next_start": 0,
                "failed_frame": -1,
                "total_time": 0.0,
                "grace": 3 - 1,
                "offset": 0,
                "current": 0,
            }

        anno_rects = anno.copy()
        if anno_rects.shape[1] == 8:
            anno_rects = vot.dataset._corner2rect(anno_rects)

        # tracking loop
        for f, img_file in tqdm(enumerate(img_files)):
            image = Image.open(img_file)
            image = image.convert("RGB")
            if f == 0:
                data = {"image": image, "box": anno_rects[0, :], "target": "all"}
                message = Message(MessageType["init"], data)
                for ventilator in ventilators:
                    ventilator.send_pyobj(message)

                for i in range(len(trackers)):
                    results[trackers[i]]["boxes"].append([1])
                    results[trackers[i]]["origin_boxes"].append([1])
                    results[trackers[i]]["times"].append(0)
            else:
                data = {"image": image, "target": "all"}
                message = Message(MessageType["track"], data)
                for ventilator in ventilators:
                    ventilator.send_pyobj(message)

                for sink in sinks:
                    result = sink.recv_pyobj()
                    result = result.data
                    name = result["name"]
                    duration = result["time"]
                    box = result["box"]
                    results[name]["boxes"].append(box)
                    results[name]["origin_boxes"].append(box)
                    results[name]["times"].append(duration)

                for i in range(len(trackers)):
                    key = trackers[i]
                    value = results[key]
                    box = value["boxes"][-1]
                    if f == value["next_start"]:
                        value["failure"] = False
                        value["failed_frame"] = -1
                        value["total_time"] = 0.0
                        value["grace"] = 3 - 1
                        value["offset"] = f

                        data = {"image": image, "box": anno_rects[f], "target": key}
                        message = Message(MessageType["init"], data)
                        for ventilator in ventilators:
                            ventilator.send_pyobj(message)

                        value["boxes"][-1] = [1]
                        value["origin_boxes"][-1] = [1]
                    elif not value["failure"]:
                        # during success frames
                        # calculate current frame
                        if value["grace"] > 0:
                            value["total_time"] += 1000.0 / 25
                            value["grace"] -= 1
                        else:
                            value["total_time"] += max(
                                1000.0 / 25, value["times"][-2] * 1000.0
                            )
                        value["current"] = value["offset"] + int(
                            np.round(np.floor(value["total_time"] * 25) / 1000.0)
                        )

                        # delayed/tracked bounding box
                        if f < value["current"]:
                            value["boxes"][-1] = value["boxes"][-2]
                            value["origin_boxes"][-1] = value["origin_boxes"][-2]

                        iou = poly_iou(anno[f], value["boxes"][-1], bound=image.size)
                        if iou <= 0.0:
                            # tracking failure
                            value["failure"] = True
                            value["failed_frame"] = f
                            value["next_start"] = value["current"] + vot.skip_initialize
                            value["boxes"][-1] = [2]
                    else:
                        # during failure frames
                        if f < value["current"]:
                            # skipping frame due to slow speed
                            value["boxes"][-1] = [0]
                            value["origin_boxes"][-1] = [0]
                            value["times"][-1] = np.NaN
                        elif f == value["current"]:
                            # current frame
                            iou = poly_iou(
                                anno[f], value["boxes"][-1], bound=image.size
                            )
                            if iou <= 0.0:
                                # tracking failure
                                value["boxes"][-1] = [2]
                                value["boxes"][value["failed_frame"]] = [0]
                                value["times"][value["failed_frame"]] = np.NaN
                        elif f < value["next_start"]:
                            # skipping frame due to failure
                            value["boxes"][-1] = [0]
                            value["origin_boxes"][-1] = [0]
                            value["times"][-1] = np.NaN

            expert_boxes = [
                results[trackers[i]]["origin_boxes"][-1] for i in range(len(trackers))
            ]
            if f == 0:
                ours.init(image, anno_rects[0, :])
                box = [1]
                offline_results = None
                duration = 0
            else:
                start_time = time.time()
                box, offline_results = ours.update(image, expert_boxes)
                duration = time.time() - start_time
            results[ours.name]["boxes"].append(box)
            results[ours.name]["times"].append(duration)
            results[ours.name]["anchor_frames"].append(offline_results)
            results[ours.name]["weights"].append(ours.algorithm.w)

            if f != 0 and offline_results is not None and ours.reset_tracker:
                data = {"image": image, "box": box, "target": "all"}
                message = Message(MessageType["init"], data)
                for ventilator in ventilators:
                    ventilator.send_pyobj(message)

        for i in range(len(trackers)):
            vot._record(
                record_files[i],
                results[trackers[i]]["boxes"],
                results[trackers[i]]["times"],
            )

        vot._record(
            record_file, results[ours.name]["boxes"], results[ours.name]["times"]
        )

        for i in range(len(trackers)):
            lines = []
            for box in results[trackers[i]]["origin_boxes"]:
                if len(box) == 1:
                    lines.append("%d" % box[0])
                else:
                    lines.append(str.join(",", ["%.4f" % t for t in box]))

            # record bounding boxes
            record_dir = os.path.dirname(origin_files[i])
            if not os.path.isdir(record_dir):
                os.makedirs(record_dir)
            with open(origin_files[i], "w") as f:
                f.write(str.join("\n", lines))

        anchor_file = record_file[: record_file.rfind("_")] + "_anchor.pkl"
        with open(anchor_file, "wb") as fp:
            pickle.dump(results[ours.name]["anchor_frames"], fp)

        weights_file = record_file[: record_file.rfind("_")] + "_weights.txt"
        np.savetxt(weights_file, results[ours.name]["weights"], fmt="%f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trackers", type=str, help="trackers")
    parser.add_argument("-d", "--datasets", type=str, help="datasets")
    args = parser.parse_args()
    trackers = args.trackers.split()
    datasets = args.datasets.split()
    aaa = AAA(len(trackers))
    if "TColor128" in datasets:
        otb = ExperimentTColor128(
            root_path + "/TColor128",
            result_dir=root_path + "/results",
            report_dir=root_path + "/reports",
        )
        run_otb(root_path, otb, aaa, trackers)
    if "OTB2013" in datasets:
        otb = ExperimentOTB(
            root_path + "/OTB",
            version=2013,
            result_dir=root_path + "/results",
            report_dir=root_path + "/reports",
        )
        run_otb(root_path, otb, aaa, trackers)
    if "OTB2015" in datasets:
        otb = ExperimentOTB(
            root_path + "/OTB",
            version=2015,
            result_dir=root_path + "/results",
            report_dir=root_path + "/reports",
        )
        run_otb(root_path, otb, aaa, trackers)
    if "VOT2018Super" in datasets:
        vot = ExperimentVOT(
            root_path + "/VOT2018",
            version=2018,
            result_dir=root_path + "/results",
            report_dir=root_path + "/reports",
        )
        vot.repetitions = 1
        run_supervised(root_path, vot, aaa, trackers)
    if "VOT2018Un" in datasets:
        vot = ExperimentVOT(
            root_path + "/VOT2018",
            version=2018,
            result_dir=root_path + "/results",
            report_dir=root_path + "/reports",
        )
        vot.repetitions = 1
        run_unsupervised(root_path, vot, aaa, trackers)
    if "VOT2018LT" in datasets:
        vot = ExperimentVOT(
            root_path + "/VOT2018LT",
            version="LT2018",
            result_dir=root_path + "/results",
            report_dir=root_path + "/reports",
        )
        vot.repetitions = 1
        run_supervised(root_path, vot, aaa, trackers)
