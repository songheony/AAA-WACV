import os
import itertools
import json
import pickle
import numpy as np
from scipy.stats import rankdata
from PIL import Image
from got10k.experiments import ExperimentOTB, ExperimentVOT, ExperimentTColor128
from got10k.utils.metrics import rect_iou, poly_iou, center_error
from pysot.eao_benchmark import EAOBenchmark
from pysot.ar_benchmark import AccuracyRobustnessBenchmark


def otb_eval(otb, trackers):
    report_dir = os.path.join(otb.report_dir, trackers[0])
    if not os.path.isdir(report_dir):
        os.makedirs(report_dir)
    report_file = os.path.join(report_dir, "performance.json")

    performance = {}
    for name in trackers:
        print("Evaluating", name)
        seq_num = len(otb.dataset)
        succ_curve = np.zeros((seq_num, otb.nbins_iou))
        prec_curve = np.zeros((seq_num, otb.nbins_ce))
        speeds = np.zeros(seq_num)

        performance.update({name: {"all": {}, "seq_wise": {}}})

        for s, (_, anno) in enumerate(otb.dataset):
            seq_name = otb.dataset.seq_names[s]
            record_file = os.path.join(otb.result_dir, name, "%s.txt" % seq_name)
            boxes = np.loadtxt(record_file, delimiter=",")
            boxes[0] = anno[0]
            assert len(boxes) == len(anno)

            ious, center_errors = otb._calc_metrics(boxes, anno)
            succ_curve[s], prec_curve[s] = otb._calc_curves(ious, center_errors)

            time_file = os.path.join(
                otb.result_dir, name, "times/%s_time.txt" % seq_name
            )
            if os.path.isfile(time_file):
                times = np.loadtxt(time_file)
                times = times[times > 0]
                if len(times) > 0:
                    speeds[s] = np.mean(1.0 / times)

            performance[name]["seq_wise"].update(
                {
                    seq_name: {
                        "success_curve": succ_curve[s].tolist(),
                        "precision_curve": prec_curve[s].tolist(),
                        "success_score": np.mean(succ_curve[s]),
                        "precision_score": prec_curve[s][20],
                        "success_rate": succ_curve[s][otb.nbins_iou // 2],
                        "speed_fps": speeds[s] if speeds[s] > 0 else -1,
                    }
                }
            )

        succ_curve = np.mean(succ_curve, axis=0)
        prec_curve = np.mean(prec_curve, axis=0)
        succ_score = np.mean(succ_curve)
        prec_score = prec_curve[20]
        succ_rate = succ_curve[otb.nbins_iou // 2]
        if np.count_nonzero(speeds) > 0:
            avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
        else:
            avg_speed = -1

        performance[name]["all"].update(
            {
                "success_curve": succ_curve.tolist(),
                "precision_curve": prec_curve.tolist(),
                "success_score": succ_score,
                "precision_score": prec_score,
                "success_rate": succ_rate,
                "speed_fps": avg_speed,
            }
        )

    for name in trackers:
        succ_curve = np.array(
            [
                performance[name]["seq_wise"][seq]["success_curve"]
                for seq in otb.dataset.seq_names
            ]
        )
        prec_curve = np.array(
            [
                performance[name]["seq_wise"][seq]["precision_curve"]
                for seq in otb.dataset.seq_names
            ]
        )
        speeds = np.array(
            [
                performance[name]["seq_wise"][seq]["speed_fps"]
                for seq in otb.dataset.seq_names
            ]
        )
        performance[name] = {
            "success_curve": np.mean(succ_curve, axis=0).tolist(),
            "precision_curve": np.mean(prec_curve, axis=0).tolist(),
            "success_score": np.mean(succ_curve, axis=0).mean(),
            "precision_score": np.mean(prec_curve, axis=0)[20],
            "success_rate": np.mean(succ_curve, axis=0)[otb.nbins_iou // 2],
            "speed_fps": np.sum(speeds) / np.count_nonzero(speeds),
        }

    with open(report_file, "w") as f:
        json.dump(performance, f, indent=4)

    return performance


def vot_eval(vot, trackers):
    eao = EAOBenchmark(vot)
    ar = AccuracyRobustnessBenchmark(vot)
    eao_results = eao.eval(trackers)
    ar_results = ar.eval(trackers)

    # assume trackers[0] is your tracker
    report_dir = os.path.join(vot.report_dir, trackers[0])
    if not os.path.isdir(report_dir):
        os.makedirs(report_dir)
    report_file = os.path.join(report_dir, "performance.json")

    performance = {}
    for name in trackers:
        print("Evaluating", name)
        performance[name] = {"seq_wise": {}}
        speeds = []

        videos = list(ar_results[name]["overlaps"].keys())
        for video in videos:
            overlaps = ar_results[name]["overlaps"][video]
            accuracy = np.nanmean(overlaps)
            failures = ar_results[name]["failures"][video]
            lost_number = np.mean(failures)
            # collect frame runtimes
            time_file = os.path.join(
                vot.result_dir, name, "baseline", video, "%s_time.txt" % video
            )
            times = np.loadtxt(time_file, delimiter=",")
            if times.ndim > 1:
                times = times[:, 0]
            times = times[~np.isnan(times)]
            times = times[times > 0]
            if len(times) > 0:
                speed = np.mean(1.0 / times)
            else:
                speed = 0
            speeds.append(speed)
            performance[name]["seq_wise"][video] = {
                "overlaps": overlaps,
                "accuracy": accuracy,
                "failures": failures,
                "lost_number": lost_number,
                "speed_fps": speed if speed > 0 else -1,
            }

        length = sum([len(x) for x in ar_results[name]["overlaps"].values()])
        overlaps = list(itertools.chain(*ar_results[name]["overlaps"].values()))
        accuracy = np.nanmean(overlaps)
        failures = list(ar_results[name]["failures"].values())
        lost_number = np.mean(np.sum(failures, axis=0))
        robustness = np.mean(np.sum(np.array(failures), axis=0) / length) * 100
        if np.count_nonzero(speeds) > 0:
            avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
        else:
            avg_speed = -1

        performance[name]["all"] = {
            "overlaps": overlaps,
            "accuracy": accuracy,
            "failures": failures,
            "lost_number": lost_number,
            "robustness": robustness,
            "speed_fps": avg_speed,
        }

        performance[name]["eao"] = eao_results[name]

    # report the performance
    with open(report_file, "w") as f:
        json.dump(performance, f, indent=4)

    return performance


def vot_un_eval(vot, trackers):
    def read_record(filename):
        with open(filename) as f:
            record = f.read().strip().split("\n")
        record = [[float(t) for t in line.split(",")] for line in record]
        return record

    def calc_metrics(boxes, anno):
        ious = rect_iou(boxes, anno)
        center_errors = center_error(boxes, anno)
        return ious, center_errors

    def calc_curves(ious, center_errors):
        ious = np.asarray(ious, float)[:, np.newaxis]
        center_errors = np.asarray(center_errors, float)[:, np.newaxis]

        thr_iou = np.linspace(0, 1, 21)[np.newaxis, :]
        thr_ce = np.arange(0, 51)[np.newaxis, :]

        bin_iou = np.greater(ious, thr_iou)
        bin_ce = np.less_equal(center_errors, thr_ce)

        succ_curve = np.mean(bin_iou, axis=0)
        prec_curve = np.mean(bin_ce, axis=0)

        return succ_curve, prec_curve

    report_dir = os.path.join(vot.report_dir, trackers[0])
    if not os.path.isdir(report_dir):
        os.makedirs(report_dir)
    report_file = os.path.join(report_dir, "performance_un.json")

    performance = {}
    for name in trackers:
        print("Evaluating", name)
        seq_num = len(vot.dataset)
        succ_curve = np.zeros((seq_num, 21))
        prec_curve = np.zeros((seq_num, 51))
        speeds = np.zeros(seq_num)

        performance.update({name: {"all": {}, "seq_wise": {}}})

        for s, (_, anno, meta) in enumerate(vot.dataset):
            anno_rects = anno.copy()
            if anno_rects.shape[1] == 8:
                anno_rects = vot.dataset._corner2rect(anno_rects)

            seq_name = vot.dataset.seq_names[s]
            record_file = os.path.join(
                vot.result_dir,
                name,
                "unsupervised",
                seq_name,
                "%s_%03d.txt" % (seq_name, 1),
            )
            boxes = read_record(record_file)
            boxes[0] = anno_rects[0]
            boxes = np.array(boxes)
            assert len(boxes) == len(anno_rects)

            ious, center_errors = calc_metrics(boxes, anno_rects)
            succ_curve[s], prec_curve[s] = calc_curves(ious, center_errors)

            time_file = os.path.join(
                vot.result_dir, name, "unsupervised", seq_name, "%s_time.txt" % seq_name
            )
            if os.path.isfile(time_file):
                times = np.loadtxt(time_file, delimiter=",")
                if times.ndim > 1:
                    times = times[:, 0]
                times = times[times > 0]
                if len(times) > 0:
                    speeds[s] = np.mean(1.0 / times)

            performance[name]["seq_wise"].update(
                {
                    seq_name: {
                        "success_curve": succ_curve[s].tolist(),
                        "precision_curve": prec_curve[s].tolist(),
                        "success_score": np.mean(succ_curve[s]),
                        "precision_score": prec_curve[s][20],
                        "success_rate": succ_curve[s][21 // 2],
                        "speed_fps": speeds[s] if speeds[s] > 0 else -1,
                        "tags": {},
                    }
                }
            )

        succ_curve = np.mean(succ_curve, axis=0)
        prec_curve = np.mean(prec_curve, axis=0)
        succ_score = np.mean(succ_curve)
        prec_score = prec_curve[20]
        succ_rate = succ_curve[21 // 2]
        if np.count_nonzero(speeds) > 0:
            avg_speed = np.sum(speeds) / np.count_nonzero(speeds)
        else:
            avg_speed = -1

        performance[name]["all"].update(
            {
                "success_curve": succ_curve.tolist(),
                "precision_curve": prec_curve.tolist(),
                "success_score": succ_score,
                "precision_score": prec_score,
                "success_rate": succ_rate,
                "speed_fps": avg_speed,
            }
        )

    for name in trackers:
        success_curve = []
        precision_curve = []
        for s, (_, anno, meta) in enumerate(vot.dataset):
            anno_rects = anno.copy()
            if anno_rects.shape[1] == 8:
                anno_rects = vot.dataset._corner2rect(anno_rects)

            seq_name = vot.dataset.seq_names[s]
            record_file = os.path.join(
                vot.result_dir,
                name,
                "unsupervised",
                seq_name,
                "%s_%03d.txt" % (seq_name, 1),
            )
            boxes = read_record(record_file)
            boxes[0] = anno_rects[0]
            boxes = np.array(boxes)

            ious, center_errors = calc_metrics(boxes, anno_rects)
            succ_curve, prec_curve = calc_curves(ious, center_errors)

            success_curve.append(succ_curve)
            precision_curve.append(prec_curve)
        performance[name] = {
            "success_curve": np.mean(success_curve, axis=0).tolist(),
            "precision_curve": np.mean(precision_curve, axis=0).tolist(),
            "success_score": np.mean(success_curve, axis=0).mean(),
            "precision_score": np.mean(precision_curve, axis=0)[20],
            "success_rate": np.mean(success_curve, axis=0)[21 // 2],
        }

    with open(report_file, "w") as f:
        json.dump(performance, f, indent=4)

    return performance


def get_video(dataset, seq_name, ttrackers):
    # function for loading results
    def read_record(filename):
        with open(filename) as f:
            record = f.read().strip().split("\n")
        record = [[float(t) for t in line.split(",")] for line in record]
        return record

    trackers = ttrackers.copy()
    trackers[trackers.index("SiamFC_Res22")] = "SiamFC_Plus"

    if isinstance(dataset, ExperimentVOT):
        for s, (img_files, anno, _) in enumerate(dataset.dataset):
            if seq_name == dataset.dataset.seq_names[s]:
                break
        anno_rects = anno.copy()
        if anno_rects.shape[1] == 8:
            anno_rects = dataset.dataset._corner2rect(anno_rects)
        record_files = [
            os.path.join(
                dataset.result_dir,
                name,
                "unsupervised",
                seq_name,
                "%s_%03d.txt" % (seq_name, 1),
            )
            for name in trackers
        ]
        record_file = os.path.join(
            dataset.result_dir,
            "Ours",
            "unsupervised",
            seq_name,
            "%s_001.txt" % seq_name,
        )
        anchor_file = os.path.join(
            dataset.result_dir,
            "Ours",
            "unsupervised",
            seq_name,
            "%s_anchor.pkl" % seq_name,
        )
        boxes = [read_record(f) for f in record_files]
        for box in boxes:
            box[0] = anno_rects[0]
        weights_file = record_file[: record_file.rfind("_")] + "_weights.txt"
    else:
        for s, (img_files, anno) in enumerate(dataset.dataset):
            if seq_name == dataset.dataset.seq_names[s]:
                break
        anno_rects = anno.copy()
        record_files = [
            os.path.join(dataset.result_dir, name, "%s.txt" % seq_name)
            for name in trackers
        ]
        anchor_file = os.path.join(
            dataset.result_dir, trackers[0], "anchor/%s_anchor.pkl" % seq_name
        )
        record_file = os.path.join(dataset.result_dir, "Ours", "%s.txt" % seq_name)
        weights_dir = os.path.join(os.path.dirname(record_file), "weights")
        weights_file = os.path.join(
            weights_dir, os.path.basename(record_file).replace(".txt", "_weights.txt")
        )
        boxes = [np.loadtxt(record_file, delimiter=",") for record_file in record_files]
    with open(anchor_file, "rb") as f:
        anchors = pickle.load(f)
    weights = np.loadtxt(weights_file)
    boxes = np.array(boxes)

    return boxes, weights, anchors, img_files, anno_rects


def calc_regret(dataset, seq_name, trackerss):
    # function for loading results
    def read_record(filename):
        with open(filename) as f:
            record = f.read().strip().split("\n")
        record = [[float(t) for t in line.split(",")] for line in record]
        return record

    trackers = trackerss.copy()
    trackers[trackerss.index("SiamFC_Res22")] = "SiamFC_Plus"

    if isinstance(dataset, ExperimentVOT):
        for s, (img_files, anno, _) in enumerate(dataset.dataset):
            if seq_name == dataset.dataset.seq_names[s]:
                break
        anno_rects = anno.copy()
        if anno_rects.shape[1] == 8:
            anno_rects = dataset.dataset._corner2rect(anno_rects)
        record_files = [
            os.path.join(
                dataset.result_dir,
                name,
                "unsupervised",
                seq_name,
                "%s_%03d.txt" % (seq_name, 1),
            )
            for name in trackers
        ]
        anchor_file = os.path.join(
            dataset.result_dir,
            "Ours",
            "unsupervised",
            seq_name,
            "%s_anchor.pkl" % seq_name,
        )
        boxes = [read_record(f) for f in record_files]
        for box in boxes:
            box[0] = anno_rects[0]
    else:
        for s, (img_files, anno) in enumerate(dataset.dataset):
            if seq_name == dataset.dataset.seq_names[s]:
                break
        anno_rects = anno.copy()
        record_files = [
            os.path.join(dataset.result_dir, name, "%s.txt" % seq_name)
            for name in trackers
        ]
        anchor_file = os.path.join(
            dataset.result_dir, trackers[0], "anchor/%s_anchor.pkl" % seq_name
        )
        boxes = [np.loadtxt(record_file, delimiter=",") for record_file in record_files]
        boxes = np.array(boxes)
    with open(anchor_file, "rb") as f:
        anchors = pickle.load(f)
    boxes = np.array(boxes)

    prev_anchor_frame = 0
    losses = np.zeros((len(anno_rects) - 1, len(trackers)))
    for frame in range(1, len(anno_rects)):
        if anchors[frame] is not None:
            for i in range(prev_anchor_frame + 1, frame + 1):
                anchor = anchors[frame][i - (prev_anchor_frame + 1)]
                losses[frame - 1] = [
                    (1.0 - rect_iou(boxes[j, i], anchor)) for j in range(len(trackers))
                ]
            prev_anchor_frame = frame
        elif frame == len(anno_rects) - 1:
            for i in range(prev_anchor_frame + 1, frame + 1):
                losses[frame - 1] = [
                    (1.0 - rect_iou(boxes[j, i], anno_rects[i]))
                    for j in range(len(trackers))
                ]

    sum_loss = np.sum(losses[:, 1:], axis=0)
    best_idx = np.argmin(sum_loss)
    regrets = [
        [
            np.sum(losses[: i + 1, j], axis=0)
            - np.sum(losses[: i + 1, best_idx + 1], axis=0)
            for i in range(len(losses))
        ]
        for j in range(len(trackers))
    ]
    return regrets


def best_expert(dataset, seq_name, trackerss):
    # function for loading results
    def read_record(filename):
        with open(filename) as f:
            record = f.read().strip().split("\n")
        record = [[float(t) for t in line.split(",")] for line in record]
        return record

    trackers = trackerss.copy()
    trackers[trackerss.index("SiamFC_Res22")] = "SiamFC_Plus"

    if isinstance(dataset, ExperimentVOT):
        for s, (img_files, anno, _) in enumerate(dataset.dataset):
            if seq_name == dataset.dataset.seq_names[s]:
                break
        anno_rects = anno.copy()
        if anno_rects.shape[1] == 8:
            anno_rects = dataset.dataset._corner2rect(anno_rects)
        record_files = [
            os.path.join(
                dataset.result_dir,
                name,
                "unsupervised",
                seq_name,
                "%s_%03d.txt" % (seq_name, 1),
            )
            for name in trackers
        ]
        anchor_file = os.path.join(
            dataset.result_dir,
            "Ours",
            "unsupervised",
            seq_name,
            "%s_anchor.pkl" % seq_name,
        )
        boxes = [read_record(f) for f in record_files]
        for box in boxes:
            box[0] = anno_rects[0]
    else:
        for s, (img_files, anno) in enumerate(dataset.dataset):
            if seq_name == dataset.dataset.seq_names[s]:
                break
        anno_rects = anno.copy()
        record_files = [
            os.path.join(dataset.result_dir, name, "%s.txt" % seq_name)
            for name in trackers
        ]
        anchor_file = os.path.join(
            dataset.result_dir, trackers[0], "anchor/%s_anchor.pkl" % seq_name
        )
        boxes = [np.loadtxt(record_file, delimiter=",") for record_file in record_files]
        boxes = np.array(boxes)
    with open(anchor_file, "rb") as f:
        anchors = pickle.load(f)
    boxes = np.array(boxes)

    prev_anchor_frame = 0
    losses = np.zeros((len(anno_rects) - 1, len(trackers)))
    for frame in range(1, len(anno_rects)):
        if anchors[frame] is not None:
            for i in range(prev_anchor_frame + 1, frame + 1):
                anchor = anchors[frame][i - (prev_anchor_frame + 1)]
                losses[frame - 1] = [
                    (1.0 - rect_iou(boxes[j, i], anchor)) for j in range(len(trackers))
                ]
            prev_anchor_frame = frame
        elif frame == len(anno_rects) - 1:
            for i in range(prev_anchor_frame + 1, frame + 1):
                losses[frame - 1] = [
                    (1.0 - rect_iou(boxes[j, i], anno_rects[i]))
                    for j in range(len(trackers))
                ]

    sum_loss = np.sum(losses[:, 1:], axis=0)
    best_idx = np.argmin(sum_loss)
    return best_idx + 1


def calc_error(dataset, seq_name, trackerss):
    # function for loading results
    def read_record(filename):
        with open(filename) as f:
            record = f.read().strip().split("\n")
        record = [[float(t) for t in line.split(",")] for line in record]
        return record

    trackers = trackerss.copy()
    trackers[trackerss.index("SiamFC_Res22")] = "SiamFC_Plus"

    if isinstance(dataset, ExperimentVOT):
        for s, (img_files, anno, _) in enumerate(dataset.dataset):
            if seq_name == dataset.dataset.seq_names[s]:
                break
        anno_rects = anno.copy()
        if anno_rects.shape[1] == 8:
            anno_rects = dataset.dataset._corner2rect(anno_rects)
        record_files = [
            os.path.join(
                dataset.result_dir,
                name,
                "unsupervised",
                seq_name,
                "%s_%03d.txt" % (seq_name, 1),
            )
            for name in trackers
        ]
        boxes = [read_record(f) for f in record_files]
        for box in boxes:
            box[0] = anno_rects[0]
    else:
        for s, (img_files, anno) in enumerate(dataset.dataset):
            if seq_name == dataset.dataset.seq_names[s]:
                break
        anno_rects = anno.copy()
        record_files = [
            os.path.join(dataset.result_dir, name, "%s.txt" % seq_name)
            for name in trackers
        ]
        boxes = [np.loadtxt(record_file, delimiter=",") for record_file in record_files]
        boxes = np.array(boxes)
    boxes = np.array(boxes)

    losses = np.zeros((len(anno_rects) - 1, len(trackers)))
    for frame in range(1, len(anno_rects)):
        losses[frame - 1] = [
            (1.0 - rect_iou(boxes[j, frame], anno_rects[frame]))
            for j in range(len(trackers))
        ]

    sum_loss = np.sum(losses[:, 1:], axis=0)
    best_idx = np.argmin(sum_loss)
    regrets = [
        [
            np.sum(losses[: i + 1, j], axis=0)
            - np.sum(losses[: i + 1, best_idx + 1], axis=0)
            for i in range(len(losses))
        ]
        for j in range(len(trackers))
    ]
    return regrets


def anchor_ratio_diff_performances(dataset, trackers, vot_un=False, disc=True):
    # function for loading results
    def read_record(filename):
        with open(filename) as f:
            record = f.read().strip().split("\n")
        record = [[float(t) for t in line.split(",")] for line in record]
        return record

    ratios = []
    mean_ratio = 0
    performances = []
    mean_performance = 0
    diffs = []
    mean_diff = 0
    total_length = 0

    if isinstance(dataset, ExperimentVOT):
        for s, (img_files, anno, _) in enumerate(dataset.dataset):
            seq_name = dataset.dataset.seq_names[s]
            bound = Image.open(img_files[0]).size

            if vot_un:
                record_files = [
                    os.path.join(
                        dataset.result_dir,
                        name,
                        "unsupervised",
                        seq_name,
                        "%s_%03d.txt" % (seq_name, 1),
                    )
                    for name in trackers
                ]
                anchor_file = os.path.join(
                    dataset.result_dir,
                    "Ours",
                    "unsupervised",
                    seq_name,
                    "%s_anchor.pkl" % seq_name,
                )
            else:
                record_files = [
                    os.path.join(
                        dataset.result_dir,
                        name,
                        "baseline",
                        seq_name,
                        "%s_%03d.txt" % (seq_name, 1),
                    )
                    for name in trackers
                ]
                anchor_file = os.path.join(
                    dataset.result_dir,
                    "Ours",
                    "baseline",
                    seq_name,
                    "%s_anchor.pkl" % seq_name,
                )
            boxes = [read_record(f) for f in record_files]
            with open(anchor_file, "rb") as f:
                anchors = pickle.load(f)
            diff = 0
            performance = 0
            ratio = 0
            for f, _ in enumerate(img_files):
                if anchors[f] is not None and ~np.any(np.isnan(anno[f])):
                    iou = (
                        poly_iou(np.array(anchors[f][-1]), anno[f], bound)[0]
                        if len(anchors[f][-1]) > 1
                        else np.NaN
                    )
                    best_iou = max(
                        [
                            poly_iou(np.array(box[f]), anno[f], bound)[0]
                            if len(box[f]) > 1
                            else np.NaN
                            for box in boxes
                        ]
                    )
                    if best_iou > iou + 0.01:
                        if disc:
                            diff += 1
                        else:
                            diff += best_iou - iou
                    performance += iou
                    ratio += 1
            if ratio != 0:
                mean_diff += diff
                mean_performance += performance
                diff /= ratio
                performance /= ratio
            diffs.append(diff)
            performances.append(performance)
            mean_ratio += ratio
            total_length += len(img_files)
            ratio /= len(img_files)
            ratios.append(ratio)
    else:
        for s, (img_files, anno) in enumerate(dataset.dataset):
            seq_name = dataset.dataset.seq_names[s]
            bound = Image.open(img_files[0]).size

            record_files = [
                os.path.join(dataset.result_dir, name, "%s.txt" % seq_name)
                for name in trackers
            ]
            boxes = [
                np.loadtxt(record_file, delimiter=",") for record_file in record_files
            ]

            anchor_file = os.path.join(
                dataset.result_dir, trackers[0], "anchor/%s_anchor.pkl" % seq_name
            )
            with open(anchor_file, "rb") as f:
                anchors = pickle.load(f)
            ratio = 0
            performance = 0
            diff = 0
            for f, _ in enumerate(img_files):
                if anchors[f] is not None and ~np.any(np.isnan(anno[f])):
                    iou = rect_iou(
                        np.array(anchors[f][-1])[None, :], anno[f][None, :], bound=bound
                    )[0]
                    best_iou = max(
                        [
                            rect_iou(
                                np.array(box[f])[None, :], anno[f][None, :], bound=bound
                            )[0]
                            for box in boxes
                        ]
                    )
                    if best_iou > iou + 0.01:
                        if disc:
                            diff += 1
                        else:
                            diff += best_iou - iou
                    performance += iou
                    ratio += 1
            if ratio != 0:
                mean_diff += diff
                mean_performance += performance
                diff /= ratio
                performance /= ratio
            diffs.append(diff)
            performances.append(performance)
            mean_ratio += ratio
            total_length += len(img_files)
            ratio /= len(img_files)
            ratios.append(ratio)
    mean_performance /= mean_ratio
    mean_diff /= mean_ratio
    mean_ratio /= total_length
    return (
        ratios,
        mean_ratio,
        diffs,
        mean_diff,
        performances,
        mean_performance,
        dataset.dataset.seq_names,
    )


def anchor_ratio_performances(dataset, trackers, vot_un=False, disc=True):
    ratios = []
    mean_ratio = 0
    performances = []
    mean_performance = 0
    video_perf = []
    total_length = 0
    video_ranks = []

    if isinstance(dataset, ExperimentVOT):
        for s, (img_files, anno, _) in enumerate(dataset.dataset):
            seq_name = dataset.dataset.seq_names[s]
            bound = Image.open(img_files[0]).size

            if vot_un:
                anchor_file = os.path.join(
                    dataset.result_dir,
                    "Ours",
                    "unsupervised",
                    seq_name,
                    "%s_anchor.pkl" % seq_name,
                )
                perf = get_performance(dataset, trackers[0], "performance_un.json")
                video_per = perf[trackers[0]]["seq_wise"][seq_name]["success_score"]
                succ_rank = [
                    perf[name]["seq_wise"][seq_name]["success_score"]
                    for name in trackers
                ]
                succ_rank = len(succ_rank) + 1 - rankdata(succ_rank, method="ordinal")
            else:
                anchor_file = os.path.join(
                    dataset.result_dir,
                    "Ours",
                    "baseline",
                    seq_name,
                    "%s_anchor.pkl" % seq_name,
                )
                perf = get_performance(dataset, trackers[0])
                video_per = perf[trackers[0]]["seq_wise"][seq_name]["success_score"]
                succ_rank = [
                    perf[name]["seq_wise"][seq_name]["success_score"]
                    for name in trackers
                ]
                succ_rank = len(succ_rank) + 1 - rankdata(succ_rank, method="ordinal")
            with open(anchor_file, "rb") as f:
                anchors = pickle.load(f)
            performance = 0
            ratio = 0
            for f, _ in enumerate(img_files):
                if anchors[f] is not None and ~np.any(np.isnan(anno[f])):
                    iou = (
                        poly_iou(np.array(anchors[f][-1]), anno[f], bound)[0]
                        if len(anchors[f][-1]) > 1
                        else np.NaN
                    )
                    performance += iou
                    ratio += 1
            if ratio != 0:
                mean_performance += performance
                performance /= ratio
            video_perf.append(video_per)
            video_ranks.append(succ_rank[0])
            performances.append(performance)
            mean_ratio += ratio
            total_length += len(img_files)
            ratio /= len(img_files)
            ratios.append(ratio)
    else:
        for s, (img_files, anno) in enumerate(dataset.dataset):
            seq_name = dataset.dataset.seq_names[s]
            bound = Image.open(img_files[0]).size

            anchor_file = os.path.join(
                dataset.result_dir, trackers[0], "anchor/%s_anchor.pkl" % seq_name
            )
            with open(anchor_file, "rb") as f:
                anchors = pickle.load(f)
            perf = get_performance(dataset, trackers[0])
            video_per = perf[trackers[0]]["seq_wise"][seq_name]["success_score"]
            succ_rank = [
                perf[name]["seq_wise"][seq_name]["success_score"] for name in trackers
            ]
            succ_rank = len(succ_rank) + 1 - rankdata(succ_rank, method="ordinal")
            ratio = 0
            performance = 0
            for f, _ in enumerate(img_files):
                if anchors[f] is not None and ~np.any(np.isnan(anno[f])):
                    iou = rect_iou(
                        np.array(anchors[f][-1])[None, :], anno[f][None, :], bound=bound
                    )[0]
                    performance += iou
                    ratio += 1
            if ratio != 0:
                mean_performance += performance
                performance /= ratio
            video_perf.append(video_per)
            video_ranks.append(succ_rank[0])
            performances.append(performance)
            mean_ratio += ratio
            total_length += len(img_files)
            ratio /= len(img_files)
            ratios.append(ratio)
    mean_performance /= mean_ratio
    mean_ratio /= total_length
    return (
        ratios,
        mean_ratio,
        performances,
        mean_performance,
        video_perf,
        dataset.dataset.seq_names,
        video_ranks,
    )


def anchor_ratio(dataset, vot_un=False):
    ratios = []
    if isinstance(dataset, ExperimentVOT):
        for s, (img_files, anno, _) in enumerate(dataset.dataset):
            ratio = 0
            seq_name = dataset.dataset.seq_names[s]
            if vot_un:
                anchor_file = os.path.join(
                    dataset.result_dir,
                    "Ours",
                    "unsupervised",
                    seq_name,
                    "%s_anchor.pkl" % seq_name,
                )
            else:
                anchor_file = os.path.join(
                    dataset.result_dir,
                    "Ours",
                    "baseline",
                    seq_name,
                    "%s_anchor.pkl" % seq_name,
                )
            with open(anchor_file, "rb") as f:
                anchors = pickle.load(f)
            for f in range(1, len(img_files) - 1):
                if anchors[f] is not None:
                    ratio += 1
            ratio /= len(img_files)
            ratios.append(ratio)
    else:
        for s, (img_files, anno) in enumerate(dataset.dataset):
            ratio = 0
            seq_name = dataset.dataset.seq_names[s]
            anchor_file = os.path.join(
                dataset.result_dir, "Ours", "anchor/%s_anchor.pkl" % seq_name
            )
            with open(anchor_file, "rb") as f:
                anchors = pickle.load(f)
            for f in range(1, len(img_files) - 1):
                if anchors[f] is not None:
                    ratio += 1
            ratio /= len(img_files)
            ratios.append(ratio)
    return ratios, dataset.dataset.seq_names


def get_performance(dataset, tracker, filename=None):
    report_dir = os.path.join(dataset.report_dir, tracker)
    report_file = os.path.join(
        report_dir, "performance.json" if filename is None else filename
    )
    with open(report_file) as f:
        performance = json.load(f)
    performance["SiamFC_Res22"] = performance["SiamFC_Plus"]
    return performance


def get_all_videos(datasets, trackers):
    succ_curve = {}
    prec_curve = {}
    seq_names = []
    for name in trackers:
        succ_curve[name] = []
        prec_curve[name] = []
    for dataset in datasets:
        if isinstance(dataset, ExperimentVOT):
            performance = get_performance(dataset, trackers[0], "performance_un.json")
            for s in range(len(dataset.dataset)):
                seq_name = dataset.dataset.seq_names[s]
                if seq_name not in [
                    "basketball",
                    "bolt1",
                    "bolt2",
                    "car1",
                    "girl",
                    "matrix",
                    "pedestrian1",
                    "shaking",
                    "singer2",
                    "soccer1",
                    "tiger",
                ]:
                    for name in trackers:
                        item = performance[name]["seq_wise"][seq_name]
                        succ_curve[name].append(item["success_curve"])
                        prec_curve[name].append(item["precision_curve"])
                    seq_names.append(seq_name)
        elif isinstance(dataset, ExperimentTColor128):
            performance = get_performance(dataset, trackers[0])
            for s in range(len(dataset.dataset)):
                seq_name = dataset.dataset.seq_names[s]
                if "_ce" in seq_name:
                    for name in trackers:
                        item = performance[name]["seq_wise"][seq_name]
                        succ_curve[name].append(item["success_curve"])
                        prec_curve[name].append(item["precision_curve"])
                    seq_names.append(seq_name)
        else:
            performance = get_performance(dataset, trackers[0])
            for s in range(len(dataset.dataset)):
                seq_name = dataset.dataset.seq_names[s]
                for name in trackers:
                    item = performance[name]["seq_wise"][seq_name]
                    succ_curve[name].append(item["success_curve"])
                    prec_curve[name].append(item["precision_curve"])
                seq_names.append(seq_name)

    mean_succ_curve = {}
    mean_prec_curve = {}

    for name in trackers:
        mean_succ_curve[name] = np.mean(succ_curve[name], axis=0)
        mean_prec_curve[name] = np.mean(prec_curve[name], axis=0)

    return succ_curve, prec_curve, mean_succ_curve, mean_prec_curve, seq_names


if __name__ == "__main__":
    root = "."
    trackers = [
        "DaSiamRPN",
        "ECO-H",
        "MCCT-H",
        "SiamDW",
        "SiamFC",
        "SiamRPN",
        "SiamMask",
        "Average",
        "MCCT",
        "Ours",
    ]
    result_folder = "results"
    report_folder = "reports"

    otbs = []
    otb_labels = []

    otb = ExperimentOTB(
        "/home/heonsong/Disk2/Dataset/OTB",
        version=2013,
        result_dir=root + "/%s" % result_folder,
        report_dir=root + "/%s" % report_folder,
    )
    otb.result_dir = os.path.join(root + "/%s" % result_folder, "OTB2015")
    otbs.append(otb)
    otb_labels.append("OTB2013")

    otb = ExperimentOTB(
        "/home/heonsong/Disk2/Dataset/OTB",
        version=2015,
        result_dir=root + "/%s" % result_folder,
        report_dir=root + "/%s" % report_folder,
    )
    otbs.append(otb)
    otb_labels.append("OTB2015")

    otb = ExperimentTColor128(
        "/home/heonsong/Disk2/Dataset/TColor128",
        result_dir=root + "/%s" % result_folder,
        report_dir=root + "/%s" % report_folder,
    )
    otbs.append(otb)
    otb_labels.append("TColor128")

    vots = []
    vot_labels = []

    vot = ExperimentVOT(
        "/home/heonsong/Disk2/Dataset/VOT2018",
        version=2018,
        result_dir=root + "/%s" % result_folder,
        report_dir=root + "/%s" % report_folder,
    )
    vot.repetitions = 1
    vots.append(vot)
    vot_labels.append("VOT2018")

    # evaluation performance
    for otb, label in zip(otbs, otb_labels):
        otb_eval(otb, trackers)
        print("Evaluated %s" % label)
    for vot, label in zip(vots, vot_labels):
        vot_eval(vot, trackers)
        vot_un_eval(vot, trackers)
        print("Evaluated %s" % label)
