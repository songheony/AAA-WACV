import copy
import random
import sys
import numpy as np
from scipy.spatial import distance
import scipy.special as sc
import networkx as nx
from got10k.trackers import Tracker
import torch
from torchvision import models
from torchvision import transforms


class WAADelayed:
    def __init__(self, n):
        self.w = np.ones(n) / n

    """
    gradient_losses should be n X len(dt)
    """

    def update(self, gradient_losses, lr):
        np_gradient_losses = np.array(gradient_losses)
        # check the number of element
        assert np_gradient_losses.shape[0] == self.w.shape[0]

        changes = lr * np_gradient_losses.sum(axis=1)
        temp = np.log(self.w + sys.float_info.min) - changes
        self.w = np.exp(temp - sc.logsumexp(temp))


class Node:
    def __init__(self, name, data):
        self.name = name
        self.data = data


class NetworkFlow:
    def __init__(self):
        self.G = nx.DiGraph()

    def reset_graph(self, width, height, init_bbox, init_feature, template):
        self.layer = 0
        self.nodes_dict = {}

        self.G.clear()

        start_node = Node("s", None)
        self.G.add_node("s", demand=-1)

        self.nodes = [[start_node]]

        self.max_dist = distance.euclidean((0, 0), (width, height))

        self.add_detections([init_bbox], [init_feature])

        self.init_node = Node("i", {"layer": None, "bbox": None, "feature": template})

    def add_detections(self, bboxes, features):
        nodes = []
        for num, bbox, feature in zip(range(len(bboxes)), bboxes, features):
            if feature is None:
                continue
            node_name = "L:%d/N:%d" % (self.layer, num)
            data = {"layer": self.layer, "bbox": np.array(bbox), "feature": feature}
            node = Node(node_name, data)
            nodes.append(node)
        self._add_nodes(nodes)

    def get_path(self):
        new_G = self._add_end_node()
        flow_dict = nx.min_cost_flow(new_G)

        path = []
        before_node_name = "s"
        while True:
            edges = flow_dict[before_node_name]
            for node_name, flow in edges.items():
                if flow == 1:
                    if node_name == "e":
                        return path
                    node = self.nodes_dict[node_name]
                    path.append(node.data["bbox"])
                    before_node_name = node_name
                    break

    def _add_end_node(self):
        new_G = copy.deepcopy(self.G)
        last_nodes = self.nodes[-1]

        end_node = Node("e", None)
        new_G.add_node("e", demand=1)

        for last_node in last_nodes:
            weight = int(self._get_cost(last_node, end_node))
            new_G.add_edge(last_node.name, end_node.name, weight=weight, capacity=1)

        return new_G

    def _add_nodes(self, nodes):
        last_nodes = self.nodes[-1]

        if len(nodes) > 0:
            for node in nodes:
                self.nodes_dict[node.name] = node
                for last_node in last_nodes:
                    weight = int(self._get_cost(last_node, node))
                    self.G.add_edge(
                        last_node.name, node.name, weight=weight, capacity=1
                    )
            self.nodes.append(nodes)
        else:
            node_name = str(self.layer) + "X"
            non_node = Node(
                node_name, {"layer": self.layer, "bbox": None, "feature": None}
            )
            self.nodes_dict[node_name] = non_node
            for last_node in last_nodes:
                weight = 0
                self.G.add_edge(last_node.name, node_name, weight=weight, capacity=1)
            self.nodes.append([non_node])
        self.layer += 1

    def _get_cost(self, node1, node2):
        if node1.name == "s" or node2.name == "e" or node1.data["bbox"] is None:
            cost = 0
        else:
            overlap_cost = self._overlap_cost(node1, node2)[0] * 100
            appearance_cost = self._appearance_cost(node1, node2) * 1000
            patch_cost = self._appearance_cost(self.init_node, node2) * 1000
            distance_cost = self._distance_cost(node1, node2) * 100
            cost = overlap_cost + appearance_cost + patch_cost + distance_cost
        return cost

    def _distance_cost(self, node1, node2):
        center1 = node1.data["bbox"][:2] + node1.data["bbox"][2:] / 2
        center2 = node2.data["bbox"][:2] + node2.data["bbox"][2:] / 2
        return distance.euclidean(center1, center2) / self.max_dist

    def _overlap_cost(self, node1, node2):
        rect1 = node1.data["bbox"]
        rect2 = node2.data["bbox"]

        if rect1.ndim == 1:
            rect1 = rect1[None, :]
        if rect2.ndim == 1:
            rect2 = rect2[None, :]

        left = np.maximum(rect1[:, 0], rect2[:, 0])
        right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
        top = np.maximum(rect1[:, 1], rect2[:, 1])
        bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

        intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
        union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
        iou = np.clip(intersect / union, 0, 1)
        return 1 - iou

    def _appearance_cost(self, node1, node2):
        x1 = node1.data["feature"]
        x2 = node2.data["feature"]
        return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


class Extractor:
    def __init__(self):
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = self.resnet.cuda()
        self.extractor = self.resnet._modules.get("avgpool")
        self.resnet.eval()
        self.transform = transforms.Compose(
            [
                transforms.Scale((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def extract(self, image, bboxes):
        features = []
        for bbox in bboxes:
            if (
                bbox[0] >= image.size[0]
                or bbox[1] >= image.size[1]
                or bbox[2] <= 0
                or bbox[3] <= 0
            ):
                feature = None
            else:
                max_x = min(image.size[0], bbox[0] + bbox[2])
                max_y = min(image.size[1], bbox[1] + bbox[3])
                min_x = max(0, bbox[0])
                min_y = max(0, bbox[1])
                x = image.crop((min_x, min_y, max_x, max_y))
                x = self._get_vector(x)
                feature = x.data.cpu().numpy()
            features.append(feature)
        return features

    def _get_vector(self, x):
        x = self.transform(x)
        x = x.unsqueeze_(0)
        x = x.cuda()
        my_embedding = torch.zeros(512)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data.view(-1).data)

        h = self.extractor.register_forward_hook(copy_data)
        self.resnet(x)
        h.remove()

        return my_embedding


class AnchorDetector:
    def __init__(self):
        self.init_feature = None

    def init(self, init_feature):
        self.init_feature = init_feature

    def detect(self, features, threshold):
        valid_i = []
        max_score = 0
        for i, feature in enumerate(features):
            if feature is None:
                continue
            score = self._compare_patch(feature)
            if score > threshold and score > max_score:
                valid_i = [i]
                max_score = score
        if len(valid_i) == 0:
            return None
        else:
            return valid_i

    def _compare_patch(self, feature):
        x1 = self.init_feature
        x2 = feature
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


class AAA(Tracker):
    def __init__(self, n_experts):
        super(AAA, self).__init__(name="Ours")

        self.fixed = -1
        self.threshold = 0.76
        self.consider_all = False
        self.reset_offline = True
        self.reset_detector = False
        self.reset_tracker = False
        self.distance_loss = False
        self.for_eao = False
        self.offline_tracking = True

        self.network = NetworkFlow()
        self.extractor = Extractor()
        self.anchor_detector = AnchorDetector()

        self.n_experts = n_experts
        self.L = 1

    def init(self, image, box):
        init_feature = self.extractor.extract(image, [box])[0]
        self.anchor_detector.init(init_feature)
        self.network.reset_graph(
            image.size[0], image.size[1], box, init_feature, init_feature
        )

        self.algorithm = WAADelayed(self.n_experts)
        self.delayed_zs = []
        self.est_D = 1
        self.real_D = 0

    def update(self, image, bboxes):
        # add all results to evaluate experts
        self.delayed_zs.append(bboxes)

        # select experts who make right result
        valid_result_idx = [i for i in range(len(bboxes)) if len(bboxes[i]) > 1]
        valid_bboxes = [bboxes[i] for i in valid_result_idx]

        # if all experts make wrong result, the algorithm also makes wrong result
        if len(valid_result_idx) == 0:
            self.network.add_detections([], [])
            result = [0]
            offline_results = None
        else:
            # extract features from valid results
            valid_features = self.extractor.extract(image, valid_bboxes)
            if self.fixed > 0:
                if len(self.delayed_zs) == self.fixed:
                    detected = self.anchor_detector.detect(valid_features, 0)
                else:
                    detected = None
            else:
                detected = self.anchor_detector.detect(valid_features, self.threshold)

            # if the frame is anchor frame
            if detected is not None:
                self.delayed_zs = np.array(self.delayed_zs)

                if self.offline_tracking:
                    # add only anchor result to offline tracker
                    self.network.add_detections(
                        [valid_bboxes[i] for i in detected],
                        [valid_features[i] for i in detected],
                    )

                    # get offline tracking results
                    network_path = self.network.get_path()
                    delayed_ys = np.array(network_path[-len(self.delayed_zs) :])
                    valid_frame_idx = [
                        i for i in range(len(delayed_ys)) if delayed_ys[i] is not None
                    ]
                    if delayed_ys.ndim == 1 and len(valid_frame_idx) == len(delayed_ys):
                        delayed_ys = np.expand_dims(delayed_ys, axis=0)
                else:
                    delayed_ys = self.delayed_zs[:, detected[0], :]

                # select only valid frames for evaluating experts
                valid_frame_idx = [
                    i for i in range(len(delayed_ys)) if delayed_ys[i] is not None
                ]

                # update the weight of experts
                self._update(
                    delayed_ys[valid_frame_idx], self.delayed_zs[valid_frame_idx]
                )

                self.delayed_zs = []

                final_box = delayed_ys[-1]
                final_feature = self.extractor.extract(image, [final_box])[0]

                # if the algorithm reset the anchor detector
                if self.reset_detector:
                    self.anchor_detector.init(final_feature)

                # if the algorithm reset the offline tracker
                if self.reset_offline:
                    # init the offline tracker with anchor result
                    self.network.reset_graph(
                        image.size[0],
                        image.size[1],
                        final_box,
                        final_feature,
                        self.anchor_detector.init_feature,
                    )

                # return anchor result
                result = final_box
                offline_results = delayed_ys
            else:
                # add all result of experts to offline tracker
                self.network.add_detections(valid_bboxes, valid_features)

                # if do not all experts make right result
                if len(valid_result_idx) != self.n_experts:
                    # if the algorithm consider to select also some experts who make wrong result
                    if self.consider_all:
                        idx = self._weighted_random_choice(self.algorithm.w)
                        # return selected expert's result
                        result = bboxes[idx]
                    # except the experts
                    else:
                        if len(valid_result_idx) == 1:
                            idx = 0
                        else:
                            idx = self._weighted_random_choice(
                                self.algorithm.w[valid_result_idx]
                            )
                        # return selected expert's result
                        result = valid_bboxes[idx]
                else:
                    # return aggregated result with experts' result
                    result = np.dot(self.algorithm.w, bboxes)

                offline_results = None

        return result, offline_results

    def _update(self, delayed_ys, delayed_zs):
        """
        delayed_ys = #frames X 4
        delayed_zs = #frames X #experts X 4
        """
        for i in range(1, len(delayed_ys) + 1):
            self.real_D += i
            if self.est_D < self.real_D:
                self.est_D *= 2

        expert_gradient_losses = np.zeros((self.n_experts, len(delayed_ys)))
        for i in range(len(delayed_ys) - 1, -1, -1):
            z = delayed_zs[i]
            y = delayed_ys[i]
            box2 = np.array(y)
            center2 = box2[:2] + box2[2:] / 2
            for ii in range(len(z)):
                box1 = np.array(z[ii])
                center1 = box1[:2] + box1[2:] / 2
                if len(z[ii]) > 1:
                    iou_cost = (1.0 - self._overlap_ratio(box1, box2)) / self.L
                    if self.distance_loss:
                        dist_cost = distance.euclidean(center1, center2) / (
                            self.network.max_dist * self.L
                        )
                        expert_gradient_losses[ii, i] = (iou_cost + dist_cost) / 2.0
                    else:
                        if self.for_eao and iou_cost >= 0.99:
                            for j in range(i, len(delayed_ys)):
                                expert_gradient_losses[ii, j] = 1
                                if j - i >= 4:
                                    break
                        else:
                            expert_gradient_losses[ii, i] = iou_cost
                else:
                    expert_gradient_losses[ii, i] = 1.0 / self.L

        self.algorithm.update(
            expert_gradient_losses, np.sqrt(self.est_D * np.log(self.n_experts))
        )

    def _weighted_random_choice(self, selected_ws):
        maxi = sum(selected_ws)
        pick = random.uniform(0, maxi)
        current = 0
        for key, value in enumerate(selected_ws):
            current += value
            if current >= pick:
                return key

    def _overlap_ratio(self, rect1, rect2):
        if rect1.ndim == 1:
            rect1 = rect1[None, :]
        if rect2.ndim == 1:
            rect2 = rect2[None, :]

        left = np.maximum(rect1[:, 0], rect2[:, 0])
        right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
        top = np.maximum(rect1[:, 1], rect2[:, 1])
        bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

        intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
        union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
        iou = np.clip(intersect / union, 0, 1)
        return iou
