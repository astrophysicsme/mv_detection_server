import torch
from torch.utils.data import DataLoader
import numpy as np

from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import status
from pydantic import BaseModel
from torchvision.ops import nms

from bootstrap import loaded_models
import lib.dataset.dataset_factory as dataset_factory
from lib.dataset.collate import collate_test

from configurations import cfg

iou_threshold = 0.1


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    threat_id: int
    threat_category: str


def calc_iou(gt_bbox, pred_bbox):
    """
    This function takes the predicted bounding box and ground truth bounding box and
    return the IoU ratio
    """
    x_top_left_gt, y_top_left_gt, x_bottom_right_gt, y_bottom_right_gt = gt_bbox
    x_top_left_p, y_top_left_p, x_bottom_right_p, y_bottom_right_p = pred_bbox

    if (x_top_left_gt > x_bottom_right_gt) or (y_top_left_gt > y_bottom_right_gt):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_top_left_p > x_bottom_right_p) or (y_top_left_p > y_bottom_right_p):
        raise AssertionError("Predicted Bounding Box is not correct", x_top_left_p, y_bottom_right_p, y_top_left_p,
                             y_bottom_right_gt)

    # if the GT bbox and predicted BBox do not overlap then iou=0
    # If bottom right of x-coordinate GT bbox is less than or above the top left of x coordinate of the predicted BBox
    if x_bottom_right_gt < x_top_left_p:
        return 0.0
    # If bottom right of y-coordinate GT bbox is less than or above the top left of y coordinate of the predicted BBox
    if y_bottom_right_gt < y_top_left_p:
        return 0.0
    # If bottom right of x-coordinate GT bbox is greater than or below the bottom right of x coordinate of
    # the predicted BBox
    if x_top_left_gt > x_bottom_right_p:
        return 0.0
    # If bottom right of y-coordinate GT bbox is greater than or below the bottom right of y coordinate of
    # the predicted BBox
    if y_top_left_gt > y_bottom_right_p:
        return 0.0

    gt_bbox_area = (x_bottom_right_gt - x_top_left_gt + 1) * (y_bottom_right_gt - y_top_left_gt + 1)
    predicted_bbox_area = (x_bottom_right_p - x_top_left_p + 1) * (y_bottom_right_p - y_top_left_p + 1)

    x_top_left = np.max([x_top_left_gt, x_top_left_p])
    y_top_left = np.max([y_top_left_gt, y_top_left_p])
    x_bottom_right = np.min([x_bottom_right_gt, x_bottom_right_p])
    y_bottom_right = np.min([y_bottom_right_gt, y_bottom_right_p])

    intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left + 1)

    union_area = (gt_bbox_area + predicted_bbox_area - intersection_area)

    return intersection_area / union_area


def astro_nms(bounding_boxes, nt):
    if len(bounding_boxes) == 0:
        return [], []
    bboxes = np.array(bounding_boxes)
    #
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #
    order = np.argsort(scores)
    picked_boxes = []
    while order.size > 0:
        index = order[-1]
        picked_boxes.append(bounding_boxes[index])

        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])
        w = np.maximum(0.0, x22 - x11 + 1)
        h = np.maximum(0.0, y22 - y11 + 1)
        intersection = w * h

        ious = intersection / (areas[index] + areas[order[:-1]] - intersection)
        left = np.where(ious < nt)
        order = order[left]
    return np.array(picked_boxes)


def check_inside(picked_boxes, bbox):
    for i in picked_boxes:
        if bbox in picked_boxes[i]:
            return i
    return 0


def astro_collect(bounding_boxes, nt=0.7):
    if len(bounding_boxes) == 0:
        return {}

    picked_boxes = {}

    bboxes = np.array(bounding_boxes)

    x1 = bboxes[:, 2]
    y1 = bboxes[:, 3]
    x2 = bboxes[:, 4]
    y2 = bboxes[:, 5]
    scores = bboxes[:, 6]

    areas = (x2.astype(int) - x1.astype(int) + 1) * (y2.astype(int) - y1.astype(int) + 1)

    order = np.argsort(scores.astype(float))
    count = 1
    for i in range(len(order)):
        for j in range(i + 1, len(order)):

            index1 = order[i]
            index2 = order[j]

            x11 = np.maximum(x1[index1].astype(int), x1[index2].astype(int))
            y11 = np.maximum(y1[index1].astype(int), y1[index2].astype(int))
            x22 = np.minimum(x2[index1].astype(int), x2[index2].astype(int))
            y22 = np.minimum(y2[index1].astype(int), y2[index2].astype(int))
            w = np.maximum(0.0, x22 - x11 + 1)
            h = np.maximum(0.0, y22 - y11 + 1)
            intersection = w * h
            #
            ious = intersection / (areas[index1] + areas[index2] - intersection)
            if ious > nt:  # two boxes overlap
                if check_inside(picked_boxes, bounding_boxes[index1]):
                    picked_boxes[check_inside(picked_boxes, bounding_boxes[index1])].append(bounding_boxes[index2])
                elif check_inside(picked_boxes, bounding_boxes[index2]):
                    picked_boxes[check_inside(picked_boxes, bounding_boxes[index2])].append(bounding_boxes[index1])
                else:
                    picked_boxes[count] = [bounding_boxes[index1], bounding_boxes[index2]]
                    count += 1

    return picked_boxes


def get_vote(coordinates, next_view):
    local_vote = 0

    for x in range(0, len(next_view)):
        next_coordinates = [
            int(next_view[x]["x1"]),
            int(next_view[x]["y1"]),
            int(next_view[x]["x2"]),
            int(next_view[x]["y2"])
        ]
        voting_iou = calc_iou(coordinates, next_coordinates)

        if voting_iou >= iou_threshold:
            local_vote += 1

    return local_vote


def check_consecutive_views(current_view, first_view, second_view):
    picked_boxes = []
    for t in range(0, len(current_view)):
        vote = 0
        current_coordinates = [
            int(current_view[t]["x1"]),
            int(current_view[t]["y1"]),
            int(current_view[t]["x2"]),
            int(current_view[t]["y2"])
        ]

        if get_vote(current_coordinates, first_view) > 0:
            vote += 1

        if get_vote(current_coordinates, second_view) > 0:
            vote += 1

        if vote == 2:
            picked_boxes.append(current_view[t])

    return picked_boxes


def merge_bounding_boxes(bounding_boxes):
    if len(bounding_boxes) == 0:
        return [], []
    bboxes = np.array(bounding_boxes)

    min_x1, min_y1 = np.minimum([bboxes[0][0], bboxes[0][1]], [bboxes[1][0], bboxes[1][1]])
    max_x2, max_y2 = np.maximum([bboxes[0][2], bboxes[0][3]], [bboxes[1][2], bboxes[1][3]])

    average_score = (bboxes[0][4] + bboxes[1][4]) / 2

    return [int(min_x1), int(min_y1), int(max_x2), int(max_y2), float(average_score)]


def calculate_nms(bounding_boxes, nt):
    if len(bounding_boxes) == 0:
        return [], []
    bboxes = np.array(bounding_boxes)
    #
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #
    order = np.argsort(scores)
    picked_boxes = []  #
    while order.size > 0:
        #
        nms_index = order[-1]
        picked_boxes.append(bounding_boxes[nms_index])
        #
        x11 = np.maximum(x1[nms_index], x1[order[:-1]])
        y11 = np.maximum(y1[nms_index], y1[order[:-1]])
        x22 = np.minimum(x2[nms_index], x2[order[:-1]])
        y22 = np.minimum(y2[nms_index], y2[order[:-1]])
        w = np.maximum(0.0, x22 - x11 + 1)
        h = np.maximum(0.0, y22 - y11 + 1)
        intersection = w * h
        #
        ious = intersection / (areas[nms_index] + areas[order[:-1]] - intersection)
        left = np.where(ious < nt)
        order = order[left]
    return np.array(picked_boxes)


def add_or_append_dict(idx, box, boxes):
    if str(idx) not in boxes.keys():
        boxes[str(idx)] = []

    if box not in boxes[str(idx)]:
        boxes[str(idx)].append(box)


def inspect_single_image():
    assert len(loaded_models) != 0

    image_dir = cfg.SERVER.IMAGES_FOR_INSPECTION_DIRECTORY

    device = cfg.SERVER.DEVICE
    classes = cfg.SERVER.CLASSES

    standard_params = {
        "img_ext": ".bmp",
        "image_path": image_dir,
        "classes": cfg.SERVER.CLASSES
    }

    results = {}

    for loaded_model in loaded_models:
        model_params = {}
        if loaded_model["additional_parameters"] is not None:
            model_params = {**loaded_model["additional_parameters"]}

        add_params = {**standard_params, **model_params}

        dataset, _ = dataset_factory.get_dataset('detect', add_params, mode='test')
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_test, num_workers=0)

        for i, data in enumerate(loader):
            image_id = str(dataset[i]["id"])
            if image_id not in results:
                results[image_id] = []

            image_data = data[0].to(device)
            image_info = data[1].to(device)

            with torch.no_grad():
                detected_class, bbox_predicted, *_ = loaded_model["model"](image_data, image_info, None)

            bbox_predicted /= image_info[0][2].item()

            scores = detected_class.squeeze()
            bbox_predicted = bbox_predicted.squeeze()

            for j in range(1, len(classes)):
                inds = torch.nonzero(scores[:, j] > 0.05).view(-1)
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, dim=0, descending=True)

                    cls_boxes = bbox_predicted[inds][:, j * 4:(j + 1) * 4].contiguous()

                    detected_class = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    detected_class = detected_class[order]
                    keep = nms(cls_boxes, cls_scores, cfg.TEST.NMS)
                    detected_class = detected_class[keep.view(-1).long()]

                    score_threshold = cfg.SERVER.MODEL_THRESHOLD
                    use_det = detected_class.cpu().numpy()
                    if use_det.shape[0] < 10:
                        use_det = astro_nms(use_det, 0.3)
                    else:
                        use_det = astro_nms(use_det[:10], 0.3)
                    for k in range(np.minimum(10, use_det.shape[0])):
                        bbox = tuple(int(np.round(x)) for x in use_det[k, :4])
                        score = use_det[k, -1]
                        if score > score_threshold:
                            detected_box = [str(image_id), str(loaded_model["name"]), int(bbox[0]), int(bbox[1]),
                                            int(bbox[2]), int(bbox[3]), float(score), int(j), str(classes[j])]
                            results[image_id].append(detected_box)

    voting_threshold = int(0.4 * len(loaded_models))

    combined_res = {}
    pallet_name = ""

    for file_name in results:
        file_id, view_id = file_name.split('v')
        pallet_name = file_id

        combined_res[view_id] = []

        result = astro_collect(results[file_name], nt=0.1)
        for j in result:
            if len(result[j]) >= voting_threshold:
                combined_boxes = np.array(result[j])
                combined_x1 = combined_boxes[:, 2].astype(int).sum(0) / len(result[j])
                combined_y1 = combined_boxes[:, 3].astype(int).sum(0) / len(result[j])
                combined_x2 = combined_boxes[:, 4].astype(int).sum(0) / len(result[j])
                combined_y2 = combined_boxes[:, 5].astype(int).sum(0) / len(result[j])
                combined_score = combined_boxes[:, 6].astype(float).sum(0) / len(result[j])

                combined_res[view_id].append(jsonable_encoder(
                    BoundingBox(x1=combined_x1, y1=combined_y1, x2=combined_x2, y2=combined_y2,
                                score=combined_score, threat_id=combined_boxes[0][7],
                                threat_category=combined_boxes[0][8])))

    optimized_results = {
        "id": pallet_name,
        "Views": []
    }
    # start_views_position = 1
    # for optimized_view_id in range(0, len(combined_res), cfg.SERVER.VIEWS_PER_PASS):
    #     end_views_position = start_views_position + cfg.SERVER.VIEWS_PER_PASS
    #     numbers = [k for k in range(start_views_position, end_views_position)]
    #     start_views_position += cfg.SERVER.VIEWS_PER_PASS
    #
    #     pass_boxes = {}
    #
    #     for box in check_consecutive_views(combined_res[str(numbers[0])], combined_res[str(numbers[1])],
    #                                        combined_res[str(numbers[2])]):
    #         if str(numbers[0]) not in pass_boxes.keys():
    #             pass_boxes[str(numbers[0])] = []
    #
    #         if box not in pass_boxes[str(numbers[0])]:
    #             pass_boxes[str(numbers[0])].append(box)
    #     for box in check_consecutive_views(combined_res[str(numbers[1])], combined_res[str(numbers[0])],
    #                                        combined_res[str(numbers[2])]):
    #         if str(numbers[1]) not in pass_boxes.keys():
    #             pass_boxes[str(numbers[1])] = []
    #
    #         if box not in pass_boxes[str(numbers[1])]:
    #             pass_boxes[str(numbers[1])].append(box)
    #     for box in check_consecutive_views(combined_res[str(numbers[1])], combined_res[str(numbers[2])],
    #                                        combined_res[str(numbers[3])]):
    #         if str(numbers[1]) not in pass_boxes.keys():
    #             pass_boxes[str(numbers[1])] = []
    #
    #         if box not in pass_boxes[str(numbers[1])]:
    #             pass_boxes[str(numbers[1])].append(box)
    #     for box in check_consecutive_views(combined_res[str(numbers[2])], combined_res[str(numbers[0])],
    #                                        combined_res[str(numbers[1])]):
    #         if str(numbers[2]) not in pass_boxes.keys():
    #             pass_boxes[str(numbers[2])] = []
    #         if box not in pass_boxes[str(numbers[2])]:
    #             pass_boxes[str(numbers[2])].append(box)
    #     for box in check_consecutive_views(combined_res[str(numbers[2])], combined_res[str(numbers[1])],
    #                                        combined_res[str(numbers[3])]):
    #         if str(numbers[2]) not in pass_boxes.keys():
    #             pass_boxes[str(numbers[2])] = []
    #
    #         if box not in pass_boxes[str(numbers[2])]:
    #             pass_boxes[str(numbers[2])].append(box)
    #     for box in check_consecutive_views(combined_res[str(numbers[2])], combined_res[str(numbers[3])],
    #                                        combined_res[str(numbers[4])]):
    #         if str(numbers[2]) not in pass_boxes.keys():
    #             pass_boxes[str(numbers[2])] = []
    #
    #         if box not in pass_boxes[str(numbers[2])]:
    #             pass_boxes[str(numbers[2])].append(box)
    #     for box in check_consecutive_views(combined_res[str(numbers[3])], combined_res[str(numbers[1])],
    #                                        combined_res[str(numbers[2])]):
    #         if str(numbers[3]) not in pass_boxes.keys():
    #             pass_boxes[str(numbers[3])] = []
    #
    #         if box not in pass_boxes[str(numbers[3])]:
    #             pass_boxes[str(numbers[3])].append(box)
    #     for box in check_consecutive_views(combined_res[str(numbers[3])], combined_res[str(numbers[2])],
    #                                        combined_res[str(numbers[4])]):
    #         if str(numbers[3]) not in pass_boxes.keys():
    #             pass_boxes[str(numbers[3])] = []
    #
    #         if box not in pass_boxes[str(numbers[3])]:
    #             pass_boxes[str(numbers[3])].append(box)
    #     for box in check_consecutive_views(combined_res[str(numbers[3])], combined_res[str(numbers[4])],
    #                                        combined_res[str(numbers[5])]):
    #         if str(numbers[3]) not in pass_boxes.keys():
    #             pass_boxes[str(numbers[3])] = []
    #
    #         if box not in pass_boxes[str(numbers[3])]:
    #             pass_boxes[str(numbers[3])].append(box)
    #     for box in check_consecutive_views(combined_res[str(numbers[4])], combined_res[str(numbers[2])],
    #                                        combined_res[str(numbers[3])]):
    #         if str(numbers[4]) not in pass_boxes.keys():
    #             pass_boxes[str(numbers[4])] = []
    #
    #         if box not in pass_boxes[str(numbers[4])]:
    #             pass_boxes[str(numbers[4])].append(box)
    #     for box in check_consecutive_views(combined_res[str(numbers[4])], combined_res[str(numbers[3])],
    #                                        combined_res[str(numbers[5])]):
    #         if str(numbers[4]) not in pass_boxes.keys():
    #             pass_boxes[str(numbers[4])] = []
    #
    #         if box not in pass_boxes[str(numbers[4])]:
    #             pass_boxes[str(numbers[4])].append(box)
    #     for box in check_consecutive_views(combined_res[str(numbers[5])], combined_res[str(numbers[3])],
    #                                        combined_res[str(numbers[4])]):
    #         if str(numbers[5]) not in pass_boxes.keys():
    #             pass_boxes[str(numbers[5])] = []
    #
    #         if box not in pass_boxes[str(numbers[5])]:
    #             pass_boxes[str(numbers[5])].append(box)
    #
    #     for key in pass_boxes:
    #         optimized_results["Views"].append({
    #             "viewid": int(key),
    #             "Box": pass_boxes[key]
    #         })

    for pass_number in range(0, cfg.SERVER.NUMBER_OF_PASSES):
        start_views_position = pass_number * cfg.SERVER.VIEWS_PER_PASS + 1
        end_views_position = start_views_position + cfg.SERVER.VIEWS_PER_PASS
        for view_number in range(start_views_position, end_views_position):
            pass_boxes = {}
            if view_number == start_views_position:
                for box in check_consecutive_views(combined_res[str(view_number)], combined_res[str(view_number + 1)],
                                                   combined_res[str(view_number + 2)]):
                    add_or_append_dict(view_number, box, pass_boxes)
            elif view_number == (start_views_position + 1):
                if view_number + 1 <= end_views_position - 1:
                    for box in check_consecutive_views(combined_res[str(view_number)],
                                                       combined_res[str(view_number - 1)],
                                                       combined_res[str(view_number + 1)]):
                        add_or_append_dict(view_number, box, pass_boxes)

                if view_number + 2 <= end_views_position - 1:
                    for box in check_consecutive_views(combined_res[str(view_number)],
                                                       combined_res[str(view_number + 1)],
                                                       combined_res[str(view_number + 2)]):
                        add_or_append_dict(view_number, box, pass_boxes)
            elif (start_views_position + 1) < view_number < (end_views_position - 2):
                for box in check_consecutive_views(combined_res[str(view_number)], combined_res[str(view_number - 2)],
                                                   combined_res[str(view_number - 1)]):
                    add_or_append_dict(view_number, box, pass_boxes)

                if view_number + 1 <= end_views_position - 1:
                    for box in check_consecutive_views(combined_res[str(view_number)],
                                                       combined_res[str(view_number - 1)],
                                                       combined_res[str(view_number + 1)]):
                        add_or_append_dict(view_number, box, pass_boxes)

                if view_number + 2 <= end_views_position + 1:
                    for box in check_consecutive_views(combined_res[str(view_number)],
                                                       combined_res[str(view_number + 1)],
                                                       combined_res[str(view_number + 2)]):
                        add_or_append_dict(view_number, box, pass_boxes)
            elif view_number == (end_views_position - 2):
                for box in check_consecutive_views(combined_res[str(view_number)], combined_res[str(view_number - 2)],
                                                   combined_res[str(view_number - 1)]):
                    add_or_append_dict(view_number, box, pass_boxes)

                if view_number + 1 <= end_views_position + 1:
                    for box in check_consecutive_views(combined_res[str(view_number)],
                                                       combined_res[str(view_number - 1)],
                                                       combined_res[str(view_number + 1)]):
                        add_or_append_dict(view_number, box, pass_boxes)
            elif view_number == (end_views_position - 1):
                for box in check_consecutive_views(combined_res[str(view_number)], combined_res[str(view_number - 2)],
                                                   combined_res[str(view_number - 1)]):
                    add_or_append_dict(view_number, box, pass_boxes)

            for key in pass_boxes:
                optimized_results["Views"].append({
                    "viewid": int(key),
                    "Box": pass_boxes[key]
                })

        start_views_position += cfg.SERVER.VIEWS_PER_PASS

    return JSONResponse(status_code=status.HTTP_200_OK, content={"Scan": jsonable_encoder(optimized_results)})
