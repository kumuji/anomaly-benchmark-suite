from __future__ import absolute_import, division, print_function

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


install_package("pycocotools==2.0.7")

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def filter_files(label_path, predictions_path, tmp_predictions_path):
    with open(predictions_path, "r") as f:
        predictions = json.load(f)
    with open(label_path, "r") as f:
        label = json.load(f)
    filtered_predictions = list()
    label_filenames = [file["id"] for file in label["images"]]
    print(f"Number of bounding box predictions: {len(predictions)}")
    for prediction in predictions:
        if prediction["image_id"] in label_filenames:
            filtered_predictions.append(prediction)
    print(f"Number of bounding box predictions for the GT: {len(filtered_predictions)}")
    filtered_predictions_path = Path(tmp_predictions_path) / predictions_path.name
    with open(filtered_predictions_path, "w") as f:
        json.dump(filtered_predictions, f)
    return filtered_predictions_path


def compute_classification_metrics(cocoGt, cocoEval, threshold=0.5):
    imgids = cocoGt.getImgIds()
    tp_arr = []
    fp_arr = []
    fn_arr = []
    ffp = 0
    for i in imgids:
        # get iou score
        iou = cocoEval.computeIoU(imgId=i, catId=1)

        # get count of gt for a given category
        gt_len = len(cocoEval._gts[i, 1])
        # dt len
        dt_len = len(cocoEval._dts[i, 1])
        ffp += dt_len

        if len(iou) == 0:
            fn = gt_len
            fn_arr.append(fn)
            tp = 0
            tp_arr.append(tp)
            fp = np.maximum(dt_len - tp, 0)
            fp_arr.append(fp)
        else:
            tp = np.sum(iou > threshold)
            if tp > gt_len:
                tpp = min(tp, gt_len)
                tp_arr.append(tpp)
                fn = np.maximum(gt_len - tpp, 0)
                fn_arr.append(fn)
                fp = np.maximum(dt_len - tpp, 0)
                fp_arr.append(fp)
            else:
                tp_arr.append(tp)
                # FN
                fn = np.maximum(gt_len - tp, 0)
                fn_arr.append(fn)
                # FP
                fp = np.maximum(dt_len - tp, 0)
                fp_arr.append(fp)
    tp = sum(tp_arr)
    fp = sum(fp_arr)
    fn = sum(fn_arr)
    ffp /= len(imgids)
    return tp, fp, fn, ffp


def main(submit_path, labels_path, output_path):
    output_path = Path(output_path)
    submit_path = Path(submit_path)
    labels_path = Path(labels_path)

    version = "v0.0.1"

    if not submit_path.is_dir:
        print(f"{submit_path} doesn't exist")

    if submit_path.is_dir() and labels_path.is_dir():
        if not output_path.exists():
            output_path.mkdir()

    output_filename = output_path / "scores.txt"

    print(
        "********************************************************************************"
    )
    print("INTERFACE:")
    print(f"Python Version: {sys.version}")
    print(f"Data: {labels_path}")
    print(f"Predictions: {submit_path}")
    print(f"Scoring Version: {version}")
    print(f"Codalab: {output_filename}")
    print(
        "********************************************************************************"
    )
    metric_keys = [
        "ap",
        "ap50",
        "ap75",
        "aps",
        "apm",
        "apl",
        "ar1",
        "ar10",
        "ar100",
        "ars",
        "arm",
        "arl",
    ]

    results = dict()
    with tempfile.TemporaryDirectory() as tmp_predictions_dir:
        for data in ["fishyscapes"]:
            print(f"Evaluating {data}")

            gt_json_path = Path(labels_path) / f"{data}_label.json"
            prediction_json_path = Path(submit_path) / f"{data}.json"

            cocoGt = COCO(str(gt_json_path))
            if data == "fishyscapes":
                prediction_json_path = filter_files(
                    gt_json_path, prediction_json_path, tmp_predictions_dir
                )
            cocoDt = cocoGt.loadRes(str(prediction_json_path))
            # running evaluation
            cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            results[data] = {
                key: cocoEval.stats[i] for i, key in enumerate(metric_keys)
            }
            (
                results[data]["tp50"],
                results[data]["fp50"],
                results[data]["fn50"],
                results[data]["ppf"],
            ) = compute_classification_metrics(cocoGt, cocoEval, threshold=0.5)

    metric_keys.extend(
        [
            "tp50",
            "fp50",
            "fn50",
            "ppf",
        ]
    )
    fslaf_coeff = 1
    results["unified"] = dict()
    for key in metric_keys:
        results["unified"][key] = results["fishyscapes"][key] * fslaf_coeff
    ret = {
        "AP": results["unified"]["ap"] * 100,
        "AP50": results["unified"]["ap50"] * 100,
        "AR1": results["unified"]["ar1"] * 100,
        "AR10": results["unified"]["ar10"] * 100,
        "AR100": results["unified"]["ar100"] * 100,
    }
    with open(output_filename, "w") as file:
        for k, v in ret.items():
            file.write(f"{k}: {v}\n")
    print(f"Results :{results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation Script For Detection Benchmark"
    )

    parser.add_argument("submit_path", help="Path to the submission file")
    parser.add_argument("labels_path", help="Path to the labels file")
    parser.add_argument("output_path", help="Path to the output file")

    args = parser.parse_args()
    main(args.submit_path, args.labels_path, args.output_path)
