#!/usr/bin/env python
from __future__ import print_function, absolute_import, division
import argparse
from pathlib import Path
import numpy as np
from collections import namedtuple
import tempfile

import evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval
from preprocess_files import prepare_submitted_files


def main(submit_path, labels_path , output_path):
    with tempfile.TemporaryDirectory() as postprocessed_files:
        postprocessed_files = Path(postprocessed_files)

        output_path = Path(output_path)
        submit_path = Path(submit_path)
        labels_path = Path(labels_path)

        if not submit_path.is_dir:
            print(f"{submit_path} doesn't exist")

        if submit_path.is_dir() and labels_path.is_dir():
            if not output_path.exists():
                output_path.mkdir()

        prepare_submitted_files(
            submission_path=submit_path, temp_folder=postprocessed_files
        )
        output_filename = output_path / "scores.txt"

        CsFile = namedtuple(
            "csFile", ["city", "sequenceNb", "frameNb", "type", "type2", "ext"]
        )

        def get_fs_file_info(parts):
            parts = parts.name
            parts = parts.split("_")
            parts = parts[:-1] + parts[-1].split(".")
            city, rest = parts[:-5], parts[-5:]
            city = ["_".join(city)]
            city.extend(rest)
            return CsFile(*city)

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = str(
            postprocessed_files.resolve().absolute()
        )
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.gtInstancesFile = str(labels_path / "gtinstances.json")
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.minRegionSizes = np.array([10, 10, 10])
        cityscapes_eval.args.quiet = True
        cityscapes_eval.getCsFileInfo = get_fs_file_info

        groundTruthImgList = sorted(list(labels_path.glob("*.png")))
        groundTruthImgList = [path.resolve().absolute() for path in groundTruthImgList]
        if len(groundTruthImgList) == 0:
            print(
                f"Cannot find any ground truth images to use for evaluation. Searched for: {cityscapes_eval.args.groundTruthSearch}"
            )

        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(
                cityscapes_eval.getPrediction(gt, cityscapes_eval.args)
            )
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )["averages"]

        ret = {
            "AP": results["allAp"] * 100,
            "AP50": results["allAp50%"] * 100,
        }
        with open(output_filename, "w") as file:
            for k, v in ret.items():
                file.write(f"{k}: {v}\n")
        print(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script based on CityScapesScripts")

    parser.add_argument("submit_path", help="Path to the submission file")
    parser.add_argument("labels_path", help="Path to the labels file")
    parser.add_argument("output_path", help="Path to the output file")

    args = parser.parse_args()
    main(args.submit_path, args.labels_path , args.output_path)
