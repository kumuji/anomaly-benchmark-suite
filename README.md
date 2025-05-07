Benchmark suite for [OoDIS Benchmark](https://vision.rwth-aachen.de/oodis)

## For Instance Segmentation

Data could be found [here](https://omnomnom.vision.rwth-aachen.de/data/ugains/fs_lost_found_instance.zip)

```bash
python scoring_program/evaluate.py data/fishyscapes_submission data/fishyscapes ./output
```

```bash
python verify_submission.py --task segmentation --expected_files assets/expected_files.txt ./submission.zip
```

## For Detection
```bash
python scoring_program/evaluate_detection.py data/fishyscapes_submission labels ./output
```

```bash
python verify_submission.py --task detection --expected_files assets/expected_files.txt ./submission.zip
```

For citation of the benchmark use:
```yaml
@inproceedings{nekrasov2025oodis,
 title={{OoDIS: Anomaly Instance Segmentation and Detection Benchmark}},
 author={Nekrasov, Alexey and Zhou, Rui and Ackermann, Miriam and Hermans, Alexander and Leibe, Bastian and Rottmann, Matthias},
 booktitle="International Conference on Robotics and Automation (ICRA)",
 year={2025}
}
```
