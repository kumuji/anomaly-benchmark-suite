Benchmark suite for [OoDIS Benchmark](https://vision.rwth-aachen.de/oodis)

## For Instance Segmentation

Data could be found [here](https://omnomnom.vision.rwth-aachen.de/data/ugains/fs_lost_found_instance.zip)

```bash
python scoring_program/evaluate.py data/fishyscapes_submission data/fishyscapes ./output
```

```bash
python verify_submission.py ./submission.zip
```

## For Detection
```bash
python scoring_program/evaluate_detection.py data/fishyscapes_submission labels ./output
```

```bash
python verify_detection_submission.py ./submission.zip
```
