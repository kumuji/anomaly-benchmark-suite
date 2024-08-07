import argparse
import json
import re
import zipfile
from pathlib import Path

CHECKMARK = "\u2713"


class ValidationException(Exception):
    pass


def verify_submitted_files(
    submission_file: Path, expected_txt_files: Path, task: str
) -> None:
    with zipfile.ZipFile(submission_file) as zip_ref:
        print("  2. Checking directory structure... ", end="", flush=True)

        directories = {
            info.filename.split("/")[0] for info in zip_ref.infolist() if info.is_dir()
        }
        files = {
            info.filename.split("/")[0]
            for info in zip_ref.infolist()
            if not info.is_dir()
        }
        expected_directories = {"fishyscapes", "roadanomaly", "roadobstacle"}
        expected_files = {"fishyscapes.json", "roadanomaly.json", "roadobstacle.json"}
        if task == "segmentation":
            if expected_directories != directories:
                raise ValidationException(
                    f"Directories found {directories}, expected: {expected_directories}"
                )
            print(CHECKMARK)
        if task == "detection":
            if not (expected_files >= files):
                raise ValidationException(
                    f"Files found {files}, expected: {expected_files}"
                )
            print(CHECKMARK)

        print("  3. Checking description file:  ", end="", flush=True)
        if "description.txt" in zip_ref.namelist():
            with zip_ref.open("description.txt") as description_file:
                zip_description = description_file.read()
                lines = zip_description.strip().decode("utf-8").split("\n")
                # checking description file is present
                code_link, paper_link, name = [""] * 3
                for line in lines:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()

                    if key == "name":
                        name = value
                    elif key == "paper_link":
                        paper_link = value
                    elif key == "code_link":
                        code_link = value
                print(
                    f"name:{name}, paper link: {paper_link}, code link: {code_link}",
                    end="",
                    flush=True,
                )
        else:
            print("description.txt is not present", end="", flush=True)
        print(CHECKMARK)

        if task == "segmentation":
            print("  4. Checking all txt and png files present... ", end="", flush=True)
            png_files = set()
            txt_files = set()

            for file_info in zip_ref.infolist():
                file_path = Path(file_info.filename)
                if file_info.is_dir():
                    continue
                elif file_path.suffix == ".txt":
                    txt_files.add(str(file_path))
                elif file_path.suffix == ".png":
                    png_files.add(str(file_path))

            txt_files = txt_files - {"description.txt"}

            with open(expected_txt_files, "r") as f:
                expected_txt_files = set(f.read().strip().split("\n"))
            # Verify that txt files contain references to png files
            missing_txt_files = expected_txt_files - txt_files
            if len(missing_txt_files) > 0:
                raise ValidationException(f"Txt files missing: {missing_txt_files}")
            unexpected_txt_files = txt_files - expected_txt_files
            if len(unexpected_txt_files) > 0:
                print(f"Unexpected files: {unexpected_txt_files}")

            png_pattern = r".*_instanceIds_\d+_1.png$"
            missing_png_files = png_files.copy()
            for txt_file in txt_files:
                with zip_ref.open(txt_file) as file:
                    lines = file.read().decode("utf-8").strip().split("\n")
                    if (len(lines) == 1) and (len(lines[0]) == 1):
                        raise ValidationException(
                            f"At least one png expected for each label image. File: {txt_file} is incorrect."
                        )
                    png_references = set()
                    for line in lines:
                        folder = Path(txt_file).parent
                        filename = line.split(" ")[0]
                        png_references.add(str(folder / filename))
                        if not re.match(png_pattern, filename):
                            raise ValidationException(
                                f"Png file in txt has name: {filename}, expected ending: *_instanceIds_N_1.png"
                            )

                    missing_png_files -= png_references
            if len(missing_png_files) > 0:
                raise ValidationException(
                    f"Some of the png files were not referenced in txt files (printing fist 100): {list(missing_png_files)[:100]}..."
                )

            print(CHECKMARK)

        if task == "detection":
            print(
                "  4. Checking all png files in the submission are present... ",
                end="",
                flush=True,
            )
            labels = dict()
            for label_file in list(expected_files):
                label_file = Path(label_file)
                with zip_ref.open(str(label_file), "r") as label_f:
                    labels[label_file.name] = json.load(label_f)

            images = list()
            for dataset_name in labels.keys():
                images.extend([file["image_id"] for file in labels[dataset_name]])
            images = set(images)

            with open(expected_txt_files, "r") as f:
                expected_txt_files = list(f.read().strip().split("\n"))
            expected_image_files = list()
            for expected_file in expected_txt_files:
                expected_file = Path(expected_file)
                expected_image_files.append(
                    str(
                        expected_file.name.replace(
                            "_gtCoarse_instanceIds_pred.txt", "_leftImg8bit.png"
                        ).replace("_instanceIds_pred.txt", ".png")
                    )
                )

            expected_image_files = set(expected_image_files)
            if images != expected_image_files:
                raise ValidationException(
                    f"Expected files mismatch: {set(images) ^ set(expected_image_files)}"
                )
            print(CHECKMARK)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Validate a submission zip file needed to evaluate on CodaLab competitions.\n\n
        The verification tool checks:\n
        1. correct folder structure,\n
        2. existence of label files,\n
        3. count of instances for each scan.
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "zipfile",
        type=str,
        help="zip file that should be validated.",
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["segmentation", "detection"],
        default="segmentation",
        help="task for which the zip file should be validated.",
    )

    parser.add_argument(
        "--expected_files",
        type=str,
        default="assets/expected_files.txt",
        help="A list of expected txt files for a prediction.",
    )

    args, _ = parser.parse_known_args()

    print(f'Validating zip archive "{args.zipfile}".\n')
    print(f" ============ {args.task:^10} ============ ")
    verify_submitted_files(Path(args.zipfile), Path(args.expected_files), args.task)
    print("\n\u001b[1;32mEverything ready for submission!\u001b[0m  \U0001f389")
