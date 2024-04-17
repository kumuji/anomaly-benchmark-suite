from pathlib import Path
import shutil
import os


def prepare_submitted_files(submission_path: Path, temp_folder: Path) -> None:
    input_folder = temp_folder
    input_folder.mkdir(exist_ok=True)


    files_list = sorted(list(submission_path.glob("*.txt")))
    for submission_file in files_list:
        label_class = 26 # car, for an easier compute

        scene_name = "_".join((submission_file.name).split("_")[:-2])
        predictions = list(submission_file.parent.glob(f"{scene_name}*.png"))
        for prediction in predictions:
            # change the class to 26
            new_filename = (
                "_".join(prediction.name.split("_")[:-1]) + f"_{label_class}.png"
            )
            new_filepath = str(input_folder / new_filename)
            shutil.copy(str(prediction), new_filepath)
            os.chmod(new_filepath, 0o755)

        prediction_list = []
        with open(submission_file, "r") as f:
            lines = f.read().strip().split("\n")
            for line in lines:
                prediction_list.append(line.split(" "))

        if (len(prediction_list) == 1) and (len(prediction_list[0]) == 1):
            raise FileNotFoundError(
                f"""No prediction found in {submission_file.name} file.
            Please provide at least one predicted instance.
            """
            )
            # no predictions for a file

        # rewrite the file
        new_submission_file = submission_file.name
        with open(input_folder / new_submission_file, "w") as f:
            for prediction in prediction_list:
                new_filename = (
                    "_".join(prediction[0].split("_")[:-1]) + f"_{label_class}.png"
                )
                f.write(f"{new_filename} {label_class} {prediction[-1]}\n")
