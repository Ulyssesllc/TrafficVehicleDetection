import os
import json
import argparse


def parse_opt():
    parser = argparse.ArgumentParser(description="Perform Multi-Object Tracking on the video segments.")
    parser.add_argument('--manifest_file', type=str, help="Input manifest.json file.", required=True)
    parser.add_argument('--original_file', type=str, help="Input predict.txt file.", required=True)
    opt = parser.parse_args()
    return opt

def main(opt):
    manifest_file = opt.manifest_file
    original_pred = opt.original_file

    preprocess_pred = open(original_pred, mode="r")
    manifest = json.load(open("manifest.json", mode="r"))
    final_output = open("predict.txt", mode="w+")

    for line in preprocess_pred:
        final_output.write(line)

    for scene in manifest.keys():
        for segment in manifest[scene].keys():
            interpolation_path = os.path.join("post_process/data_reorganized", scene, segment, "box_addition", "yolo_interpolate.txt")

            with open(interpolation_path, mode="r") as addition:
                for line in addition:
                    final_output.write(line)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)