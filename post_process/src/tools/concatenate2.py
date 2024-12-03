import os
import json
import argparse

def parse_opt():
    parser = argparse.ArgumentParser(description="Concatenate smoothing results.")
    parser.add_argument('--manifest_file', type=str, help="Input manifest.json file.", required=True)
    opt = parser.parse_args()
    return opt

    
def main(opt):
    manifest_file = opt.manifest_file

    manifest = json.load(open(manifest_file, mode="r"))
    final_output = open("predict_concat.txt", mode="w+")

    for scene in manifest.keys():
        for segment in manifest[scene].keys():
            print(f"Concatenating {scene} - {segment}")

            smoothing_path = os.path.join("post_process/data_reorganized", scene, segment, "smoothing", "yolo_smooth.txt")

            with open(smoothing_path, mode="r") as addition:
                for line in addition:
                    final_output.write(line)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)