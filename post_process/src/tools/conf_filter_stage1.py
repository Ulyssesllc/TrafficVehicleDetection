import os
import argparse

filter_config = {
    "cam_11": {
        0: 0.2,
        1: 0.3,
        2: 0.99,
        3: 0.99
    },
    "cam_13": {
        0: 0.08,
        1: 0.08,
        2: 0.08,
        3: 0.08
    }
}

def parse_opt():
    parser = argparse.ArgumentParser(description="Class-specific confidence threshold.")
    parser.add_argument('--input_file', type=str, help="Input predict.txt file.", required=True)
    parser.add_argument('--output_file', type=str, help="Input predict.txt file.", default="predict_18.txt")
    opt = parser.parse_args()
    return opt

def main(opt):
    input_file = opt.input_file
    output_file = opt.output_file

    src_file = open(input_file, mode="r")
    dst_file = open(output_file, mode="w+")

    for line in src_file:
        components = line.strip().split()
        conf = float(components[-1])
        cls = int(components[1])

        if components[0].startswith("cam_11"):
            if conf > filter_config["cam_11"][cls]:
                dst_file.writelines(line)
        elif components[0].startswith("cam_13"):
            if conf > filter_config["cam_13"][cls]:
                dst_file.writelines(line)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)