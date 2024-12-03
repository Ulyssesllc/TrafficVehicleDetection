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
        0: 0.4,
        1: 0.08,
        2: 0.08,
        3: 0.08
    }
}

def parse_opt():
    parser = argparse.ArgumentParser(description="Class-specific confidence threshold.")
    parser.add_argument('--input_file', type=str, help="Input predict.txt file.", required=True)
    parser.add_argument('--output_file', type=str, help="Input predict.txt file.", default="predict.txt")
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

        if conf <= 0.001: conf = 0.5
        new_line = f"{components[0]} {components[1]} {components[2]} {components[3]} {components[4]} {components[5]} {conf:6f}\n"

        # dst_file.writelines(new_line)

        if components[0].startswith("cam_11"):
            if conf > filter_config["cam_11"][cls]:
                dst_file.writelines(new_line)
        elif components[0].startswith("cam_13"):
            if conf > filter_config["cam_13"][cls]:
                dst_file.writelines(new_line)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)