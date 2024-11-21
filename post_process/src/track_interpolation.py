import os
import json
import argparse

def interpolate_tracks(scene, segment, max_gap_threshold, img_width=1280, img_height=720):
    """
    Perform track interpolation for missing frames in a MOT-formatted file based on track IDs.
    Tracks with gaps larger than the threshold will not be interpolated.

    Args:
        input_mot_file (str): Path to the input MOT file.
        output_interpolated_file (str): Path to save the interpolated track file.
        max_gap_threshold (int): Maximum allowed gap for interpolation.

    Output:
        Saves the interpolated track information to the specified file and prints 
        the number of interpolated bounding boxes generated.
    """
    import collections

    input_mot_file = f"post_process/data_reorganized/{scene}/{segment}/smoothing/mot_smooth.txt"
    os.makedirs(f"post_process/data_reorganized/{scene}/{segment}/box_addition", exist_ok=True)
    output_mot_interpolation_file = f"post_process/data_reorganized/{scene}/{segment}/box_addition/mot_interpolate.txt"
    output_yolo_interpolation_file = f"post_process/data_reorganized/{scene}/{segment}/box_addition/yolo_interpolate.txt"

    # Read input MOT file and group by track ID
    track_data = collections.defaultdict(list)
    unique_filenames = {}  # Map filename to a unique frame ID
    frame_counter = 1  # Start frame enumeration from 1

    with open(input_mot_file, 'r') as mot_file:
        for line in mot_file:
            filename, track_id, x1, y1, x2, y2, cls, conf = line.strip().split(',')
            track_id = int(track_id)

            # Assign a unique frame ID to each filename
            if filename not in unique_filenames:
                unique_filenames[filename] = frame_counter
                frame_counter += 1

            frame_id = unique_filenames[filename]
            track_data[track_id].append({
                "frame_id": frame_id,
                "filename": filename,
                "x1": int(float(x1)),
                "y1": int(float(y1)),
                "x2": int(float(x2)),
                "y2": int(float(y2)),
                "cls": int(cls),
                "conf": float(conf)
            })

    interpolated_tracks = []
    interpolated_count = 0

    # Interpolate missing frames
    for track_id, frames in track_data.items():
        frames = sorted(frames, key=lambda x: x["frame_id"])  # Sort frames by frame ID
        for i in range(len(frames) - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]

            # Calculate gaps
            frame_gap = next_frame["frame_id"] - current_frame["frame_id"]
            if 1 < frame_gap <= max_gap_threshold:
                # Linearly interpolate bounding boxes
                for gap_frame in range(1, frame_gap):
                    interp_frame_id = current_frame["frame_id"] + gap_frame
                    alpha = gap_frame / frame_gap

                    # Interpolate bounding box coordinates
                    interp_x1 = round(current_frame["x1"] + alpha * (next_frame["x1"] - current_frame["x1"]))
                    interp_y1 = round(current_frame["y1"] + alpha * (next_frame["y1"] - current_frame["y1"]))
                    interp_x2 = round(current_frame["x2"] + alpha * (next_frame["x2"] - current_frame["x2"]))
                    interp_y2 = round(current_frame["y2"] + alpha * (next_frame["y2"] - current_frame["y2"]))

                    # Retrieve the filename corresponding to the interpolated frame ID
                    interp_filename = [key for key, value in unique_filenames.items() if value == interp_frame_id][0]

                    # Add interpolated data
                    interpolated_tracks.append({
                        "filename": interp_filename,
                        "track_id": track_id,
                        "x1": interp_x1,
                        "y1": interp_y1,
                        "x2": interp_x2,
                        "y2": interp_y2,
                        "cls": current_frame["cls"],
                        "conf": current_frame["conf"]
                    })
                    interpolated_count += 1

    # Write interpolated tracks to the output file
    with open(output_mot_interpolation_file, 'w') as out_file:
        for track in interpolated_tracks:
            out_file.write(
                f"{track['filename']},{track['track_id']},{track['x1']},"
                f"{track['y1']},{track['x2']},{track['y2']},"
                f"{track['cls']},{0.07}\n"
            )

    with open(output_yolo_interpolation_file, 'w') as out_file:
        for track in interpolated_tracks:
            xn = ((track['x2'] + track['x1']) / 2) / img_width
            yn = ((track['y2'] + track['y1']) / 2) / img_height
            wn = (track['x2'] - track['x1']) / img_width
            hn = (track['y2'] - track['y1']) / img_height

            out_file.write(f"{track['filename']} {track['cls']} {xn} {yn} {wn} {hn} {0.07}\n")

    # Print the number of interpolated bounding boxes
    print(f"Number of additional bounding boxes generated for {scene} - {segment}: {interpolated_count}")

def parse_opt():
    parser = argparse.ArgumentParser(description="Perform Multi-Object Tracking on the video segments.")
    parser.add_argument('--manifest_file', type=str, help="Input manifest.json file.", required=True)
    parser.add_argument('--scene_to_process', type=str, default="all", help="Specify which scene to be tracked. Set to all to track every scene.")
    parser.add_argument('--segment_to_process', type=str, default="none", help="Specify which segment to be tracked.")
    parser.add_argument('--visualization', type=str, default="yes", help="Output the visualization or not.")
    opt = parser.parse_args()
    return opt

def main(opt):
    manifest_file = opt.manifest_file
    scene_to_process = opt.scene_to_process
    segment_to_process = opt.segment_to_process

    manifest = json.load(open(manifest_file))

    if scene_to_process == "all":
        print("Going for all!")
        for scene in manifest.keys():
            for segment in manifest[scene].keys():
                interpolate_tracks(scene, segment, 5)
    else:
        interpolate_tracks(scene_to_process, segment_to_process, 5)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
