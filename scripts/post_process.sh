python post_process/src/sort_reid_tracker.py --manifest_file manifest.json --scene all

python post_process/src/label_smoother.py --manifest_file manifest.json --scene all

python post_process/src/track_interpolation.py --manifest_file manifest.json --scene all

python post_process/src/tools/concatenate_results.py --manifest_file manifest.json --original_file predict_18.txt