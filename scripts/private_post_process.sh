python post_process/src/sort_reid_tracker.py --manifest_file manifest.json --scene all

python post_process/src/label_smoother.py --manifest_file manifest.json --scene all

python post_process/src/tools/concatenate2.py --manifest_file manifest.json

python post_process/src/tools/conf_filter_stage2.py --input_file predict_concat.txt

# python post_process/src/track_interpolation.py --manifest_file manifest.json --scene all

# python post_process/src/tools/concatenate_results.py --manifest_file manifest.json --original_file predict_18.txt

# python post_process/src/tools/concatenate_results.py --manifest_file manifest.json \
#                                                     --original_file predict_18.txt \
#                                                     --scenes src_1 \
#                                                     --segments segment0 segment1 segment2 segment3

# python post_process/src/tools/intra_class_nms_stage2.py --input_file predict.txt \
#                                            --output_file predict.txt \
#                                            --iou_thresh 0.7