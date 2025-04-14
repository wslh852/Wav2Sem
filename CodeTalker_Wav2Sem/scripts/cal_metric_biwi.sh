export PYTHONPATH=./
python main/cal_metric.py \
--pred_path "RUN/BIWI/CodeTalker_s2/result_metric1/npy" \
--gt_path "BIWI/vertices_npy" \
--region_path "BIWI/regions" \
--templates_path "BIWI/templates.pkl"