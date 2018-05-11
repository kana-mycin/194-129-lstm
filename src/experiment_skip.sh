python3 seq_clf_tf.py --model_dir compare_baseline --steps 10000 --l2 0.01 --cell_type baseline
python3 seq_clf_tf.py --model_dir compare_skip --steps 10000 --l2 0.01 --cell_type skip
python3 seq_clf_tf.py --model_dir compare_rrn --steps 10000 --l2 0.01 --cell_type rrn
python3 seq_clf_tf.py --model_dir compare_multiscale --steps 10000 --l2 0.01 --cell_type multiscale
