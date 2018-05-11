python3 seq_clf_tf.py --model_dir compare_baseline --steps 5000 --l2 0.01 --cell_type baseline
python3 seq_clf_tf.py --model_dir compare_skip --steps 5000 --l2 0.01 --cell_type skip
python3 seq_clf_tf.py --model_dir compare_multiscale --steps 5000 --l2 0.01 --cell_type multiscale
