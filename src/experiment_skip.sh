python3 seq_clf_tf.py --model_dir compare_baseline --steps 10000 --l2 0.01 --cell_type baseline
python3 seq_clf_tf.py --model_dir compare_skip --steps 10000 --l2 0.01 --cell_type skip
python3 seq_clf_tf.py --model_dir compare_rrn --steps 10000 --l2 0.01 --cell_type rrn
python3 seq_clf_tf.py --model_dir compare_multiscale --steps 10000 --l2 0.01 --cell_type multiscale

python3 seq_clf_tf.py --model_dir high_dim_reg --steps 20000 --l2 0.1 --hidden 512 --dropout 0.5

python3 seq_clf_tf.py --dataset AGnews --model_dir AGnews_baseline --steps 10000 --l2 0.01 --cell_type baseline
python3 seq_clf_tf.py --dataset AGnews --model_dir AGnews_skip --steps 10000 --l2 0.01 --cell_type skip
python3 seq_clf_tf.py --dataset AGnews --model_dir AGnews_rrn --steps 10000 --l2 0.01 --cell_type rrn
python3 seq_clf_tf.py --dataset AGnews --model_dir AGnews_multiscale --steps 10000 --l2 0.01 --cell_type multiscale

python3 seq_clf_tf.py --dataset IMDB --model_dir IMDB_baseline --steps 10000 --l2 0.01 --cell_type baseline
python3 seq_clf_tf.py --dataset IMDB --model_dir IMDB_skip --steps 10000 --l2 0.01 --cell_type skip
python3 seq_clf_tf.py --dataset IMDB --model_dir IMDB_rrn --steps 10000 --l2 0.01 --cell_type rrn
python3 seq_clf_tf.py --dataset IMDB --model_dir IMDB_multiscale --steps 10000 --l2 0.01 --cell_type multiscale
