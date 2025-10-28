安装环境
(cpu环境)
tqdm

conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1

conda install -y numpy pandas matplotlib scikit-learn

建一个train文件夹

把kaggle的train数据放到train里

python data_process_norm.py --root train --outdir data_norm --features x y s a dir o absolute_yardline_number ball_land_x ball_land_y --lookback 0 --val_ratio 0.1 --test_ratio 0.1 --normalize

python train_rnn_opt.py --data_dir data_norm --out_dir runs_gru_opt --epochs 40 --batch_size 512 --hidden 192 --layers 2 --dropout 0.2 --lr 2e-3 --gamma 0.97
