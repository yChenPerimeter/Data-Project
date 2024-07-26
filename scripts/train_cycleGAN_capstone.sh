set -ex
python train.py --dataroot ./datasets/capstone --name capstoneDCIS_cyclegan --model cycle_gan --pool_size 50 --no_dropout
