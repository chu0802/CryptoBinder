import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelmax, argrelmin
from scipy.ndimage import gaussian_filter1d as gaussian
from itertools import combinations
import argparse
from pathlib import Path
import time

def data_split(data, interval=9):
    return np.stack([data[i:i+interval] for i in range(len(data)-interval+1)])

def rsv_value(data, interval=9):
    split_data = data_split(data, interval=interval)
    M, m = split_data.max(axis=1), split_data.min(axis=1)
    return (split_data[:, -1] - m) / (M - m)

def kd_value(data, ratio=2):
    M, m = ratio/(ratio+1), 1/(ratio+1)
    ret = [0.5]
    for r in data:
        ret.append(ret[-1]*M + r*m)
    return np.array(ret)[1:]

def get_kdj(data, interval=9, ratio=2):
    rsv = rsv_value(data)
    k = kd_value(rsv)
    d = kd_value(k)
    j1 = 3*d - 2*k
    j2 = 3*k - 2*d
    return np.r_[np.zeros((8, 4)), np.c_[k, d, j1, j2]]

def get_scaled_kdj(data, scale=1, interval=9, ratio=2):
    n_d = [data[list(range(i, len(data), scale))] for i in range(scale)]
    per_kdj = [get_kdj(d) for d in n_d]
    kdjs = np.zeros((len(data), 4))
    for i, k in enumerate(per_kdj):
        kdjs[list(range(i, len(data), scale))] = k
    return kdjs

def less(a):
    return 1 if a[0] < a[1] else (0 if a[0] == a[1] else -1)

def get_labels(data, interval=8):
    return np.array([sum(map(less, combinations(data[i:i+interval+1], 2))) for i in range(len(data)-interval)]) / np.arange(interval+1).sum()

def argument_parsing():
    p = argparse.ArgumentParser()
    p.add_argument('--kd_interval', default=9)
    p.add_argument('--kd_ratio', default=2)
    p.add_argument('--kd_blur_sigma', default=1)
    p.add_argument('--feature_length', default=48)
    p.add_argument('--label_blur_sigma', default=10)
    p.add_argument('--label_interval', default=48)
    p.add_argument('--data_dir', type=Path, default='./data')
    return p.parse_args()
    
def main(args):
    d = yf.download(tickers='BTC-USD', period='2y', interval = "1h")['Close']
    blur_d = gaussian(d, sigma=args.kd_blur_sigma)
    
    kdj_1h = get_kdj(blur_d, interval=args.kd_interval, ratio=args.kd_ratio)[24*(args.kd_interval-1):]
    kdj_4h = get_scaled_kdj(blur_d, scale=4, interval=args.kd_interval, ratio=args.kd_ratio)[24*(args.kd_interval-1):]
    kdj_24h = get_scaled_kdj(blur_d, scale=24, interval=args.kd_interval, ratio=args.kd_ratio)[24*(args.kd_interval-1):]
    kdj_features = data_split(np.c_[kdj_1h, kdj_4h, kdj_24h], interval=args.feature_length)
    truth_d = gaussian(d, sigma=args.label_blur_sigma)[24*(args.kd_interval-1)+args.feature_length-1:]
    y = get_labels(truth_d, interval=args.label_interval)
    x = kdj_features[:-args.label_interval]
    
    args.data_dir.mkdir(exist_ok=True, parents=True)
    output_path = args.data_dir / (time.strftime("%Y_%m_%d_%H") + '.npz')
    with output_path.open('wb') as f:
        np.savez(f, x=x, y=y, info=vars(args), allow_pickle=True)
    
if __name__ == '__main__':
    args = argument_parsing()
    main(args)