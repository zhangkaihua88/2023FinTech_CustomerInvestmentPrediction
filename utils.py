# source: https://github.com/rickyxume/TianChi_RecSys_AntiSpam
import numpy as np
from sklearn.metrics import f1_score, fbeta_score


def find_best_threshold(y_true, y_pred, l=0.1, r=0.6, p=0.01, sample_weight=None):
    thresholds = np.arange(l, r, p)
    print(f"以精度为{p}在[{thresholds[0]},{thresholds[-1]}]范围内搜索F1最佳阈值", end=">>")
    fscore = np.zeros(shape=(len(thresholds)))
    for index, elem in enumerate(thresholds):
        thr2sub = np.vectorize(lambda x: 1 if x > elem else 0)
        y_preds = thr2sub(y_pred)
        # fscore[index] = f1_score(y_true, y_preds)
        fscore[index] = fbeta_score(y_true, y_preds, beta=2, sample_weight=sample_weight)
    index = np.argmax(fscore)
    thresholdOpt = thresholds[index]
    fscoreOpt = round(fscore[index], ndigits=8)
    print(f'最佳阈值:={thresholdOpt}->F1={fscoreOpt}')
    return thresholdOpt, fscoreOpt


def get_optimal_Fscore(true, pred, sample_weight=None):
    # 由粗到细查找会比较快
    p = 0.1
    thr, best_fscore = find_best_threshold(true, pred, 0.1, 0.9, p, sample_weight)
    p /= 5
    thr, best_fscore = find_best_threshold(true, pred, round(thr-0.1, 4), round(thr+0.1, 4), p, sample_weight)
    p /= 5
    thr_optimal, best_fscore = find_best_threshold(true, pred, round(thr - 0.05, 4), round(thr + 0.05, 4), p, sample_weight)
    return thr_optimal, best_fscore
