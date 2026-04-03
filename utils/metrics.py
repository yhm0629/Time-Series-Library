import numpy as np
from sklearn.metrics import precision_recall_curve, auc, f1_score

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

def calculate_metrics(scores, labels):
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    auprc = auc(recalls, precisions)

    f1_scores = 2 * recalls * precisions / (recalls + precisions + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[min(best_f1_idx, len(thresholds)-1)]

    preds = (scores > best_threshold).astype(int)
    f1_e = get_event_f1(preds, labels)

    return best_f1, f1_e, auprc

def get_event_f1(preds, labels):
    def find_events(target):
        events = []
        state = 0
        start = 0
        for i, v in enumerate(target):
            if v == 1 and state == 0:
                state, start = 1, i
            elif v == 0 and state == 1:
                state = 0
                events.append((start, i))
        if state == 1: events.append((start, len(target)))
        return events

    gt_events = find_events(labels)
    pred_events = find_events(preds)

    hits_recall = sum(1 for s, e in gt_events if np.any(preds[s:e] == 1))
    recall_e = hits_recall / len(gt_events) if gt_events else 0

    hits_precision = sum(1 for s, e in pred_events if np.any(labels[s:e] == 1))
    precision_e = hits_precision / len(pred_events) if pred_events else 0

    return 2 * precision_e * recall_e / (precision_e + recall_e + 1e-8)
