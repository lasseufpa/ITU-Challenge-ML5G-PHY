# Usage example: python topK.py --beam_test_label beam_test_label_fake.txt --beam_test_pred bream_test_pred_fake.csv

import tensorflow as tf
import numpy as np
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--beam_test_label',help='Ground truth file', type=str, required=False,\
                    default='./beam_test_label_s009.csv')
parser.add_argument('--beam_test_pred', help='Predictions file', type=str,  required=False,\
                    default='../submission_baseline_example/beam_test_pred.csv')
args = parser.parse_args()

labels = pd.read_csv(args.beam_test_label,header=None)
pred = pd.read_csv(args.beam_test_pred,header=None)

print(labels.shape, pred.shape)

print('Top 1 =', tf.math.in_top_k(labels.values[:,0], pred.values, 1, name=None).numpy().mean())
print('Top 2 =', tf.math.in_top_k(labels.values[:,0], pred.values, 2, name=None).numpy().mean())
print('Top 10 =', tf.math.in_top_k(labels.values[:,0], pred.values, 10, name=None).numpy().mean())
