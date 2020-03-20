from keras import backend as K
import os
import collections
import re
import string

def accuracy(y_true, y_pred):

    def calculate_accuracy(true_and_pred):
        y_true, y_pred_start, y_pred_end = true_and_pred

        start_probability = y_pred_start[K.cast(y_true[0], dtype='int32')]
        end_probability = y_pred_end[K.cast(y_true[1], dtype='int32')]
        return (start_probability + end_probability) / 2.0

    y_true = K.squeeze(y_true, axis=1)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]
    accuracy = K.map_fn(calculate_accuracy, (y_true, y_pred_start, y_pred_end), dtype='float32')
    return K.mean(accuracy, axis=0)

def compute_accuracy(y_true, y_pred_start, y_pred_end):

    def calculate_accuracy(y_true, y_pred_start, y_pred_end):

        start_probability = y_pred_start[K.cast(y_true[0], dtype='int32')]
        end_probability = y_pred_end[K.cast(y_true[1], dtype='int32')]
        return (start_probability + end_probability) / 2.0

    y_true = K.squeeze(y_true, axis=1)
    
    accuracy = K.map_fn(calculate_accuracy(y_true, y_pred_start, y_pred_end), dtype='float32')
    return K.mean(accuracy, axis=0)


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_EM(y_true, y_pred):

    actual_tokens = normalize_answer(y_true)
    pred_tokens = normalize_answer(y_pred['answer'])

    em = int(actual_tokens == pred_tokens)
    
    return em



def compute_f1(y_true, y_pred):

    actual_tokens = normalize_answer(y_true).split()
    pred_tokens = normalize_answer(y_pred['answer']).split()

    common = collections.Counter(actual_tokens) & collections.Counter(pred_tokens)

    num_same = sum(common.values())
    if len(actual_tokens) == 0 or len(pred_tokens) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(actual_tokens == pred_tokens)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(actual_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
