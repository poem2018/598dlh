import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

def masked_softmax(inputs, mask):
    inputs = inputs - tf.reduce_max(inputs, keepdims=True, axis=-1)
    exp = tf.exp(inputs) * mask
    result = tf.math.divide_no_nan(exp, tf.reduce_sum(exp, keepdims=True, axis=-1))
    return result
    

def f1(y_true_hot, y_pred, metrics='weighted'):
    result = np.zeros_like(y_true_hot)
    for i in range(len(result)):
        true_number = np.sum(y_true_hot[i] == 1)
        result[i][y_pred[i][:true_number]] = 1
    return f1_score(y_true=y_true_hot, y_pred=result, average=metrics)


def top_k_prec_recall(y_true_hot, y_pred, ks):
    a = np.zeros((len(ks), ))
    r = np.zeros((len(ks), ))
    for pred, true_hot in zip(y_pred, y_true_hot):
        true = np.where(true_hot == 1)[0].tolist()
        t = set(true)
        for i, k in enumerate(ks):
            p = set(pred[:k])
            it = p.intersection(t)
            a[i] += len(it) / k
            r[i] += len(it) / min(k, len(t))
    return a / len(y_true_hot), r / len(y_true_hot)

def load_data(dataFile, labelFile, timeFile):
	test_set_x = np.array(pickle.load(open(dataFile, 'rb')))
	test_set_y = np.array(pickle.load(open(labelFile, 'rb')))
	test_set_t = None
	if len(timeFile) > 0:
		test_set_t = np.array(pickle.load(open(timeFile, 'rb')))

	def len_argsort(seq):
		return sorted(range(len(seq)), key=lambda x: len(seq[x]))

	sorted_index = len_argsort(test_set_x)
	test_set_x = [test_set_x[i] for i in sorted_index]
	test_set_y = [test_set_y[i] for i in sorted_index]
	if len(timeFile) > 0:
		test_set_t = [test_set_t[i] for i in sorted_index]
	
	test_set = (test_set_x, test_set_y, test_set_t)

	return test_set


# def calculate_auc(test_model, dataset, options):
# 	inputDimSize = options['inputDimSize']
# 	numClass = options['numClass']
# 	batchSize = options['batchSize']
# 	useTime = options['useTime']
# 	predictTime = options['predictTime']
	
# 	n_batches = int(np.ceil(float(len(dataset[0])) / float(batchSize)))
# 	aucSum = 0.0
# 	dataCount = 0.0
# 	for index in xrange(n_batches):
# 		batchX = dataset[0][index*batchSize:(index+1)*batchSize]
# 		batchY = dataset[1][index*batchSize:(index+1)*batchSize]
# 		if predictTime:
# 			batchT = dataset[2][index*batchSize:(index+1)*batchSize]
# 			x, y, t, t_label, mask, lengths = padMatrixWithTimePrediction(batchX, batchY, batchT, options)
# 			auc = test_model(x, y, t, t_label, mask, lengths)
# 		elif useTime:
# 			batchT = dataset[2][index*batchSize:(index+1)*batchSize]
# 			x, y, t, mask, lengths = padMatrixWithTime(batchX, batchY, batchT, options)
# 			auc = test_model(x, y, t, mask, lengths)
# 		else:
# 			x, y, mask, lengths = padMatrixWithoutTime(batchX, batchY, options)
# 			auc = test_model(x, y, mask, lengths)
# 		aucSum += auc * len(batchX)
# 		dataCount += float(len(batchX))
# 	return aucSum / dataCount
 


def lr_decay(total_epoch, init_lr, split_val):
    lr_map = [init_lr] * total_epoch
    if len(split_val) > 0:
        assert split_val[0][0] > 1   #parameter set
        # print("<>",split_val[-1][0])
        assert split_val[-1][0] <= total_epoch
        current_split_index = 0
        current_lr = init_lr
        next_epoch, next_lr = split_val[current_split_index]
        for i in range(total_epoch):
            if i < next_epoch - 1:
                lr_map[i] = current_lr
            else:
                current_lr = next_lr
                lr_map[i] = current_lr
                current_split_index += 1
                if current_split_index >= len(split_val):
                    next_epoch = total_epoch + 1
                else:
                    next_epoch, next_lr = split_val[current_split_index]

    def lr_schedule_fn(epoch, lr):
        return lr_map[epoch]

    return lr_schedule_fn


class EvaluateCallBack(Callback):
    def __init__(self, data_gen, y):
        super().__init__()
        self.data_gen = data_gen
        self.y = y

    def on_epoch_end(self, epoch, logs=None):
        raise NotImplementedError


class evalCode(EvaluateCallBack):
    def on_epoch_end(self, epoch, logs=None):
        step_size = len(self.data_gen)
        preds = []
        for i in range(step_size):
            batch_codes_x, batch_visit_lens = self.data_gen[i]
            output = self.model(inputs={
                'visit_codes': batch_codes_x,
                'visit_lens': batch_visit_lens
            }, training=False)
            # logits = tf.math.sigmoid(output)
            logits = output
            pred = tf.argsort(logits, axis=-1, direction='DESCENDING')
            preds.append(pred.numpy())
        preds = np.vstack(preds)
        f1_score = f1(self.y, preds)
        prec, recall = top_k_prec_recall(self.y, preds, ks=[10, 20, 30, 40])
        print('\t', 'f1_score:', f1_score, '\t', 'top_k_recall:', recall)


class evalHF(EvaluateCallBack):
    def on_epoch_end(self, epoch, logs=None):
        step_size = len(self.data_gen)
        print(step_size)
        preds, outputs = [], []
        for i in range(step_size):
            batch_codes_x, batch_visit_lens = self.data_gen[i]
            output = self.model(inputs={
                'visit_codes': batch_codes_x,
                'visit_lens': batch_visit_lens
            }, training=False)
            
            outputs.append(tf.squeeze(output).numpy())
            pred = tf.squeeze(tf.cast(output > 0.5, tf.int32))
            preds.append(pred.numpy())
        # print("!!!!",outputs)
        outputs = np.concatenate(outputs)
        preds = np.concatenate(preds)
        auc = roc_auc_score(self.y, outputs)
        f1_score_ = f1_score(self.y, preds)
        print('\t', 'auc:', auc, '\t', 'f1_score:', f1_score_)




if __name__ == '__main__':
    print("111")
