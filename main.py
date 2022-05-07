import os

from model import Sherbet, SherbetFeature, medical_codes_loss
from utils import evalCode, evalHF

import random
import _pickle as pickle
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np

#########################utils###################################
class DataGenerator:
    def __init__(self, inputs, shuffle=True, batch_size=32):
        assert len(inputs) > 0
        self.inputs = inputs
        self.idx = np.arange(len(inputs[0]))
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

    def data_length(self):
        return len(self.idx)

    def __len__(self):
        n = self.data_length()
        len_ = n // self.batch_size
        return len_ if n % self.batch_size == 0 else len_ + 1

    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        index = self.idx[start:end]
        data = []
        for x in self.inputs:
            data.append(x[index])
        return data

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size



#######################hyper parameters##########################
task_m_h = 'h'  #'m'
use_embedding_init = True
pretrain = False
train_test = True
use_hierarc_decoder = True
pretrain_path = './saved/hyperbolic/mimic3/sherbet_a/sherbet_pretrain' 

hyper_params_conf = {
    'pretrain_epoch': 50,  #1000 #parameter set
    'pretrain_batch_size': 128,
    'epoch': 200,
    'batch_size': 32,
    'code_embedding_size': 128,
    'hiddens': [64],
    'attention_size_code': 64,
    'attention_size_visit': 32,
    'patient_size': 64,
    'patient_activation': tf.keras.layers.LeakyReLU(),
    'gnn_dropout_rate': 0.2,# 0.8,
    'decoder_dropout_rate': 0.17 #0.02 #
}
#################################################################


######################dataloader#################################
seed = 6669
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

dataset_path = os.path.join('data', 'mimic3')
#######
encoded_path = os.path.join(dataset_path, 'encoded')
standard_path = os.path.join(dataset_path, 'standard')
code_maps = pickle.load(open(os.path.join(encoded_path, 'code_maps.pkl'), 'rb'))  ##
codes_dataset = pickle.load(open(os.path.join(standard_path, 'codes_dataset.pkl'), 'rb'))
hf_dataset = pickle.load(open(os.path.join(standard_path, 'heart_failure.pkl'), 'rb'))
auxiliary = pickle.load(open(os.path.join(standard_path, 'auxiliary.pkl'), 'rb'))
pretrain_codes_data = pickle.load(open(os.path.join(standard_path, 'pretrain_codes_dataset.pkl'), 'rb'))
######

code_map = code_maps['code_map']
code_map_pretrain = code_maps['code_map_pretrain']

train_codes_data = codes_dataset['train_codes_data']
valid_codes_data = codes_dataset['valid_codes_data']
test_codes_data = codes_dataset['test_codes_data']

train_hf_y = hf_dataset['train_hf_y']
valid_hf_y = hf_dataset['valid_hf_y']
test_hf_y = hf_dataset['test_hf_y']

code_levels = auxiliary['code_levels']
code_levels_pretrain = auxiliary['code_levels_pretrain']
subclass_maps = auxiliary['subclass_maps']
subclass_maps_pretrain = auxiliary['subclass_maps_pretrain']
code_code_adj = auxiliary['code_code_adj']

(pretrain_codes_x, pretrain_codes_y, pretrain_y_h, pretrain_visit_lens) = pretrain_codes_data
(train_codes_x, train_codes_y, train_y_h, train_visit_lens) = train_codes_data
(valid_codes_x, valid_codes_y, valid_y_h, valid_visit_lens) = valid_codes_data
(test_codes_x, test_codes_y, test_y_h, test_visit_lens) = test_codes_data
#################################################################


######################train######################################
feature_model_conf = {
    'code_num': len(code_map_pretrain),
    'code_embedding_init': None,
    'adj': code_code_adj,
    'max_visit_num': train_codes_x.shape[1]
}

if use_embedding_init:
    if pretrain or (not train_test):
        embedding_init = pickle.load(open('./saved/hyperbolic/mimic3_leaf_embeddings', 'rb'))
        feature_model_conf['code_embedding_init'] = embedding_init
sherbet_feature = SherbetFeature(feature_model_conf, hyper_params_conf)
# print("feature",sherbet_feature)

###have pretrain######
pretrain_model_conf = {
    'use_hierarchical_decoder': use_hierarc_decoder,
    'subclass_dims': np.max(code_levels_pretrain, axis=0) if use_hierarc_decoder else None,
    'subclass_maps': subclass_maps_pretrain if use_hierarc_decoder else None,
    'output_dim': len(code_map_pretrain),
    'activation': None
}

if pretrain:
    pretrain_x = {
        'visit_codes': pretrain_codes_x,
        'visit_lens': pretrain_visit_lens
    }
    if use_hierarc_decoder:
        pretrain_x['y_trues'] = pretrain_y_h
        pretrain_y = None
    else:
        pretrain_y = pretrain_codes_y.astype(np.float32)

    init_lr = 1e-2
    # split_val = [(20, 1e-3), (150, 1e-4), (500, 1e-5)]
    split_val = [(10, 1e-3)]   #parameter set
    lr_schedule_fn = lr_decay(total_epoch=hyper_params_conf['pretrain_epoch'], init_lr=init_lr, split_val=split_val)
    lr_scheduler = LearningRateScheduler(lr_schedule_fn)

    if use_hierarc_decoder:
        loss_fn = None 
    else:
        loss_fn = medical_codes_loss

    sherbet_pretrain = Sherbet(sherbet_feature, pretrain_model_conf, hyper_params_conf)
    sherbet_pretrain.compile(optimizer='rmsprop', loss=loss_fn)
    sherbet_pretrain.fit(x=pretrain_x, y=pretrain_y, batch_size=hyper_params_conf['pretrain_batch_size'], 
                        epochs=hyper_params_conf['pretrain_epoch'], callbacks=[lr_scheduler])
    sherbet_pretrain.save_weights(pretrain_path)

#########train#########
else:
    if train_test:
        sherbet_pretrain = Sherbet(sherbet_feature, pretrain_model_conf, hyper_params_conf)
        sherbet_pretrain.load_weights(pretrain_path)

    x_data = {
        'visit_codes': train_codes_x,
        'visit_lens': train_visit_lens
    }
    valid_x = {
        'visit_codes': valid_codes_x,
        'visit_lens': valid_visit_lens
    }
    if task_m_h =='m':
        y_data = train_codes_y.astype(np.float32)
        valid_y = valid_codes_y.astype(np.float32)
        test_y = test_codes_y.astype(np.float32)
        print("train dataser:", len(valid_y),len(y_data))
    else:
        y_data =  train_hf_y.astype(np.float32)
        valid_y = valid_hf_y.astype(np.float32)
        test_y = test_hf_y.astype(np.float32)
        print("train dataser:", len(valid_y),len(y_data))

    # mimic3 h a, b, c
    init_lr = 1e-2
    split_val = [(25, 1e-3), (40, 1e-4), (45, 1e-5)]
    lr_schedule_fn = lr_decay(total_epoch=hyper_params_conf['epoch'], init_lr=init_lr, split_val=split_val)
    lr_scheduler = LearningRateScheduler(lr_schedule_fn)

    test_codes_gen = DataGenerator([test_codes_x, test_visit_lens], shuffle=False, batch_size=128)
    
    if task_m_h =='m':
        loss_fn =  medical_codes_loss
        test_callback = evalCode(test_codes_gen, test_y)
    else:
        loss_fn = 'binary_crossentropy'
        test_callback = evalHF(test_codes_gen, test_y)
        
    model_conf = {
        'use_hierarchical_decoder': False,
        'output_dim': len(code_map) if task_m_h=='m' else 1,
        'activation': None if task_m_h=='m' else 'sigmoid'
    }

    sherbet = Sherbet(sherbet_feature, model_conf, hyper_params_conf)
    sherbet.compile(optimizer='rmsprop', loss=loss_fn)
    history = sherbet.fit(x=x_data, y=y_data, batch_size=hyper_params_conf['batch_size'], 
                        epochs=hyper_params_conf['epoch'], callbacks=[lr_scheduler, test_callback])
    sherbet.summary()
