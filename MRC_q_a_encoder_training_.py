#! -*- coding:utf-8 -*-
import json

from keras.backend import tensorflow_backend
from util import get_model_by_name, read_data_for_qa, compute_input_arrays, seq_padding, read_data_for_qa_single_bert, \
    compute_input_arrays_pairsent, check_top_n_acc
import random
import keras
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from sklearn.metrics import classification_report
import numpy as np

pseg = ''
cls_num = 1
output_len = cls_num
Threshold = 1
min_learning_rate = 2e-5

from bert4keras.models import build_transformer_model as build_bert_model
from bert4keras.tokenizers import Tokenizer
import tensorflow as tf
from sklearn.model_selection import GroupKFold
import os
from sklearn.metrics import f1_score


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def triplet_loss(model_anchor, model_positive, model_negative, margin):
    distance1 = tf.sqrt(tf.reduce_sum(tf.pow(model_anchor - model_positive, 2), 1, keepdims=True))
    distance2 = tf.sqrt(tf.reduce_sum(tf.pow(model_anchor - model_negative, 2), 1, keepdims=True))
    return tf.reduce_mean(tf.maximum(distance1 - distance2 + margin, 0)) + 1e-9


def process_json(jsonstr):
    res_list = list()
    try:
        for one in json.loads(jsonstr):
            for onekey in one:
                res_list.append(str(one[onekey]))
    except:
        return jsonstr
    return ' '.join(res_list)


def sparse_categorical_crossentropy(y_true, y_pred):
    """自定义稀疏交叉熵
    这主要是因为keras自带的sparse_categorical_crossentropy不支持求二阶梯度。
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.categorical_crossentropy(y_true, y_pred)


def bert_model(params, model_config):
    orig_bert_model_a = build_bert_model(model_config.config_path, model_config.checkpoint_path,
                                         model=model_config.model_type,
                                         )
    # orig_bert_model_q = build_bert_model(model_config.config_path, model_config.checkpoint_path,
    #                                      model=model_config.model_type,
    #                                      )
    for l in orig_bert_model_a.layers:
       l.trainable = True

    # for l in orig_bert_model_q.layers:
    #    l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    # x3_in = Input(shape=(None,))
    # x4_in = Input(shape=(None,))

    # x5_in = Input(shape=(None,))
    # x6_in = Input(shape=(None,))

    r_out = Input(shape=(None,), dtype='float32')

    x = orig_bert_model_a([x1_in, x2_in])
    # x_p = orig_bert_model_a([x3_in, x4_in])
    # x_n = orig_bert_model_a([x5_in, x6_in])

    first_token = Lambda(lambda x: x[:, 0])(x)
    embedding = first_token
    # first_token = Dropout(params[0], name='dp1')(first_token)
    # first_token_p = Lambda(lambda x: x[:, 0])(x_p)
    # first_token_n = Lambda(lambda x: x[:, 0])(x_n)
    # embedding = first_token_p
    first_token = Dense(32, name="dense_mid",
                      kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                      )(first_token)
    first_out = Dense(1, name="dense_output",
                      kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                      )(first_token)

    p_out = first_out
    model = Model(
        [x1_in, x2_in ],
        (embedding,p_out)
    )

    train_model = Model([x1_in, x2_in, r_out],
                        embedding)

    # final_loss = triplet_loss(x, x_p, x_n, 1.1)
    final_loss = keras.losses.binary_crossentropy(r_out,p_out)

    train_model.add_loss(final_loss)
    train_model.compile(
        optimizer=Adam(min_learning_rate),  # 用足够小的学习率
    )
    train_model.summary()

    return train_model, model


cur_config = get_model_by_name('bert')
tokenizer = Tokenizer(cur_config.spm_path)  # 建立分词器
use_postag = False


def predict_vec(d, nn_model):
    text = d
    if use_postag:
        # s_nv, s_v = mask_sent_by_verb(text)
        x1, x2 = tokenizer.encode(text, text)
    else:
        x1, x2 = tokenizer.encode(text)
    X1 = [x1]
    X2 = [x2]
    X1 = seq_padding(X1)
    X2 = seq_padding(X2)
    R = nn_model.predict([X1, X2])[0]
    return R


def predict_one_sent(d, model):
    text = d
    lab_array = predict_vec(text, model)
    # lab_array = [str(ele) for ele in lab_array]
    return lab_array


def read_valid_dataset(input_categories):
    res = list()
    for idx, one_line in enumerate(input_categories):
        q = one_line[0]
        if len(str(one_line[2])) <= 0 or ',' in str(one_line[2]): continue
        label = str(one_line[3]).split('###')[0]
        sim = str(int(float(str(label))))
        res.append((q, sim))
    return res


def get_sheet_by_name(CAT):
    res = {"HALO-BIGROOM": 0, "差旅系统": 1, "成本管理系统": 3, "HR-π": 2}
    return res[CAT]


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser('传入参数：***.py')
    parser.add_argument('-m', '--model', default='bert')
    parser.add_argument('-e', '--epoch', default='1')
    parser.add_argument('-d', '--dropout', default='0.05')
    parser.add_argument('-f', '--flooding_thres', default='0.0')
    parser.add_argument('-p', '--use_postag', default='False')
    parser.add_argument('-r', '--random', default='0')

    args = parser.parse_args()
    name_ = args.model
    cur_config = get_model_by_name(name_)

    maxlen = 510
    batch_size = 12
    epoch_num = int(args.epoch)
    dropout = float(args.dropout)
    Threshold = float(args.flooding_thres)
    use_postag = bool(args.use_postag)
    min_learning_rate = 2e-5  #

    token_dict = {}
    triple_sents, all_test_anwsers = read_data_for_qa_single_bert("longzhu.xlsx", test_num=10, train_ratio=0.04)
    # _, all_test_anwsers = read_data_for_qa("longzhu_testset.xlsx", test_num=19804-4, train_ratio=1)

    quest_label_dict = dict()
    label_question_dict = dict()
    for one in triple_sents:
        question_ = one[0]
        label_question_dict[one[1]] = question_
        quest_label_dict[question_] = one[1]
    cls_num = len(list(set(list(label_question_dict.keys()))))
    label_indx_dict = {one: idx for idx, one in enumerate(list(set(list(label_question_dict.keys()))))}
    print('class num ', cls_num)
    index_label_dict = {v: k for k, v in label_indx_dict.items()}
    json.dump(label_indx_dict, open('label_index_map.json', 'w', encoding='utf-8'), ensure_ascii=False, )
    json.dump(index_label_dict, open('index_label_map.json', 'w', encoding='utf-8'), ensure_ascii=False, )
    json.dump(label_question_dict, open('label_question_map.json', 'w', encoding='utf-8'), ensure_ascii=False, )
    # all_test_indexes = [ label_indx_dict[str(int(float(one[1])))] for one in all_test_anwsers]
    quest_label_list = [(k, v) for k, v in quest_label_dict.items()]
    df_train = [one[0] for one in quest_label_list]
    output_categories = [float(label_indx_dict[one[1]]) for one in quest_label_list]
    df_val = list()
    valid_output_categories = list()

    # df_val = [one[0] for one in valid_data]
    # valid_output_categories = [label_indx_dict[one[1]] for one in valid_data]

    tr_len = -1
    df_train = df_train[:tr_len]
    output_categories = output_categories[:tr_len]

    tokenizer = Tokenizer(cur_config.spm_path)  # 建立分词器



    def comput_f1_by_label(trues, preds):
        # print(trues.shape, preds.shape)
        for index, (col_trues, col_pred) in enumerate(zip(trues.T, preds.T)):
            # print(col_trues.shape, col_pred.shape)
            col_pred = [1 if one_pred >= 0.5 else 0 for one_pred in col_pred]
            print(index, classification_report(col_trues, col_pred))


    class CustomCallback(tf.keras.callbacks.Callback):
        def __init__(self, valid_data, test_data, batch_size=16, fold=None, encoder=None):
            self.valid_inputs = valid_data
            self.valid_outputs = valid_data[-1]
            self.test_inputs = test_data

            self.batch_size = batch_size
            self.fold = fold
            self.f1 = 0
            self.encoder = encoder

        def on_train_begin(self, logs={}):
            self.valid_predictions = []
            self.test_predictions = []

        def on_epoch_begin(self, epoch, logs={}):
            print('on epoch end???')
            pred_list = list()
            val_list = list()
            all_cand_text = [one[1] for one in self.test_inputs]
            all_valid_text = [one[0] for one in self.test_inputs]
            all_cand_index = [idx for idx, one in enumerate(self.test_inputs)]
            # valid_input_holder = []
            # all_valid_ids = compute_input_arrays_pairsent([one[0] for one in self.test_inputs], maxlen=maxlen, tokenizer=tokenizer)
            # valid_input_holder.extend(all_valid_ids)
            # valid_input_holder.extend(all_valid_ids)
            # encoder_valid_outputvecs = self.encoder.predict(valid_input_holder, batch_size=self.batch_size)
            # self.valid_predictions.append(
            #     encoder_valid_outputvecs[0])
            # all_cand_ids = compute_input_arrays(all_cand_text, tokenizer=tokenizer)
            # valid_cand_holder = []
            # valid_cand_holder.extend(all_cand_ids)
            # valid_cand_holder.extend(all_cand_ids)
            # all_cand_vecs = self.encoder.predict(valid_cand_holder, batch_size=self.batch_size)
            # all_cand_vecs = all_cand_vecs[1]
            for idx, one_query in enumerate(all_valid_text):
                all_valid_text = [( one_query, one_test[1]) for one_test in self.test_inputs]
                valid_input_holder = []
                all_valid_ids = compute_input_arrays_pairsent(all_valid_text, maxlen=maxlen,
                                                              tokenizer=tokenizer)
                valid_input_holder.extend(all_valid_ids)
                # valid_input_holder.extend(all_valid_ids)//////////////////////////
                encoder_valid_outputvecs = self.encoder.predict(valid_input_holder, batch_size=self.batch_size)[1]

                dot_list = encoder_valid_outputvecs.tolist()
                dot_list = [num[0] for num in dot_list]
                # for one_vec in all_cand_vecs.tolist():
                #     dot_list.append(np.linalg.norm(one_vec - one))
                # dot_list = np.linalg.norm(all_cand_vecs -one,-1)
                # print(dot_list)
                rank_list, flag = check_top_n_acc(dot_list, idx, 3, all_cand_text, high2low=True)
                max_idx = np.argmax(dot_list)
                if flag:
                    pred_one = str(int(all_cand_index[idx]))
                else:
                    pred_one = str(int(all_cand_index[max_idx]))
                pred_list.append(pred_one)

                val_out = str(int(all_cand_index[idx]))
                val_list.append(val_out)
                if not flag:
                    print('wrong q:', all_valid_text[idx], " |c: ",
                          all_cand_text[idx])
                    for one_id in rank_list:
                        print(one_id)
                print('=======================================')
            print(classification_report(val_list, pred_list))
            f1 = f1_score(val_list, pred_list, average='weighted')
            print('total f1: ', f1)
            if f1 > self.f1:
                self.encoder.save_weights('it_nlu-{name}.h5'.format(name=epoch))
                self.f1 = f1

        def check_top_n_acc(self, dot_list, correct_id, top_n, cand_texts):
            correct_text = str(cand_texts[correct_id]).strip()
            text_score_pair_dict = {str(one_text).strip():dot_list[idx]  for idx , one_text in enumerate(cand_texts)}
            cand_texts = list(set([str(one_text).strip() for one_text in cand_texts]))
            new_dot_list = [text_score_pair_dict[text] for idx, text in enumerate(cand_texts)]
            correct_id = cand_texts.index(correct_text)
            min_ids = np.argsort(new_dot_list)[:top_n]
            if correct_id in min_ids:
                return min_ids, True
            else:
                return min_ids, False


    def slice(x, index):
        return x[:, :, index]


    def train_and_predict(model, train_data, valid_data, test_data,
                          epochs, batch_size, fold):
        custom_callback = CustomCallback(
            valid_data=valid_data,
            test_data=test_data,
            batch_size=batch_size,
            fold=fold, encoder=model[1])

        model[0].fit(train_data, epochs=epochs,
                     batch_size=batch_size, callbacks=[custom_callback])

        return custom_callback


    def data_processor():
        pass


    def compute_output_arrays(df, columns):
        return np.asarray(df).reshape((-1, 1))


    def mask_sent_by_verb(long_text):
        line = long_text.strip()
        words = pseg.cut(line)
        none_verb_mask = 'XXX'
        verb_mask = 'VVV'
        without_verb = list()
        only_verb = list()
        for word, flag in words:
            if flag[0].lower() == 'v':
                only_verb.append(word)
                without_verb.append(verb_mask)
            else:
                only_verb.append(none_verb_mask)
                without_verb.append(word)
        with_v_sent = ''.join(only_verb)
        without_v_sent = ''.join(without_verb)
        return without_v_sent, with_v_sent


    def get_combos(tuple_list, n):
        res_list = list()
        while len(res_list) < n:
            one_tuple = list()
            for tup in tuple_list:
                one_ele = random.sample(tup, 1)[0]
                one_tuple.append(one_ele)
            res_list.append(one_tuple)
        return res_list


    from flask import Flask
    from flask import request, abort
    from flask import make_response, jsonify

    PARAM_TEXT = 'text'
    RESPONSE_OK = 200
    INVAILDPARAM = -1
    DECODE_TEXT_ERROR = 600
    model = None


    def predict_vec(d, nn_model):
        text = d[:maxlen]
        if use_postag:
            # s_nv, s_v = mask_sent_by_verb(text)
            x1, x2 = tokenizer.encode(text, text)
        else:
            x1, x2 = tokenizer.encode(text)
        X1 = [x1]
        X2 = [x2]
        X1 = seq_padding(X1)
        X2 = seq_padding(X2)
        R = nn_model.predict([X1, X2])[0]
        return R


    app = Flask(__name__)


    @app.route('/intent', methods=['POST'])
    def home():
        STATUS = "status"
        global graph
        try:
            data = json.loads(request.get_data())
            text = data[PARAM_TEXT][:256]
        except Exception as e:
            print(e)
            text = None
            res_dict = None
            error_num = DECODE_TEXT_ERROR
            return abort(error_num)
        bytes_result = None
        error_num = RESPONSE_OK
        with session_global.as_default():
            with session_global.graph.as_default():
                res_all = predict_one_sent(text, model)
                prob = np.max(res_all)
                res_all = np.argmax(res_all)
                code_question = index_label_dict[int(res_all)]
                result_main = label_question_dict[str(code_question)]
                res_dict = {
                    "intent_list": [{
                        "intent": result_main,
                        "intent_id": str(int(float(code_question))),
                        "base_code": "RENSHIBASE",
                        "intent_cat1": "人事政策",
                        "intent_cat2": "车辆使用费",
                        "prob": str(prob)
                    }],
                    "slot_dic": {},
                    "ret_code": 0
                }
                print('predicted ', res_dict)

                # Token logic
                if res_dict:
                    # bytes_result = wrap_2_json_str(res_dict)
                    bytes_result = make_response(jsonify(res_dict), 200)
                else:
                    error_num = INVAILDPARAM
                if error_num == 200 and bytes_result:
                    return bytes_result
                else:
                    return abort(error_num)


    session_global = None

    outputs = [one[1] for one in triple_sents]
    outputs = np.array(outputs)
    anchors = [one[0] for one in triple_sents]
    # positives = [one[1] for one in triple_sents]
    # negatives = [one[2] for one in triple_sents]
    inputs_anchor = compute_input_arrays_pairsent(anchors, maxlen, tokenizer=tokenizer)
    # inputs_pos = compute_input_arrays(positives, maxlen, tokenizer=tokenizer)
    # inputs_neg = compute_input_arrays(negatives, maxlen, tokenizer=tokenizer)
    # df_val_inputs = inputs_pos
    histories = []
    test_preds = []

    comb = [dropout, 1, 'sigmoid', 9]
    print('test combination ', comb)

    K.clear_session()
    session_global = tf.Session()
    tensorflow_backend.set_session(session_global)
    graph = tf.get_default_graph()
    train_model, model = bert_model(comb, cur_config)
    train_idx = list(range(len(triple_sents)))
    valid_idx = list(range(len(valid_output_categories)))
    list_train_idx = list(train_idx)
    list_valid_idx = list(valid_idx)
    batch_c1 = len(triple_sents) % batch_size
    batch_c2 = len(valid_output_categories) % batch_size
    if batch_c1 != 0:
        train_idx = train_idx[:-batch_c1]
    if batch_c2 != 0:
        valid_idx = valid_idx[:-batch_c2]
    inputs_anchor = [one_array[train_idx] for one_array in inputs_anchor]
    outputs = outputs[train_idx]
    train_in = []
    train_in.extend(inputs_anchor)
    # train_in.extend(inputs_pos)
    # train_in.extend(inputs_neg)
    train_in.append(outputs)

    valid_in = []
    # valid_in.extend(df_val_inputs)
    # valid_in.append(np.array(valid_output_categories))

    # train_inputs = [train_in[i][train_idx] for i in range(len(train_in))]
    train_inputs = train_in
    print(len(train_inputs[0]), len(train_inputs[1]), len(train_inputs[2]))
    valid_inputs = [valid_in[i][valid_idx] for i in range(len(valid_in))]
    history = train_and_predict((train_model, model),
                                train_data=train_inputs,
                                valid_data=train_inputs,
                                test_data=all_test_anwsers,
                                epochs=epoch_num, batch_size=batch_size,
                                fold=0)


