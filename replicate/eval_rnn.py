import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import argparse
import pickle
import json
import csv
import re
import os
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tensorflow.keras.layers import LSTM, Input, Dot, Softmax, Multiply, Concatenate, Subtract, Dense, Lambda, Embedding, Dropout
from tensorflow.keras.layers import Bidirectional
import matplotlib.pyplot as plt


def define_tokenizer(train_sentences, val_sentences, test_sentences):
    sentences = pd.concat([train_sentences, val_sentences, test_sentences])
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(sentences)
    
    return tokenizer


def encode(sentences, tokenizer, maxlen):
    encoded_sentences = tokenizer.texts_to_sequences(sentences)
    encoded_sentences = tf.keras.preprocessing.sequence.pad_sequences(encoded_sentences, maxlen=maxlen, padding='post')
    
    return encoded_sentences


def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')


def build_matrix(embeddings_index,word_index,max_features):
    embedding_matrix = np.zeros((max_features, 300))
    for word, i in tqdm(word_index.items()):
        if i >= max_features: continue
        try:
            embedding_vector = embeddings_index[word]
        except:
            embedding_vector = embeddings_index["unknown"]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def pipeline(tf_data, buffer_size=100, batch_size=32):
    tf_data = tf_data.shuffle(buffer_size)    
    tf_data = tf_data.prefetch(tf.data.experimental.AUTOTUNE)
    tf_data = tf_data.padded_batch(batch_size, padded_shapes=(([None],[None]),[]))
    
    return tf_data


def val_pipeline(tf_data, batch_size=1):        
    tf_data = tf_data.prefetch(tf.data.experimental.AUTOTUNE)
    tf_data = tf_data.padded_batch(batch_size, padded_shapes=(([None],[None]),[]))
    
    return tf_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mr",
        default='1-1',
        type=str,
        required=True
    )
    parser.add_argument(
        "--data_dir",
        default='./data',
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        default='./model',
        type=str,
        required=True
    )
    args = parser.parse_args()
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))   
    tf.config.list_physical_devices('GPU')

    pd.set_option('display.max_colwidth', -1)

    file_path = args.data_dir+'/source.jsonl'
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    question = []
    passage = []
    idx = []
    label = []
    for line in lines:
        i = json.loads(line.strip("\n"))
        question.append(i["question"])
        passage.append(i["passage"])
        idx.append(i["idx"])
        if i["label"]:
            label.append(1)
        else:
            label.append(0)
    f = open(args.data_dir+'/source.csv','w', encoding="utf-8")
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(("idx", "question", "passage", "label"))
    for i in range(len(idx)):
        writer.writerow((idx[i], question[i], passage[i], label[i]))
    f.close()

    file_path = args.data_dir+'/follow-up.jsonl'
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    question = []
    passage = []
    idx = []
    label = []
    for line in lines:
        i = json.loads(line.strip("\n"))
        question.append(i["question"])
        passage.append(i["passage"])
        idx.append(i["idx"])
        if i["label"]:
            label.append(1)
        else:
            label.append(0)
    f = open(args.data_dir+'/follow-up.csv','w', encoding="utf-8")
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(("idx", "question", "passage", "label"))
    for i in range(len(idx)):
        writer.writerow((idx[i], question[i], passage[i], label[i]))
    f.close()

    tokenizer = tf.keras.preprocessing.text.Tokenizer()

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    output_sources = []
    output_follows = []
    
    val_data = pd.read_csv(
        args.data_dir+'/source.csv', 
        usecols=['question', 'passage', 'label'], 
        dtype={'question': str, 'passage': str, 'label': np.int32}
    )

    SentenceLen = 256
    val_encoded_questions = encode(val_data['question'], tokenizer, SentenceLen)
    val_encoded_passages = encode(val_data['passage'], tokenizer, SentenceLen)


    EMBEDDING_FILE = './crawl-300d-2M.vec'
    max_features = len(tokenizer.word_index) + 1
    embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in open(EMBEDDING_FILE,encoding='utf8') if len(o)>100)
    fasttext_embedding_matrix = build_matrix(embeddings_index,tokenizer.word_index,max_features)

    tf_val_data = tf.data.Dataset.from_tensor_slices(((val_encoded_passages, val_encoded_questions), val_data['label'].values))

    tf_val_data = val_pipeline(tf_val_data, batch_size=128)

    embedding = tf.keras.layers.Embedding(
        len(tokenizer.word_index) + 1,
        300,
        embeddings_initializer = tf.keras.initializers.Constant(fasttext_embedding_matrix),
        trainable = False
    )

    bilstm1 = Bidirectional(LSTM(100, return_sequences=True, dropout=0.2))
    bilstm2 = Bidirectional(LSTM(100, return_sequences=True, dropout=0.2))

    i1 = Input(shape=(SentenceLen,), dtype='float32', name='input_question')
    i2 = Input(shape=(SentenceLen,), dtype='float32', name='input_passage')

    x1 = embedding(i1)
    x2 = embedding(i2)

    x1 = bilstm1(x1)
    x2 = bilstm1(x2)

    K = tf
    e = Dot(axes=2)([x1, x2])
    e1 = Softmax(axis=2)(e)
    e2 = Softmax(axis=1)(e)
    e1 = Lambda(K.expand_dims, arguments={'axis' : 3})(e1)
    e2 = Lambda(K.expand_dims, arguments={'axis' : 3})(e2)

    _x1 = Lambda(K.expand_dims, arguments={'axis' : 1})(x2)
    _x1 = Multiply()([e1, _x1])
    _x1 = Lambda(K.reduce_sum, arguments={'axis' : 2})(_x1)
    _x2 = Lambda(K.expand_dims, arguments={'axis' : 2})(x1)
    _x2 = Multiply()([e2, _x2])
    _x2 = Lambda(K.reduce_sum, arguments={'axis' : 1})(_x2)
    
    m1 = Concatenate()([x1, _x1, Subtract()([x1, _x1]), Multiply()([x1, _x1])])
    m2 = Concatenate()([x2, _x2, Subtract()([x2, _x2]), Multiply()([x2, _x2])])

    y1 = bilstm2(m1)
    y2 = bilstm2(m2)

    mx1 = Lambda(K.reduce_max, arguments={'axis' : 1})(y1)
    av1 = Lambda(K.reduce_mean, arguments={'axis' : 1})(y1)
    mx2 = Lambda(K.reduce_max, arguments={'axis' : 1})(y2)
    av2 = Lambda(K.reduce_mean, arguments={'axis' : 1})(y2)

    y = Concatenate()([av1, mx1, av2, mx2])
    y = Dense(100, activation='tanh')(y)
    y = Dropout(0.2)(y)
    y = Dense(1, activation='sigmoid', name='output')(y)

    model = tf.keras.Model(inputs=[i1, i2], outputs=y)
    model.summary()

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.0001),
        metrics=['accuracy', 'Precision', 'Recall']
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=10, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(args.output_dir+"/model_{epoch:02d}-{val_accuracy:.4f}.h5",verbose=0, save_weights_only=False,monitor='val_accuracy',save_best_only=False),
    ]

    model.load_weights(args.output_dir+"/model_RNN.h5")

    predictions = model.predict(tf_val_data)

    predictions = np.concatenate(predictions).round().astype(int)
    for i in predictions:
        output_sources.append(i)


    val_data = pd.read_csv(
        args.data_dir+'/follow-up.csv', 
        usecols=['question', 'passage', 'label'], 
        dtype={'question': str, 'passage': str, 'label': np.int32}
    )

    SentenceLen = 256
    val_encoded_questions = encode(val_data['question'], tokenizer, SentenceLen)
    val_encoded_passages = encode(val_data['passage'], tokenizer, SentenceLen)


    EMBEDDING_FILE = './crawl-300d-2M.vec'
    max_features = len(tokenizer.word_index) + 1
    embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in open(EMBEDDING_FILE,encoding='utf8') if len(o)>100)
    fasttext_embedding_matrix = build_matrix(embeddings_index,tokenizer.word_index,max_features)

    tf_val_data = tf.data.Dataset.from_tensor_slices(((val_encoded_passages, val_encoded_questions), val_data['label'].values))

    tf_val_data = val_pipeline(tf_val_data, batch_size=128)

    embedding = tf.keras.layers.Embedding(
        len(tokenizer.word_index) + 1,
        300,
        embeddings_initializer = tf.keras.initializers.Constant(fasttext_embedding_matrix),
        trainable = False
    )

    bilstm1 = Bidirectional(LSTM(100, return_sequences=True, dropout=0.2))
    bilstm2 = Bidirectional(LSTM(100, return_sequences=True, dropout=0.2))

    i1 = Input(shape=(SentenceLen,), dtype='float32', name='input_question')
    i2 = Input(shape=(SentenceLen,), dtype='float32', name='input_passage')

    x1 = embedding(i1)
    x2 = embedding(i2)

    x1 = bilstm1(x1)
    x2 = bilstm1(x2)

    K = tf
    e = Dot(axes=2)([x1, x2])
    e1 = Softmax(axis=2)(e)
    e2 = Softmax(axis=1)(e)
    e1 = Lambda(K.expand_dims, arguments={'axis' : 3})(e1)
    e2 = Lambda(K.expand_dims, arguments={'axis' : 3})(e2)

    _x1 = Lambda(K.expand_dims, arguments={'axis' : 1})(x2)
    _x1 = Multiply()([e1, _x1])
    _x1 = Lambda(K.reduce_sum, arguments={'axis' : 2})(_x1)
    _x2 = Lambda(K.expand_dims, arguments={'axis' : 2})(x1)
    _x2 = Multiply()([e2, _x2])
    _x2 = Lambda(K.reduce_sum, arguments={'axis' : 1})(_x2)
    
    m1 = Concatenate()([x1, _x1, Subtract()([x1, _x1]), Multiply()([x1, _x1])])
    m2 = Concatenate()([x2, _x2, Subtract()([x2, _x2]), Multiply()([x2, _x2])])

    y1 = bilstm2(m1)
    y2 = bilstm2(m2)

    mx1 = Lambda(K.reduce_max, arguments={'axis' : 1})(y1)
    av1 = Lambda(K.reduce_mean, arguments={'axis' : 1})(y1)
    mx2 = Lambda(K.reduce_max, arguments={'axis' : 1})(y2)
    av2 = Lambda(K.reduce_mean, arguments={'axis' : 1})(y2)

    y = Concatenate()([av1, mx1, av2, mx2])
    y = Dense(100, activation='tanh')(y)
    y = Dropout(0.2)(y)
    y = Dense(1, activation='sigmoid', name='output')(y)

    model = tf.keras.Model(inputs=[i1, i2], outputs=y)
    model.summary()

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.0001),
        metrics=['accuracy', 'Precision', 'Recall']
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=10, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(args.output_dir+"/model_{epoch:02d}-{val_accuracy:.4f}.h5",verbose=0, save_weights_only=False,monitor='val_accuracy',save_best_only=False),
    ]

    model.load_weights(args.output_dir+"/model_RNN.h5")

    predictions = model.predict(tf_val_data)

    predictions = np.concatenate(predictions).round().astype(int)
    for i in predictions:
        output_follows.append(i)
    
    violations = []
    num = 0
    for source in output_sources:
        follow = output_follows[num]
        if source != follow:
            violations.append(source)
        num += 1
    
    if args.mr in ["1-1", "1-2", "1-3", "1-4"]:
        print("violation rate: ", 1-len(violations)/len(output_sources))
    elif args.mr in ["2-1", "2-2", "2-3"]:
        print("violation rate: ", len(violations)/len(output_sources))
    
    
    os.remove(args.data_dir+'/source.csv')
    os.remove(args.data_dir+'/follow-up.csv')

if __name__ == "__main__":
    main()







