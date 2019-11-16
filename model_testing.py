from create_dataset import create_dataset,heading

# General Packages
import pandas as pd
import re
import numpy as np
import random
import time

# Basic ML Packages
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score,classification_report

# CRF Packages
from sklearn_crfsuite import CRF

# BiLSTM Packages
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional,Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from gensim.models import Word2Vec

def train_crf(x,y):
    '''train a crf model on x and y data'''
    crf = CRF(algorithm='lbfgs',
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)
    crf.fit(x, y)
    return crf

def eval_crf(crf,x_test,y_test):
    '''Evaluate a trained crf model on x and y data'''
    pred = crf.predict(x_test)
    report = classification_report(y_test,pred)
    heading("CRF Test Set Classification Report")
    print(report)
    f1 = f1_score(y_test,pred)
    heading(f"Test F1-Score:{f1}")
    return f1

def create_weight_matrix(word_ids,embeddings_index):
    '''Loads pretrained weights and maps them to specified word index'''
    embedding_matrix = np.zeros((len(word_ids),100))
    count = 0
    oov_words = []
    for word,idx in word_ids.items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]
        else:
            if word == "<PAD>":
                embedding_matrix[idx] = np.array([999]*100)
            else:
                oov_words.append(word)
    return embedding_matrix,oov_words    

def create_BiLSTM(n_words,n_tags,embedding_size,max_len,embed_weights):
    '''Create a Single Layer BiLSTM Model'''
    model = Sequential()
    model.add(Embedding(n_words,
                        embedding_size,
                        weights=[embed_weights],
                        trainable=False,
                        input_length=max_len))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.2)))
    model.add(TimeDistributed(Dense(n_tags, activation="softmax")))
    return model

def train_BiLSTM(model,x_train,y_train,batch_size,epochs,val_split):
    '''Train a keras model given x and y data, and training parameters'''
    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.0001,
                               patience=0,
                               mode='min',
                               verbose=1)
    
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(x_train, y_train, 
                        batch_size=32, 
                        epochs=epochs, 
                        validation_split=val_split, 
                        verbose=1,
                        callbacks=[early_stop]
                       )
    return history

def get_id_mappings(ids):
    '''Swaps items of dictionary'''
    return {str(i[1]):i[0] for i in ids.items()}

def transform_ids_to_tags(preds,tag_ids):
    '''Transforms tag_ids to word representation'''
    id_to_tags = get_id_mappings(tag_ids)

    tag_seqs = []
    for seq in preds:
        tag_seqs.append([id_to_tags[str(i)] for i in seq])
    return tag_seqs

def get_real_labels(model,x_test,y_test,tag_ids):
    '''Get word labels of true values and model predictions'''
    test_preds = np.argmax(model.predict(x_test),axis=-1)
    true_vals = np.argmax(y_test,axis=-1)
    test_preds = transform_ids_to_tags(test_preds,tag_ids)
    true_vals = transform_ids_to_tags(true_vals,tag_ids)
    return true_vals,test_preds

def eval_BiLSTM(model,x_test,y_test,tag_ids):
    '''Evaluates a trained keras model give x,y and tag_ids'''
    true_vals,preds = get_real_labels(model,x_test,y_test,tag_ids)
    report = classification_report(true_vals,preds)
    heading("BiLSTM Test Set Classification Report")
    print(report)
    f1 = f1_score(true_vals,preds)
    heading(f"Test F1-Score:{f1}")
    return f1

def save_f1(file,f1,n_samples,model_desc,note):
    '''Saves F1-Score to specified file'''
    with open(file,'a') as f:
        results = f"\n{f1},{n_samples},{model_desc},{time.ctime()},{note}"
        f.writelines(results)
    print("~~~Results Successfully Saved~~~")
    return

def train_and_eval_crf(date_file,save_file,save=True):
    '''Train and evaluate a CRF model. Option to save F1 Score.'''
    x,y = create_dataset(data_file,"CRF")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    crf = train_crf(x_train,y_train)
    f1 = eval_crf(crf,x_test,y_test)
    if save:
        desc = "Simple CRF Model"
        note = "None"
        save_f1(save_file,f1,len(x),desc,note)
    return

def train_and_eval_BiLSTM(data_file,save_file,save=True,embed_size=100,epochs=50,batch_size=32,val_size=0.1):
    '''Train and evaluate a Keras BiLSTM Model. Option to save F1 Score.'''
    x,y,word_ids,tag_ids = create_dataset(data_file,"LSTM")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    custom_emb = Word2Vec.load("./data/word2vec_numeric_encs.model")
    embed_matrix,_ = create_weight_matrix(word_ids,custom_emb)
    model = create_BiLSTM(len(word_ids),len(tag_ids),embed_size,50,embed_matrix)
    history = train_BiLSTM(model,x_train,y_train,batch_size,epochs,val_size)
    f1 = eval_BiLSTM(model,x_test,y_test,tag_ids)
    if save:
        desc = f"BiLSTM-EmbedSize-{embed_size}-Word2Vec"
        note = "Word2Vec Embeddings"
        save_f1(save_file,f1,len(x),desc,note)
    return 

if __name__ == "__main__":
    # Testing for train_and_eval functions
    data_file = "./data/Medical NER V2 2000.tsv"
    results_file = "./data/model_results.csv"
    heading("Training CRF Model...")
    train_and_eval_crf(data_file,results_file)
    heading("Training BiLSTM Model...")
    train_and_eval_BiLSTM(data_file,results_file)