# General Packages
import re
import numpy as np

# For LSTM Dataset
from tensorflow.keras.utils import to_categorical

# For CRF Dataset
from nltk import pos_tag

def read_turks(file):
    '''Read Dataturks TSV file'''
    with open(file) as f:
        lines = [i.rstrip().split("\t") for i in f.readlines()]
    return lines

def clean_words(word_ents,l):
    '''removes quote and comma characters from'''
    new_word_ents = []
    for ents in word_ents:
        word = ents[0]
        if l:
            word = word.lower()
        word = word.replace('"','')
        ents[0] = word
        new_word_ents.append(ents)
    return new_word_ents

def create_seqs(word_ents):
    '''Eliminate empty list entries and trailing periods'''
    seqs = []
    seq = []
    for ents in word_ents:
        if len(ents)>1:
            if len(ents[0])>0:
                if ents[0][-1] == ".":
                    seq.append([ents[0][:-1],ents[1]])
                else:
                    seq.append(ents)
        else:
            seqs.append(seq)
            seq=[]
    return seqs

def expand_word(ent):
    '''Splits at specified special characters keeping special characters as their own value'''
    words = re.split('([/\-\%><])',ent[0])
    return [[i,ent[1]] for i in words if len(i)>0]

def expand_special_chars(seqs):
    '''Expands special characters in words into seperate words while '''
    new_seqs = []
    for seq in seqs:
        new_seq = []
        for word in seq:
            new_seq += expand_word(word)
        new_seqs.append(new_seq)
    return new_seqs

def encode_numerics(seq):
    '''Add encodings for common number types'''
    enc_seq = []
    for ent in seq:
        enc = ent[0].strip()
        if re.match("^\d$",ent[0]) != None:
            enc = "<1DigitNum>"
        elif re.match("^\d\d$",ent[0]) != None:
            enc = "<2DigitNum>"
        elif re.match("^\d\d\d$",ent[0]) !=None:
            enc = "<3DigitNum>"
        elif re.match("^\d{4}$",ent[0]) != None:
            enc = "4DigitNum"
        elif re.match("^\d*\.\d*$",ent[0]) != None:
            enc = "<DecimalNum>"
        elif re.match("^\d+,\d+$",ent[0]) != None:
            enc = "<CommaNum>"
        elif re.match("^\d+'?s$",ent[0]) !=None:
            enc = "<RangeNum>"
            
        enc_seq.append([enc,ent[1]])
    return enc_seq

def clean_tags(word_ents):
    '''adds IOB scheme to tags'''
    new_ents = []
    for i in range(0,len(word_ents)):
        if word_ents[i][1] == "O":
            tag = word_ents[i][1]
        else:
            if not i:
                tag = "B-"+word_ents[i][1]
            else:
                if (word_ents[i][1] != word_ents[i-1][1]):
                    tag = "B-"+word_ents[i][1]
                else:
                    tag = "I-"+word_ents[i][1]

        new_ents.append([word_ents[i][0],tag])
    return new_ents

def pad_seq(seq,max_len):
    '''Pad / truncate a sequence to a specified length'''
    padded_seq = seq+[["<PAD>","O"]]*max_len
    return padded_seq[:max_len]
    
def pad_sequences(sequences,max_len=None):
    '''Pad / Truncate a list of sequences to a specified length'''
    if max_len == None:
        max_len = max(len(seq) for seq in sequences)
    return [pad_seq(seq,max_len) for seq in sequences]

def get_word_ids(sentances,tag=False):
    '''Enumerates all unique words from a list of sequences'''
    words = []
    for sentance in sentances:
        words += list([word[tag] for word in sentance])
    word_dict = {word:i for i,word in enumerate(set(words))}
    return word_dict

def words_to_ids(sentances,word_ids,tag_ids):
    '''Maps words and entity tags to numeric indices of respective dictionary'''
    vector = []
    for sentance in sentances:
        vector.append(list([[word_ids[w[0]],tag_ids[w[1]]] for w in sentance]))
    return np.array(vector)

def create_x_y(matrix,n_tags):
    '''Splits Sequences and Entities into x and y variables'''
    x = []
    y = []
    for sequences in matrix:
        xi = [i[0] for i in sequences]
        yi = [i[1] for i in sequences]
        x.append(xi)
        y.append(yi)
    y = np.array([to_categorical(i,n_tags) for i in y])
    return np.array(x),np.array(y)

def create_sequences(file,pad=True,lower=True,max_len=50):
    '''Creates dataset with optional padding and lowercasing for sequence models'''
    word_ents = read_turks(file)
    new_ents = clean_words(word_ents,lower)
    seqs = create_seqs(new_ents)
    ex_seqs = expand_special_chars(seqs)
    enc_seqs = [encode_numerics(seq) for seq in ex_seqs]
    cleaned_tag_seqs = [clean_tags(ents) for ents in enc_seqs]
    if pad:
        padded_seqs = pad_sequences(enc_seqs,max_len)
        return padded_seqs
    else:
        return cleaned_tag_seqs

def add_pos(seqs):
    '''Add POS tags to sequences for CRF Model'''
    new_seqs = []
    for sentance in seqs:
        words = [word[0] for word in sentance]
        pos = pos_tag(words)        
        new_seq = [pos[i]+(sentance[i][1],) for i in range(len(sentance))]
        new_seqs.append(new_seq)
    return new_seqs

def word2features(sent, i):
    '''
    
    From:
    https://www.depends-on-the-definition.com/named-entity-recognition-conditional-random-fields-python/
    
    '''
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def encode_for_CRF(cleaned_tag_seqs,verbose):
    '''Formats sequences for use in CRF models'''
    pos_seqs = add_pos(cleaned_tag_seqs)
    x = [sent2features(s) for s in pos_seqs]
    y = [sent2labels(s) for s in pos_seqs]
    if verbose:
        print("X-shape:",np.array(x).shape)
        print(x[0][:5])
        print('')
        print("Y-shape:",np.array(y).shape)
        print(y[0][:5])
    return x,y

def encode_for_LSTM(padded_seqs,verbose):
    '''Formats sequenes for use in Neural Sequence Models'''
    word_ids = get_word_ids(padded_seqs)
    tag_ids = get_word_ids(padded_seqs,tag=True)
    vectors = words_to_ids(padded_seqs,word_ids,tag_ids)
    n_tags = len(tag_ids)
    x,y = create_x_y(vectors,n_tags)
    if verbose:
        print("X-shape:",x.shape)
        print(x[0][:5])
        print('')
        print("Y-shape:",y.shape)
        print(y[0][:5])
    return x,y,word_ids,tag_ids

def create_dataset(file,type,max_len=50,verbose=False):
    '''Creates dataset for a CRF or Neural Sequence Model'''
    if type == "CRF":
        seqs = create_sequences(file,pad=False,lower=False)
        x,y = encode_for_CRF(seqs,verbose)
        return x,y
    if type == "LSTM":
        seqs = create_sequences(file)
        x,y,word_ids,tag_ids = encode_for_LSTM(seqs,verbose)
        return x,y,word_ids,tag_ids
    else:
        return print("Options are 'CRF' or 'LSTM'." )

def line(n):
    '''Prints a dashed line of specified length'''
    print("-"*n)
    return

def heading(text,n=60):
    '''Prints text with above and below dashed line borders'''
    line(n)
    print(f"\n{text}\n")
    line(n)
    return 

if __name__ == "__main__":
    file = "./data/Medical NER V2 2000.tsv"
    heading("LSTM Test Output")
    x_lstm,y_lstm,_,_ = create_dataset(file,type="LSTM",verbose=True)
    heading("CRF Test Output")
    x_crf,y_crf = create_dataset(file,type="CRF",verbose=True)
    heading("~~~Datasets Loading Successfully~~~")