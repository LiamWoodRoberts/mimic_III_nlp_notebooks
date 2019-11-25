import pandas as pd
import json
from seqeval.metrics import f1_score,classification_report
from sklearn.model_selection import train_test_split
from nltk import pos_tag
from sklearn_crfsuite import CRF

def read_turks(file):
    with open(file) as f:
        lines = [i.rstrip().split("\t") for i in f.readlines()]
    return lines

def create_seqs(word_ents):
    seqs = []
    seq = []
    for ents in word_ents:
        if len(ents)>1:
            if len(ents[0])>0:
                if ents[0][-1] == ".":
                    seq.append([ents[0][:-1],ents[1]])
                if len(ents[0])>1:
                    seq.append([ents[0].replace(",",""),ents[1]])
                else:
                    seq.append(ents)
        else:
            seqs.append([i for i in seq if len(i[0])>0])
            seq=[]
    return seqs

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

def add_pos(seqs):
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

def create_crf_data(file):
    seqs = read_turks(file)
    seqs = create_seqs(seqs)
    seqs = [clean_tags(ents) for ents in seqs]
    seqs = add_pos(seqs)
    x = [sent2features(s) for s in seqs]
    y = [sent2labels(s) for s in seqs]
    tokens = [sent2tokens(s) for s in seqs]
    return x,y,tokens

def train_crf(x,y):
    crf = CRF(algorithm='lbfgs',
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)

    crf.fit(x, y)
    
    return crf

def eval_crf(x,y,split_size = 0.1):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_size, random_state=42)
    crf = train_crf(x_train,y_train)

    # Evaluate Test Performance
    pred = crf.predict(x_test)
    report = classification_report(y_test,pred)
    print("Test Results:\n","-"*60)
    print(report)
    print("-"*60)
    f1 = f1_score(y_test,pred)
    return f1

def load_sentances(file):
    with open(file) as f:
            lines = [i.rstrip().split(" ") for i in f.readlines()]
    return lines

def add_pos_sentances(seqs):
    new_seqs = []
    for sentance in seqs:
        pos = pos_tag(sentance)        
        new_seqs.append(pos)
    return new_seqs

def remove_null_words(seq):
    return [word for word in seq if len(word)>0]

def prep_unannotated_data(file):
    # Load Data 
    seqs = load_sentances(file)
    
    # Remove null words
    seqs = [remove_null_words(seq) for seq in seqs]
    
    # Add CRF Features
    seqs = add_pos_sentances(seqs)
    return seqs

def create_annotations(crf,seqs):
    x = [sent2features(s) for s in seqs]
    ner_tags = crf.predict(x)
    return ner_tags

def combine_B_I_tags(seq):
    combi_seq = []
    phrase = ""
    for word,tag in seq:
        if tag == "O":
            if len(phrase)>0:
                combi_seq.append([phrase,phrase_tag])
                phrase = ""
            combi_seq.append([word,tag])
        else:
            if tag[0] == "B":
                if len(phrase)>0:
                    combi_seq.append([phrase,phrase_tag])
                    phrase = ""
                phrase = word
                phrase_tag = tag[2:]
            if tag[0] == "I":
                phrase += " "+word
    if len(phrase)>0:
        combi_seq.append([phrase,phrase_tag])
    return combi_seq

def create_dataset(crf,seqs):
    ner_tags = create_annotations(crf,seqs)
    data = []
    for i in range(len(seqs)):
        data_entry = [[seqs[i][j][0],ner_tags[i][j]] for j in range(len(seqs[i]))]
        data.append(data_entry)
    data = [combine_B_I_tags(seq) for seq in data]
    return data

def get_points(seq):
    start = 0
    end = 0
    starts = []
    ends = []
    for entry in seq:
        starts.append(start)
        start += len(entry[0])+1
        ends.append(start-2)
    new_seq = [[seq[i][0],seq[i][1],starts[i],ends[i]] for i in range(len(seq))]
    return new_seq

def get_annotation(seq,ignore_ents):
    new_seq = get_points(seq)
    annotations = []
    for text,tag,start,end in new_seq:
        if tag not in ignore_ents:
            annot = {}
            annot["label"] = [tag]
            annot["points"] = [{"start":start,"end":end,"text":text}]
            annotations.append(annot)
    return annotations

def write_json(file,data,ignore_tags):
    with open(file,"w") as f:
        for seq in data:
            line = {}
            line["content"] = " ".join([i[0] for i in seq])
            line["annotation"]  = get_annotation(seq,ignore_tags)
            line["extras"] = {"Name":"ColumnName","Class":"ColumnValue"}
            f.write(json.dumps(line))
            f.write("\n")
    print("~~~Annotations Saved~~~")
    return
    

def smart_predict(annotated_file,unannotated_file,save_file,ignore_tags=["O"],eval_only=False):
    '''
    Inputs:
    
    annotated_file - file path to labelled data in .tsv format outputed from dataturks
    
    unannotated_file - file path to .txt file with each line containing an unannotated sentance
    
    save_file - file path to annotations generatated by crf model for upload to dataturks
    
    eval_only - if set to True will print a classification report using 10% of the annotated data as
                a test set. Can be used to determine which tags to ignore.
    
    Outputs:
    
    None
    
    Desc:
    
    Loads annotated data, trains crf model. Makes predictions on unannottated sentances and saves
    output for upload to dataturks annotation service. 
    
    
    '''
    
    x,y,_ = create_crf_data(annotated_file)
    
    if eval_only:
        eval_crf(x,y)
        return
    
    crf = train_crf(x,y)

    seqs = prep_unannotated_data(unannotated_file)
    data = create_dataset(crf,seqs)

    write_json(save_file,data,ignore_tags)
    return


if __name__ == "__main__":
    annotated_file = "./data/Medical NER V2 4000.tsv"
    unannoted_file = "./data/sentances_0-10.txt"
    save_file = "./data/pre_annot_sentances_0-10.txt"

    ignore_tags = ["O",
               "Duration",
               "BODY",
              ]
              
    smart_predict(annotated_file,unannoted_file,save_file,ignore_tags)