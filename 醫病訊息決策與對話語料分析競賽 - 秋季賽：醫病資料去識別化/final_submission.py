from kashgari.corpus import ChineseDailyNerCorpus
import os
import re

#train_x, train_y = ChineseDailyNerCorpus.load_data('train')
#valid_x, valid_y = ChineseDailyNerCorpus.load_data('validate')
#test_x, test_y  = ChineseDailyNerCorpus.load_data('test')

#print(f"train data count: {len(train_x)}")
#print(f"validate data count: {len(valid_x)}")
#print(f"test data count: {len(test_x)}")
file_path_train = 'train_2.txt'
data_path_train = 'train_2.data'

valid_path = 'dev.txt'
valid_data_path ='dev'



import tensorflow as tf
#tf.compat.v1.enable_eager_execution()

print(tf.__version__)

def loadInputFile(file_path):
    trainingset = list()  # store trainingset [content,content,...]
    position = list()  # store position [article_id, start_pos, end_pos, entity_text, entity_type, ...]
    mentions = dict()  # store mentions[mention] = Type
    with open(file_path, 'r', encoding='utf8') as f:
        file_text=f.read().encode('utf-8').decode('utf-8-sig')
    datas=file_text.split('\n\n--------------------\n\n')[:-1]
    for data in datas:
        data=data.split('\n')
        content=data[0]
        trainingset.append(content)
        annotations=data[1:]
        for annot in annotations[1:]:
            annot=annot.split('\t') #annot= article_id, start_pos, end_pos, entity_text, entity_type
            position.extend(annot)
            mentions[annot[3]]=annot[4]
    
    return trainingset, position, mentions


def CRFFormatData(trainingset, position, path):
    if (os.path.isfile(path)):
        os.remove(path)
    outputfile = open(path, 'a', encoding= 'utf-8')

    # output file lines
    count = 0 # annotation counts in each content
    tagged = list()
    for article_id in range(len(trainingset)):
        trainingset_split = list(trainingset[article_id])
        while '' or ' ' in trainingset_split:
            if '' in trainingset_split:
                trainingset_split.remove('')
            else:
                trainingset_split.remove(' ')
        start_tmp = 0
        for position_idx in range(0,len(position),5):
            if int(position[position_idx]) == article_id:
                count += 1
                if count == 1:
                    start_pos = int(position[position_idx+1])
                    end_pos = int(position[position_idx+2])
                    entity_type=position[position_idx+4]
                    if start_pos == 0:
                        token = list(trainingset[article_id][start_pos:end_pos])
                        whole_token = trainingset[article_id][start_pos:end_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ','')) == 0:
                                continue
                            # BIO states
                            if token_idx == 0:
                                label = 'B-'+entity_type
                            else:
                                label = 'I-'+entity_type
                            
                            output_str = token[token_idx] + ' ' + label + '\n'
                            outputfile.write(output_str)

                    else:
                        token = list(trainingset[article_id][0:start_pos])
                        whole_token = trainingset[article_id][0:start_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ','')) == 0:
                                continue
                            
                            output_str = token[token_idx] + ' ' + 'O' + '\n'
                            outputfile.write(output_str)

                        token = list(trainingset[article_id][start_pos:end_pos])
                        whole_token = trainingset[article_id][start_pos:end_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ','')) == 0:
                                continue
                            # BIO states
                            if token[0] == '':
                                if token_idx == 1:
                                    label = 'B-'+entity_type
                                else:
                                    label = 'I-'+entity_type
                            else:
                                if token_idx == 0:
                                    label = 'B-'+entity_type
                                else:
                                    label = 'I-'+entity_type

                            output_str = token[token_idx] + ' ' + label + '\n'
                            outputfile.write(output_str)

                    start_tmp = end_pos
                else:
                    start_pos = int(position[position_idx+1])
                    end_pos = int(position[position_idx+2])
                    entity_type=position[position_idx+4]
                    if start_pos<start_tmp:
                        continue
                    else:
                        token = list(trainingset[article_id][start_tmp:start_pos])
                        whole_token = trainingset[article_id][start_tmp:start_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ','')) == 0:
                                continue
                            output_str = token[token_idx] + ' ' + 'O' + '\n'
                            outputfile.write(output_str)

                    token = list(trainingset[article_id][start_pos:end_pos])
                    whole_token = trainingset[article_id][start_pos:end_pos]
                    for token_idx in range(len(token)):
                        if len(token[token_idx].replace(' ','')) == 0:
                            continue
                        # BIO states
                        if token[0] == '':
                            if token_idx == 1:
                                label = 'B-'+entity_type
                            else:
                                label = 'I-'+entity_type
                        else:
                            if token_idx == 0:
                                label = 'B-'+entity_type
                            else:
                                label = 'I-'+entity_type
                        
                        output_str = token[token_idx] + ' ' + label + '\n'
                        outputfile.write(output_str)
                    start_tmp = end_pos

        token = list(trainingset[article_id][start_tmp:])
        whole_token = trainingset[article_id][start_tmp:]
        for token_idx in range(len(token)):
            if len(token[token_idx].replace(' ','')) == 0:
                continue

            
            output_str = token[token_idx] + ' ' + 'O' + '\n'
            outputfile.write(output_str)

        count = 0
    
        output_str = '\n'
        outputfile.write(output_str)
        ID = trainingset[article_id]

        if article_id%10 == 0:
            print('Total complete articles:', article_id)

    # close output file
    outputfile.close()

training, position_train, mentions_train = loadInputFile(file_path_train)
valid_set, position_val, mentions_val = loadInputFile(valid_path)


CRFFormatData(training, position_train, data_path_train)
CRFFormatData(valid_set, position_val, valid_data_path)

# load `train.data` and separate into a list of labeled data of each text
# return:
#   data_list: a list of lists of tuples, storing tokens and labels (wrapped in tuple) of each text in `train.data`
#   traindata_list: a list of lists, storing training data_list splitted from data_list
#   testdata_list: a list of lists, storing testing data_list splitted from data_list
def Dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data=f.readlines()#.encode('utf-8').decode('utf-8-sig')
    data_list, data_list_tmp = list(), list()
    article_id_list=list()
    idx=0
    for row in data:
        data_tuple = tuple()
        if row == '\n':
            article_id_list.append(idx)
            idx+=1
            data_list.append(data_list_tmp)
            data_list_tmp = []
        else:
            row = row.strip('\n').split(' ')
            data_tuple = (row[0], row[1])
            data_list_tmp.append(data_tuple)
    if len(data_list_tmp) != 0:
        data_list.append(data_list_tmp)
    
    # here we random split data into training dataset and testing dataset
    # but you should take `development data` or `test data` as testing data
    # At that time, you could just delete this line, 
    # and generate data_list of `train data` and data_list of `development/test data` by this function
    return data_list, article_id_list


data_list,article_id_list= Dataset(data_path_train)
valid_data_list,valid_article_id_list = Dataset(valid_data_path)


def y_Preprocess(data_list):
    label_list = list()
    for idx_list in range(len(data_list)):
        label_list_tmp = list()
        for idx_tuple in range(len(data_list[idx_list])):
            label_list_tmp.append(data_list[idx_list][idx_tuple][1])
        label_list.append(label_list_tmp)
    return label_list

def x_Preprocess(data_list):
    label_list = list()
    for idx_list in range(len(data_list)):
        label_list_tmp = list()
        for idx_tuple in range(len(data_list[idx_list])):
            label_list_tmp.append(data_list[idx_list][idx_tuple][0])
        label_list.append(label_list_tmp)
    return label_list


x = x_Preprocess(data_list)
y = y_Preprocess(data_list)

x_test = x_Preprocess(valid_data_list)
y_test = y_Preprocess(valid_data_list)


from sklearn.model_selection import train_test_split


#X_train,valid_x,Y_train,valid_y = train_test_split(x, y, test_size=0.20, random_state=42)
#x_train,test_x,y_train,test_y,  = train_test_split(X_train,Y_train,test_size=0.10, random_state=42) # 0.25 x 0.8 = 0.2


#X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

X_train, valid_x, y_train, valid_y = train_test_split(x, y, test_size=0.20, random_state=1) # 0.25 

#train_x, train_y = ChineseDailyNerCorpus.load_data('train')
#valid_x, valid_y = ChineseDailyNerCorpus.load_data('validate')
#test_x, test_y  = ChineseDailyNerCorpus.load_data('test')

test_file_path='test_set.txt'
def testloadInputFile(file_path):
    trainingset = list()  # store trainingset [content,content,...]
    position = list()  # store position [article_id, start_pos, end_pos, entity_text, entity_type, ...]
    mentions = dict()  # store mentions[mention] = Type
    with open(file_path, 'r', encoding='utf8') as f:
        file_text=f.read().encode('utf-8').decode('utf-8-sig')
    datas=file_text.split('\n\n--------------------\n\n')[:-1]
    for data in datas:
        data=data.split('\n')
        content=data[1]
        trainingset.append(content)
        annotations=data[1:]
        for annot in annotations[1:]:
            annot=annot.split('\t') #annot= article_id, start_pos, end_pos, entity_text, entity_type
            position.extend(annot)
            mentions[annot[3]]=annot[4]

    return trainingset, position, mentions

testingset, position, mentions = testloadInputFile(test_file_path)

from typing import Dict, Any

from tensorflow import keras

from kashgari.tasks.labeling.base_model import BaseLabelingModel
from kashgari.layers import L
from tensorflow.python.keras.layers import Layer
from keras.optimizers import Adam, SGD


#from kashgari.corpus import SMP2018ECDTCorpus
from tensorflow import keras
import kashgari
#kashgari.config.use_CuDNN_cell = True
from kashgari.embeddings import BERTEmbedding

from kashgari.tasks.labeling import BiLSTM_Model, CNN_LSTM_Model

from sklearn.metrics import f1_score, recall_score, precision_score

from keras.callbacks import Callback
from keras import backend as K
from kashgari.callbacks import EvalCallBack
import logging
logging.basicConfig(level='DEBUG')

class NonMasking(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)
    def build(self, input_shape):
        input_shape = input_shape
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    def call(self, x, mask=None):
        return x
    def get_output_shape_for(self, input_shape):
        return input_shape


class DoubleBLSTMModel(BaseLabelingModel):
    """Bidirectional LSTM Sequence Labeling Model"""

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get hyper parameters of model
        Returns:
            hyper parameters dict
        """
        return {
            'layer_blstm1': {
                'units': 128,
                'return_sequences': True
            },
            'layer_blstm2': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dropout': {
                'rate': 0.4
            },
            'layer_time_distributed': {},
            'layer_activation': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self):
        """
        build model architectural
        """
        output_dim = len(self.processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # Define your layers
        layer_blstm1 = L.Bidirectional(L.LSTM(**config['layer_blstm1']),
                                       name='layer_blstm1')
        layer_blstm2 = L.Bidirectional(L.LSTM(**config['layer_blstm2']),
                                       name='layer_blstm2')

        layer_dropout = L.Dropout(**config['layer_dropout'],
                                  name='layer_dropout')

        layer_time_distributed = L.TimeDistributed(L.Dense(output_dim,
                                                           **config['layer_time_distributed']),
                                                   name='layer_time_distributed')
        layer_activation = L.Activation(**config['layer_activation'])

        # Define tensor flow
        tensor = layer_blstm1(embed_model.output)
        tensor = layer_blstm2(tensor)
        tensor = layer_dropout(tensor)
        tensor = layer_time_distributed(tensor)
        output_tensor = layer_activation(tensor)

        # Init model
        self.tf_model = keras.Model(embed_model.inputs, output_tensor)

tf_board_callback = keras.callbacks.TensorBoard(log_dir='model/logs', update_freq=1000)

bert_embed = BERTEmbedding('/tmp2/mluleki/NLPIR/project/chinese_L-12_H-768_A-12',
                                task=kashgari.LABELING,
                                sequence_length=256)

model = DoubleBLSTMModel(bert_embed)
# This step will build token dict, label dict and model structure
model.build_model(X_train,y_train, valid_x, valid_y)
# Compile model with custom optimizer, you can also customize loss and metrics.
# optimizer = RAdam()
# model.compile_model(optimizer=optimizer, loss=categorical_focal_loss(gamma=2.0, alpha=0.25))

eval_callback = EvalCallBack(kash_model=model,
                             valid_x=valid_x,
                             valid_y=valid_y,
                             step=2)
model.compile_model()

# Train model
model.fit(X_train, y_train, valid_x, valid_y, batch_size=16, epochs=90, callbacks=[eval_callback])
model.evaluate(x_test, y_test)
model.save('full_model')

new_model = kashgari.utils.load_model('full_model')
#loaded_model = load_model('full_model', load_weights=False)
new_model.tf_model.load_weights("full_model/model_weights.h5")


def cut_text(text, lenth):
    textArr = re.findall('.{' + str(lenth) + '}', text)
    textArr.append(text[(len(textArr) * lenth):])
    return textArr


def extract_labels(text, ners):
    ner_reg_list = []
    #if ners:
    #    new_ners = []
    #    for ner in ners:
    #        new_ners += ner;
    #print(len(ners))
    #print(len(text))

    for word, tag in zip(text, ners):
            #if tag != 'O':
        ner_reg_list.append((word, tag))
    # 输出模型的NER识别结果
    #print(len(ners))
    #print(len(text))
    #print(len(ner_reg_list))
    #print(ner_reg_list)
    
    labels = {}
    entity =None
    entities = []
    start = 0
    last = 0
    if ner_reg_list:
        for i, item in enumerate(ner_reg_list):
            if item[1].startswith('B'):
                label = ""
                entity  = {}
                end = i + 1
                while end <= len(ner_reg_list) - 1 and ner_reg_list[end][1].startswith('I'):
                    end += 1

                ner_type = item[1].split('-')[1]
                
                if ner_type not in labels.keys():
                    labels[ner_type] = []

                label += ''.join([item[0] for item in ner_reg_list[i:end]])
                 
                entity['start'] = i
                entity['end'] =  end
                labels[ner_type].append(label) 
                
                entity['tag'] = ner_type
                entity['name'] = label
                entities.append(entity)
    return entities


flat_y = []
full_y = []

#text_input = testingset[1]


#flat_x = [item for sublist in ners for item in sublist]

from tqdm import tqdm

#for j in tqdm(range(0,len(testingset))):
#    texts = cut_text(testingset[j], 256)
#    ners = new_model.predict([[char for char in text] for text in texts])
#    flat_y = [item for sublist in ners for item in sublist]
#    full_y.append((flat_y))

#labels =  extract_labels_text(text_input, ners)

#labels = extract_labels(text_input, ners)

def divide_chunks(l, n):      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

def get_Final(final_x):
    if len(final_x) >= 256:
        x = divide_chunks(final_x,256)
        full_x = []
        full_preds = []
        flat_x = []
    
        for item in x:
            #counter = counter+1
            texts = cut_text(item, 256)
            pred_tags= new_model.predict([[char for char in text] for text in texts])

            #pred_tags = new_model.predict(char for char in txt)
            #full_x.append(item)
            #full_x.append(remove_elements(tokens,final_x,counter))
            
            if len(pred_tags)>1:
                for c in pred_tags[0]:
                    full_preds.append(c)
                full_preds.append('O')
            else:
                for c in pred_tags[0]:
                    full_preds.append(c)

        #flat_x = [item for sublist in full_x for item in sublist]
        #flat_pred = [t for sublist in full_preds for t in sublist]
        #flat_pred = [item for sublist in full_preds for item in sublist]    
     
        
        #flat_list = []
        #for p in full_preds:
        #    for s_list in p:
        #        flat_list.append(s_list)

        return full_preds 
    else:
        texts = cut_text(final_x, 256)
        pred_tags= new_model.predict([[char for char in text] for text in texts])
        #pred_tags = new_model.predict(final_x)
        return  pred_tags


#texts = cut_text(sentence[test_id], 256)
#ners = new_model.predict([[char for char in text] for text in texts])
#ners = new_model.predict([[char for char in text] for text in texts])
#flat_ners = [tem for sublist in ners for tem in sublist]


def targetsLoad (sentence):
    output="article_id\tstart_position\tend_position\tentity_text\tentity_type\n"
    for test_id in tqdm(range(0,len(sentence))):
        flat_ners = get_Final(sentence[test_id])
        labels = extract_labels(sentence[test_id],flat_ners)   
        for s in labels:
            line=str(test_id)+'\t'+str(s['start'])+'\t'+str(s['end'])+'\t'+  ''.join(s['name'])+'\t'+s['tag']
            output+=line+'\n'
    return output

target_output = targetsLoad(testingset)
output_path='output.tsv'

with open(output_path,'w',encoding='utf-8') as f:
    f.write(target_output)

