#!/usr/bin/env python
#-*- encoding: utf8 -*-

import os
import json
import numpy as np
import rospy
import rospkg
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from enum import Enum
from mind_msgs.msg import Reply, RaisingEvents
from mind_msgs.srv import ReadData, ReadDataResponse, WriteData, WriteDataResponse, RegisterData, RegisterDataResponse, GetDataList, GetDataListResponse, GoogleEntity

from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config

class PosTagger:
    
    def __init__(self):
        
        self.entities = {
            '<patient_name>' : None,
            '<address>' : None,
            '<time>' : None,
        }

        rospy.loginfo('\033[94m[%s]\033[0m initialized.'%rospy.get_name())        
        self.srv_read_data = rospy.Service('%s/read_utterance'%rospy.get_name(), 
                                    GoogleEntity, self.extract_entities)

        config = Config()
        
        self.model = NERModel(config)
        self.model.build()
        self.model.restore_session(config.dir_model)

        test = CoNLLDataset(config.filename_test, config.processing_word,
                            config.processing_tag, config.max_iter)

        self.model.evaluate(test)

    
    def extract_entities(self, utterance):
        utterance = utterance.utterance

        if utterance == 'clear':
            self.entities = {
                '<patient_name>' : None,
                '<address>' : None,
                '<time>' : None,
            }
            entities = json.dumps(self.entities)
            result_string = utterance
        
        elif utterance != '':
            tokenized = utterance.lower().strip().split(" ")
            preds = self.model.predict(tokenized)

            # result
            print(tokenized, preds)

            for p in preds:
                if p == 'B-PER' or p == 'I-PER':
                    if self.entities['<patient_name>'] != None:
                        self.entities['<patient_name>'] += ' ' + tokenized[preds.index(p)]
                        tokenized.remove(tokenized[preds.index(p)])
                    else:
                        self.entities['<patient_name>'] = tokenized[preds.index(p)]
                        tokenized = [r.replace(tokenized[preds.index(p)], 
                                            '<patient_name>') for r in tokenized]
                
                elif p == 'B-LNUM' or p == 'B-LOC' or p == 'I-LOC':
                    if self.entities['<address>'] != None:
                        try:
                            self.entities['<address>'] += ' ' + tokenized[preds.index(p)]
                        except IndexError:
                            self.entities['<address>'] += ' ' + tokenized[preds.index(p)-1]
                        
                        try:
                            tokenized.remove(tokenized[preds.index(p)])
                        except IndexError:
                            tokenized.remove(tokenized[preds.index(p)-1])
                    else:
                        self.entities['<address>'] = tokenized[preds.index(p)]
                        tokenized = [r.replace(tokenized[preds.index(p)], 
                                            '<address>') for r in tokenized]
                
                elif p == 'B-TIME' or p == 'I-TIME':
                    if self.entities['<time>'] != None:
                        self.entities['<time>'] += ' ' + tokenized[preds.index(p)]
                        tokenized.remove(tokenized[preds.index(p)])
                    else:
                        self.entities['<time>'] = tokenized[preds.index(p)]
                        tokenized = [r.replace(tokenized[preds.index(p)], 
                                            '<time>') for r in tokenized]
            entities = json.dumps(self.entities)
            result_string = ' '.join(tokenized)
        else:
            rospy.logerr('utterance empty')
    
        return entities, result_string
        

        

if __name__ == '__main__':
    rospy.init_node('pos_tagging', anonymous=False)
    p = PosTagger()
    rospy.spin()
