from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from MosaMeta import MosaAbstStackModel

class MosaLabelMeanModel(MosaAbstStackModel):
    def __init__(self,label_col):
        self.lab2mean = {}
        self.label_col = label_col
    
    def fit(self,x,y):
        min_df = pd.concat([x[self.label_col],y],axis = 1)
        min_df.columns = ["label","tar"]
        for name,g in min_df.groupby("label"):
            self.lab2mean[name] = g["tar"].mean()
            
    def predict(self,x):
        return list(map(lambda l:self.lab2mean[l],x[self.label_col].values))
    
    def reset(self):
        self.lab2mean = {}

class MosaStackBase:
    def fit_trans_train(self,x,y):
        skf = StratifiedKFold()
        buf = np.zeros(len(y))
        
        for train_ind,test_ind in skf.split(x,y):
            
            train_x = x.iloc[train_ind]
            test_x = x.iloc[test_ind]
            train_y = y.iloc[train_ind]

            self.model.fit(train_x,train_y)
            pred = self.model.predict(test_x)
            
            buf[test_ind] = pred
            self.model.reset()
        
        self.model.fit(x,y)
        temp_dic = {self.asname:buf}
        return x.assign(**temp_dic)
            
    def trans_test(self,x):
        pred = self.model.predict(x)
        temp_dic = {self.asname:pred}
        return x.assign(**temp_dic)
        
    

class MosaTargetEncoder(MosaStackBase):
    def __init__(self,label_col,assign_name=None):
        self.model = MosaLabelMeanModel(label_col)
        if assign_name:
            self.asname = assign_name
        else:
            self.asname = label_col+"_TME"
        