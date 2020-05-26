import numpy as np
import pandas as pd

class MosaDropper:
    def __init__(self,drop):
        self.dpcol = drop
    def fit(self,x,y):
        return self
    def transform(self,x):
        return x.drop(self.dpcol,axis=1)
    
class MosaDtypeFixer:
    def __init__(self,tars):
        self.tars = tars
    def fit(self,x,y):
        return self
    def transform(self,x):
        
        return x.astype({name:"float32" for name in self.tars})
        

class MosaStatsNumericNanfiller:
    def __init__(self,tar_col,filltype="outliner"):
        self.tar_col = tar_col
        self.filltype = filltype
        self.stuff = None
    
    def fit(self,x,y):
        if self.filltype == "outliner":
            rep_min = x[self.tar_col].min()
            self.stuff = rep_min-1
        elif self.filltype == "mean":
            rep_mean = x[self.tar_col].mean()
            self.stuff = rep_mean
        return self
    def transform(self,x):
        isnan = np.zeros(len(x.values))
        isnan[x[self.tar_col].isna()] = 1
        temp_dic = {self.tar_col+"_isNan":isnan}
        hoge = x.assign(**temp_dic)
        return hoge.fillna({self.tar_col:self.stuff})
    
class MosaPredNanfiller:
    def __init__(self,feature_cols,target_col,model):
        self.model = model
        self.target_col = target_col
        self.feature_cols = feature_cols
        
    def fit(self,x,y):
        valid = x[x[self.tar_col].notna()]
        self.model.fit(valid[self.feature_cols],valid[self.target_col])
        return self
        
    def transform(self,x):
        invalid = x[x[self.tar_col].isna()] 
        pred = self.model.predict(invalid[self.feature_cols])
        hoge = x
        hoge[x[self.tar_col].isna()] = pred
        return hoge
            
        