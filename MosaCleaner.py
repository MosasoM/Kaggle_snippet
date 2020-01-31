import numpy as np

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
    def transform(self,x):
        isnan = np.zeros(len(x.values))
        isnan(x[self.tar_col].isna()) = 1
        temp_dic = {self.tar_col+"_isNan":isnan}
        hoge = x.assign(**temp_dic)
        return hoge.fillna({self.tar_col:self.stuff})
    
# class predictive_nanfiller:
    
            
        