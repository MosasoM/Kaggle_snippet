from MosaMeta import MosaAbstTrans
import pandas as pd
import numpy as np

class MosaDateTimeEncoder(MosaAbstTrans):
    def __init__(self,target,splitter,basename):
        self.target = target
        self.basename = basename
        self.sep = splitter
        
    def fit(self,x,y):
        return self
    
    def transform(self,x):
        tar = x[self.target].values
        lgt = len(x[self.target].values[0].split(self.sep))
        base = np.zeros((len(x.values),lgt))
        for i in range(len(x.values)):
            for j,val in enumerate(map(int,x[self.target].values[i].split(self.sep))):
                base[i][j] = val
        cols = [self.basename+str(i) for i in range(lgt)]
        df = pd.DataFrame(base)
        df.columns = cols
        df = df.reset_index(drop=True)
        xt = x.reset_index(drop=True)
        tp = pd.concat([xt,df],axis=1)
        return tp

class MosaLabelEncoder(MosaAbstTrans):
    def __init__(self,target,add_isnan=False,assign_name=None):
        self.num2label = {}
        self.target = target
        self.add_isnan = add_isnan
        if assign_name:
            self.asname = assign_name
        else:
            self.asname = target+"_LE"
    
    def fit(self,x,y):
        ind = 0
        for val in x[self.target].unique():
            if pd.isna(val):
                self.num2label[val] = -1
            else:
                self.num2label[val] = ind
                ind += 1
        return self
            
    def transform(self,x):
        tar = x[self.target].values
        tar = list(map(lambda l:self.num2label[l],tar))
        temp_dic = {self.asname:tar}
        hoge = x.assign(**temp_dic)
        if self.add_isnan:
            isnan = np.zeros(len(x.values))
            isnan[hoge.query("{}==-1".format(self.asname)).index] = 1
            temp_dic = {self.target+"isnan":isnan}
            hoge = hoge.assign(**temp_dic)
        return hoge

class MosaOneHotEncoder(MosaAbstTrans):
    def __init__(self,target_col,asname_base=None):
        self.tar_col = target_col
        if asname_base:
            self.asname_base = asname_base
        else:
            self.asname_base = target_col+"_OH"
        self.cat_num = 0
        
    def fit(self,x,y):
        self.cat_num = x[self.target_col].max()
        return self
        
    def transform(self,x):
        buf = np.zeros((len(x.values),self.cat_num))
        for i,val in enumerate(x[self.tar_col].values):
            if val != -1:
                buf[i][val-1] = 1
        t_df = pd.DataFrame(buf)
        t_df.columns = [self.asname_base+str(i+1) for i in range(self.cat_num)]
        t_df = t_df.reset_index(drop=True)
        xt = x.reset_index(drop=True)
        return pd.concat([xt,t_df],axis=1)
