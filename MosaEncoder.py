from MosaMeta import MosaAbstTrans
import pandas as pd

class MosaLabelEncoder(MosaAbstTrans):
    def __init__(self,target,add_isnan=True,assign_name=None):
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
            
    def transform(self,x):
        tar = x[self.target].values
        tar = list(map(lambda l:self.num2label[l],tar))
        temp_dic = {self.asname:tar}
        hoge = x.assign(**temp_dic)
        if self.add_isnan:
            isnan = np.zeros(len(x.values))
            isnan[hoge.query("@self.asname==-1").index] = 1
            temp_dic = {self.target+"isnan":isnan}
            hoge = hoge.assign(**temp_dic)
        return hoge
    
