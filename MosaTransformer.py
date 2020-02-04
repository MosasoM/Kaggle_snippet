from sklearn.preprocessing import StandardScaler,MinMaxScaler
from MosaMeta import MosaAbstTrans

class MosaStandardScaler(MosaAbstTrans):
    def __init__(self,tar_cols):
        self.model = StandardScaler()
        self.tar_cols = tar_cols
    def fit(self,x):
        val = x[self.tar_cols]
        self.model.fit(val)
    
    def transform(self,x,y):
        tf = self.model.transform(x[self.tar_cols])
        tf = pd.DataFrame(tf)
        tf.columns = [col+"_stand_scale" for col in self.tar_cols]
        hoge = x.drop(self.tar_cols,axis=1)
        hoge = pd.concat([hoge,tf],axis=1)
        return hoge
    
class MosaMinMaxScaler(MosaAbstTrans):
    def __init__(self,tar_cols):
        self.model = MinMaxScaler()
        self.tar_cols = tar_cols
        
    def fit(self,x):
        val = x[self.tar_cols]
        self.model.fit(val)
    
    def transform(self,x,y):
        tf = self.model.transform(x[self.tar_cols])
        tf = pd.DataFrame(tf)
        tf.columns = [col+"_minmax_scale" for col in self.tar_cols]
        hoge = x.drop(self.tar_cols,axis=1)
        hoge = pd.concat([hoge,tf],axis=1)
        return hoge
    
class MosaClipper(MosaAbstTrans):
    def __init__(self,tar_cols):
        self.upper = None
        self.lower = None
        self.tar_cols = tar_cols
        
    def fit(self,x):
        val = x[self.tar_cols]
        self.lower,self.upper = np.percentile(val,[1,99],axis=0)
    
    def transform(self,x,y):
        tf = x[self.tar_cols].clip(self.lower,self.upper,axis=1)
        tf.columns = [col+"_clip1_99" for col in self.tar_cols]
        hoge = x.drop(self.tar_cols,axis=1)
        hoge = pd.concat([hoge,tf],axis=1)
        return hoge
    