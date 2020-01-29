from MosaMeta import MosaAbstTrans

class MosaLabelEncoder(MosaAbstTrans):
    def __init__(self,target,assign_name=None):
        self.num2label = {}
        self.target = target
        if assign_name:
            self.asname = assign_name
        else:
            self.asname = target+"_LE"
    
    def fit(self,x,y):
        ind = 0
        for val in x[self.target].unique():
            self.num2label[val] = ind
            ind += 1
            
    def transform(self,x):
        tar = x[self.target].values
        tar = list(map(lambda l:self.num2label[l],tar))
        temp_dic = {self.asname:tar}
        return x.assign(**temp_dic)
    
