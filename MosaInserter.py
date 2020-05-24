import pandas as pd
import numpy as np

"""
meory_richがTrueのときはpandasのqueryではなく、予め外部データをid:{col:val,col:val...}
のディクショナリに変更することで、queryの計算量を落とす。
現状複数条件やIDかぶりには対応していないが、しかたない…
"""

class MosaInserter:
    outer_data = {}
    is_rich = {}
    def __init__(self,basename,tar,insert_cols,query_col,source_name,source_data,memory_rich = True):
        self.sname = source_name
        self.cols = insert_cols
        self.tar = tar
        self.qcol = query_col
        self.bname = basename
        if source_name not in MosaInserter.outer_data:
#             MosaInserter.outer_data[source_name] = source_data
            MosaInserter.is_rich[source_name] = memory_rich
            if not memory_rich:
                MosaInserter.outer_data[source_name] = source_data
            else:
                tars = source_data[query_col].values
                tp_data = source_data.drop(query_col,axis=1)
                nec_col = tp_data.columns
                tp_data = tp_data.values
                hoge = {}
                for i,key in enumerate(tars):
                    temp = {}
                    for j in range(len(nec_col)):
                        colname = nec_col[j]
                        temp[colname] = tp_data[i][j]
                    hoge[key] = temp
                MosaInserter.outer_data[source_name] = hoge
    
    def fit(self,x,y):
        
        return self
    
    def transform(self,x):
        hoge = x[self.tar].values
        
        if MosaInserter.is_rich[self.sname]:
            buf = np.zeros((len(x),len(self.cols)))
            tdata = MosaInserter.outer_data[self.sname]
            for i,val in enumerate(hoge):
                for j,col in enumerate(self.cols):
                    buf[i][j] = tdata[val][col]
            df = pd.DataFrame(buf)
            df.columns = [self.bname+col for col in self.cols]
            df = df.reset_index(drop=True)
            xt = x.reset_index(drop=True)
            tp = pd.concat([xt,df],axis=1)
            return tp
        else:  
            buf = np.zeros((len(x),len(self.cols)+1))
            tdata = MosaInserter.outer_data[self.sname][self.cols+[self.qcol]]
            for i,val in enumerate(hoge):
                fuga = tdata.query("{}=={}".format(self.qcol,val)).values[0]
                for j,val2 in enumerate(fuga):
                    buf[i][j] = val2
            df = pd.DataFrame(buf)
            df.columns = [self.bname+col for col in self.cols]+[self.qcol]
            df = df.drop(self.qcol,axis = 1)

            df = df.reset_index(drop=True)
            xt = x.reset_index(drop=True)
            tp = pd.concat([xt,df],axis=1)
            return tp