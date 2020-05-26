from MosaMeta import MosaAbstTrans
import numpy as np
import pandas as pd


class MosaCross(MosaAbstTrans):
    def __init__(self,name,tar1,tar2,func):
        self.tar1 = tar1
        self.tar2 = tar2
        self.func = func
        self.name = name
        
    def fit(self,x,y):
        return self
    
    def transform(self,x):
        td1 = x[self.tar1].values
        td2 = x[self.tar2].values
        buf = np.zeros(len(td1))
        for i in range(len(td1)):
            buf[i] = self.func(td1[i],td2[i])
        tx = x.assign(hoge=buf)
        hoge = list(tx.columns)
        hoge[-1] = self.name
        tx.columns = hoge
        return tx