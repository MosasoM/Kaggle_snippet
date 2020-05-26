from MosaMeta import MosaAbstTrans
import numpy as np
import pandas as pd


class MosaCross(MosaAbstTrans):
    def __init__(self,name,tars,func):
        self.tars = tars
        self.func = func
        self.name = name
        
    def fit(self,x,y):
        return self
    
    def transform(self,x):
        td = x[self.tars].values
        buf = np.zeros(len(td))
        for i in range(len(td)):
            buf[i] = self.func(*td[i])
        tx = x.assign(hoge=buf)
        hoge = list(tx.columns)
        hoge[-1] = self.name
        tx.columns = hoge
        return tx