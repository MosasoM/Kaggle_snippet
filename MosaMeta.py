from abc import ABCMeta
from abc import abstractmethod

class MosaAbstTrans(metaclass = ABCMeta):
    @abstractmethod
    def fit(self,x):
        pass
    
    @abstractmethod
    def transform(self,x,y):
        pass
    
    
class MosaAbstStackModel(metaclass = ABCMeta):
    @abstractmethod
    def fit(self,x,y):
        pass
    
    @abstractmethod
    def predict(self,x):
        pass
    
    @abstractmethod
    def reset(self):
        pass