from abc import ABCMeta
from abc import abstractmethod

class MosaAbstTrans(metaclass = ABCMeta):
    @abstractmethod
    def fit(self,x):
        pass
    
    @abstractmethod
    def transform(self,x,y):
        pass
    
class MosaAbstStack(metaclass = ABCMeta):
    @abstractmethod
    def fit_transform_train(self,x,y):
        pass
    
    @abstractmethod
    def transform_test(self,x):
        pass
    
    @abstractmethod
    def reset_model(self):
        pass