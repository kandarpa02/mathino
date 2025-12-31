from .dataset_base import dataset

__all__ = ["mnist"]

class mnist(dataset):
    def __init__(self, path=None):
        super().__init__(
            path='mathino/data/storage/mnist.npz' if path==None else path, 
            link="https://drive.google.com/file/d/1Bmft6ApTEqadnWR5m-7Ozuy7x3pAnlYx/view?usp=sharing"
            )
    