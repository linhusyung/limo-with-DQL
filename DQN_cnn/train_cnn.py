import numpy as np

class Dataset():
    def __init__(self):
        self.train_input_root = 'dataset/train_input'
        self.target=np.load('target/Target.npy')

if __name__ == '__main__':
    c=Dataset()
    print(c.target.shape,c.target)