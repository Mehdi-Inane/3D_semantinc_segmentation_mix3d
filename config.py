

class Config:
    def __init__(self):
        self.batch_size =  2
        self.epoch =  200
        self.learning_rate= 1e-3
        self.gpu= 1
        self.num_point= 1024
        self.optimizer= 'Adam'
        self.weight_decay= 1e-4
        self.normal= True
        self.lr_decay= 0.5
        self.step_size= 20
        self.nneighbor= 8
        self.nblocks= 2
        self.transformer_dim= 128
        self.name= 'Hengshuang'
        self.num_class = 13
        self.input_dim = 3