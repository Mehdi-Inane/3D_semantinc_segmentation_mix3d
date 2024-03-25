from dataset.dataset import S3DISDataset

data_path = 'data/'
dataset = S3DISDataset(data_path)
print(dataset[0][0].shape)
print(dataset[0][1].shape)
print(dataset[0][2].shape)