import numpy as np
import torch
from torchvision import transforms, datasets

# CWD from which this is called should contain the following
# dataImages -> Air -> (# 256 x 256 rgb images)
# dataImages -> Sat -> (# 256 x 256 rgb images)
# dataImages -> Road -> (# 256 x 256 rgb images)
def imageLoader(inputPath, targetPath, Height=128, Width=128, shuffleData=False):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5 ],
                                         std=[0.5 ,0.5, 0.5])
    # define transform
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataTransform = transforms.Compose([
        transforms.Scale((Height,Width)),
        transforms.ToTensor(),
        normalize
    ])
    airData = datasets.ImageFolder(targetPath, transform=dataTransform)
    satData = datasets.ImageFolder(inputPath, transform=dataTransform)
    
    assert (len(airData) == len(satData))
    lenData = len(satData)
    
    data = torch.FloatTensor(lenData, 2, 3, Height, Width).zero_()
    
    dataSort = range(lenData)
    if(shuffleData == True):
        dataSort = np.random.permutation(lenData)
    
    i = 0
    for j in dataSort:
        data[i][0] = satData[j][0]
        data[i][1] = airData[j][0]
        i = i + 1
    
    return data

def imageShuffler(data):
    dataSort = np.random.permutation(len(data))
    
    newData = torch.FloatTensor(len(data), 2, 3, len(data[0][0][0]), len(data[0][0][0][0])).zero_()
    
    i = 0
    for j in dataSort:
        newData[i][0] = data[j][0]
        newData[i][1] = data[j][1]
        i = i + 1
    
    return newData