import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from models.Unet import UnetGenerator
from models.PixelDiscriminator import PixelDiscriminator
from GANLoss import GANLoss
from anotherDatasetLoader import imageLoader
import os
from PIL import Image
from tools import * 
#%matplotlib inline

"""Hyperparameters"""
img_dim_size = 32 # Starting image height and width
doublingEpochs = [101, 201, 301, 401] #epochs at which the dimensions of images trained on are doubled
adam_learning_rate = 0.0003
batch_size = 4
image_save_step = 5
max_epochs = 500
loader = imageLoader("input/", "target/", img_dim_size, img_dim_size,shuffleData=False)
pool_size = 12 #the size of image buffer that stores previously generated images
fake_AB_pool = ImagePool(pool_size)

g = UnetGenerator(3,3,5)
d = PixelDiscriminator(6)
g.cuda()
d.cuda()
g.apply(weights_init_xavier)
d.apply(weights_init_xavier)

optimizer_G = optim.Adam(g.parameters(),
                                lr=adam_learning_rate)
optimizer_D = optim.Adam(d.parameters(),
                                lr=adam_learning_rate)
criterion = GANLoss()
L1_loss = nn.MSELoss()
lam = 0.5

num_its = int((loader.size()[0] // batch_size)/2)
i = num_its-1
X_valid = [loader[i*batch_size: (i+1)*batch_size,0], loader[(i+1)*batch_size: (i+2)*batch_size,0], loader[(i+2)*batch_size: (i+3)*batch_size,0]]
Y_valid = [loader[i*batch_size: (i+1)*batch_size,1], loader[(i+1)*batch_size: (i+2)*batch_size,1], loader[(i+2)*batch_size: (i+3)*batch_size,1]]
ACC_REAL = []
ACC_FAKE = []
G_LOSS = []
D_LOSS = []
D_LOSS_REAL = []
D_LOSS_FAKE = []
G_LOSS_L1 = []
G_LOSS_GAN = []
for epoch in range(max_epochs):

  if(epoch in doublingEpochs):
    img_dim_size = img_dim_size*2
    loader = imageLoader("input/", "target/", img_dim_size, img_dim_size,shuffleData=False)
    X_valid = [loader[(num_its-1)*batch_size: (num_its)*batch_size,0], loader[(num_its)*batch_size: (num_its+1)*batch_size,0], loader[(num_its+1)*batch_size: (num_its+2)*batch_size,0]]
    Y_valid = [loader[(num_its-1)*batch_size: (num_its)*batch_size,1], loader[(num_its)*batch_size: (num_its+1)*batch_size,1], loader[(num_its+1)*batch_size: (num_its+2)*batch_size,1]]
    fake_AB_pool = ImagePool(pool_size)

  loss_D_avg = 0
  loss_G_avg = 0
  loss_D_R = 0
  loss_D_F = 0
  loss_G_L1s = 0
  loss_G_GANs = 0
  pred_real_correct = 0
  pred_real_total = 0
  pred_fake_correct = 0
  pred_fake_total = 0
  for i in range(num_its-1):
    X_batch = loader[i*batch_size: (i+1)*batch_size,0]
    Y_batch = loader[i*batch_size: (i+1)*batch_size,1]
    optimizer_D.zero_grad()
    optimizer_G.zero_grad()
    """Forward pass"""
    inputs = Variable(X_batch.cuda())
    real_images = Variable(Y_batch.cuda())
    fake_images = g(inputs)

    """Discriminator"""
    # Fake images
    fake_AB = torch.cat((inputs, fake_images), 1)
    fake_AB_d = fake_AB_pool.query(fake_AB.data)
    pred_fake = d(fake_AB_d.detach())
    loss_D_fake = criterion(pred_fake, False)
    loss_D_Value = loss_D_fake.data[0]
    (loss_D_fake/2).backward()

    # Real images
    real_AB = torch.cat((inputs, real_images),1 )
    pred_real = d(real_AB)
    loss_D_real = criterion(pred_real, True)
    loss_D_Value = loss_D_Value + loss_D_real.data[0]
    (loss_D_real/2).backward()

    optimizer_D.step()    
    """Generator"""
    
    pred_fake = d(fake_AB)
    loss_G_GAN = criterion(pred_fake, True)
    loss_G_L1 = L1_loss(fake_images, real_images)
    loss_G = loss_G_GAN + loss_G_L1*lam
    

    loss_G.backward()


    optimizer_G.step()

    
    """Stats gathering"""
    pred_real_numpy = pred_real.data[0].cpu().numpy() >= 0.5
    pred_fake_numpy = pred_fake.data[0].cpu().numpy() >= 0.5    
    loss_D_avg += loss_D_Value
    loss_G_avg += loss_G.data[0]
    pred_real_correct += pred_real_numpy.sum()
    pred_real_total += pred_real_numpy.size
    pred_fake_total += pred_fake_numpy.size
    pred_fake_correct += pred_fake_numpy.size - pred_fake_numpy.sum()
    
    loss_D_R += loss_D_real.data[0]
    loss_D_F += loss_D_fake.data[0]
    loss_G_L1s += loss_G_L1.data[0]
    loss_G_GANs += loss_G_GAN.data[0]

  acc_real = pred_real_correct / pred_real_total
  acc_fake = pred_fake_correct / pred_fake_total
  ACC_REAL.append(acc_real)
  ACC_FAKE.append(acc_fake)
  G_LOSS.append(loss_G_avg / num_its)
  D_LOSS.append(loss_D_avg / num_its)
  G_LOSS_L1.append(loss_G_L1s / num_its)
  G_LOSS_GAN.append(loss_G_GANs / num_its)
  D_LOSS_REAL.append(loss_D_R / num_its)
  D_LOSS_FAKE.append(loss_D_F / num_its)
    
  print(loss_G_avg/num_its, loss_D_avg/num_its, pred_fake_correct/pred_fake_total, pred_real_correct/ pred_real_total)
  print("loss D real:",loss_D_R / num_its,"loss D f:", loss_D_F / num_its, "L1:", loss_G_L1s / num_its,"G Gan:", loss_G_GANs / num_its)
  if epoch % image_save_step == 0:
        for valBatch in range(3):
            inputs = Variable(X_valid[valBatch].cuda())
            real_images = Variable(Y_valid[valBatch].cuda())
            fake_images = g(inputs)
            for i in range(4):
                printNum = i+(4*valBatch)
                name = "{}_{}".format(epoch, printNum)
                save_result_images(inputs, real_images, fake_images, name, i = i, path="results")        
        print("Image saved.")
        plt.figure(figsize=(20,10))
        plt.plot(D_LOSS_FAKE, label="D Fake")
        plt.plot(D_LOSS_REAL, label="D Real")
        plt.plot(G_LOSS_GAN, label="Generator GAN")
        plt.plot(G_LOSS_L1,"--", label='L1 loss')
        plt.legend()
        plt.savefig("results/test"+ str(epoch))
        plt.show()