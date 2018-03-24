
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#import matplotlib.pyplot as plt
from torch.autograd import Variable
from Patchmodels .Unet import UnetGenerator
# from models.PixelDiscriminator import PixelDiscriminator
from Patchmodels.PatchDiscriminator import *
from GANLoss import GANLoss
from anotherDatasetLoader import imageLoader
import os
from PIL import Image
from tools import * 
import torchvision
from statistics import *


# In[3]:


"""Hyperparameters"""
learning_rate = 0.0003
batch_size = 16
max_epochs = 1000
imsize = 32
lam = 0.1
pool_size = 50 #the size of image buffer that stores previously generated images
resize_image_epochs = 200 # Number of epochs until double increase image size
resize_image_scale = 2
loader_train = imageLoader("data/gta/input/train", "data/gta/target1/train", imsize, imsize,shuffleData=True)
loader_val = imageLoader("data/gta/input/val", "data/gta/target1/val", imsize, imsize)
fake_AB_pool = ImagePool(pool_size)


# In[4]:


g = UnetGenerator(3,3,5)
d = NLayerDiscriminator(6)
g.cuda()
d.cuda()
g.apply(weights_init_xavier)
d.apply(weights_init_xavier)
stats = Statistics()


# In[5]:


optimizer_G = optim.Adam(g.parameters(),
                                lr=learning_rate)
optimizer_D = optim.Adam(d.parameters(),
                                lr=learning_rate)
criterion = GANLoss()
L1_loss = nn.MSELoss()


# In[6]:


num_its = loader_train.size()[0] // batch_size
i = num_its-1
X_valid = loader_val[:,0]
Y_valid = loader_val[:,1]
image_save_step = 10

for epoch in range(max_epochs):


  for i in range(num_its):
    X_batch = loader_train[i*batch_size: (i+1)*batch_size,0]
    Y_batch = loader_train[i*batch_size: (i+1)*batch_size,1]
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

    # Real images
    real_AB = torch.cat((inputs, real_images),1 )
    pred_real = d(real_AB)
    loss_D_real = criterion(pred_real, True)

    loss_D =  (loss_D_fake + loss_D_real)*0.5
    loss_D.backward()
    optimizer_D.step()    
    """Generator"""
    
    pred_fake = d(fake_AB)
    loss_G_GAN = criterion(pred_fake, True)
    loss_G_L1 = L1_loss(fake_images, real_images)
    loss_G = loss_G_GAN + loss_G_L1*lam
    

    loss_G.backward()


    optimizer_G.step()
    """Stats gathering"""
    stats.update_accuracies(pred_real, pred_fake)
    stats.update_loss(loss_D_real, loss_D_fake, loss_G_L1, loss_G_GAN)
  stats.append_results(num_its)
  inputs = Variable(X_valid.cuda())
  real_images = Variable(Y_valid.cuda())
  fake_images = g(inputs)
  test_l1_loss = L1_loss(fake_images.detach(), real_images)    
  stats.update_test_loss(test_l1_loss)
  stats.print_results(epoch)
  if epoch % image_save_step == 0:
    name = "{}".format(epoch)
    save_result_image_in_one(inputs, real_images, fake_images, name, path="results2")        
    print("Image saved.")
    stats.save_plot(epoch,"results2")
  if (epoch+1) % resize_image_epochs == 0:
    imsize *= 2
    loader_train = imageLoader("data/gta/input/train", "data/gta/target1/train", imsize, imsize,shuffleData=True)
    loader_val = imageLoader("data/gta/input/val", "data/gta/target1/val", imsize, imsize)
    print("Increasing imsize. Current size: {}x{}".format(imsize, imsize))
    X_valid = loader_val[:,0]
    Y_valid = loader_val[:,1]
    fake_AB_pool = ImagePool(pool_size) # Reset image pool history


        


# In[ ]:


plt.imshow(im)
plt.show()
print(im.mean(axis=(0,1))*255)
print(im.var(axis=(0,1))*255)


# In[ ]:


plt.imshow(im)
print(im.mean(axis=(0,1))*255)
print(im.var(axis=(0,1))*255)
plt.show()


# In[ ]:


from PIL import Image



# In[ ]:


im = (im*255).astype('uint8')


# In[ ]:


im = Image.fromarray(im)


# In[ ]:


im


# In[ ]:


im.rotate(90)


# In[ ]:


im = reshape_image(Variable(loader_train[28,1]))
plt.imshow(im)
print(im.mean(axis=(0,1))*255)
print((im*255).var(axis=(0,1)).mean())
plt.show()


# In[ ]:


plt.imshow(im)
plt.show()
print(im.mean(axis=(0,1))*255)


# In[ ]:


plt.imshow(loader[4,0,0].numpy(), cmap="gray")


# In[ ]:


loader.size()


# In[ ]:


plt.figure(figsize=(20,10))
plt.plot(D_LOSS_FAKE, label="D Fake")
plt.plot(D_LOSS_REAL, label="D Real")
plt.plot(G_LOSS_GAN, label="Generator GAN")
plt.plot(G_LOSS_L1,"--", label='L1 loss')
plt.legend()
plt.savefig("results2/test")


# In[ ]:


plt.figure(figsize=(16,10))
plt.plot(ACC_REAL, label="Accuray real")
plt.plot(ACC_FAKE, label="Accuracy fake")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

