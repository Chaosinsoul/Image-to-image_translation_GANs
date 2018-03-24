from PIL import Image
import os
import numpy as np
import torch.nn as nn
import random
import torch
from torch.autograd import Variable

# im: A pytorch variable with a single image
def reshape_image(im, denormalize=True):
    R, G, B = im.data.cpu().numpy()
    im = np.concatenate((R[:,:,np.newaxis], G[:,:, np.newaxis], B[:,:,np.newaxis]), 2)
    if denormalize:
        im += 1
        im /= 2
    
    return im

# im: single numpy array with image in correct format
# name: ...
# path: relative path to save 
def save_image(im, name,path):
    im = Image.fromarray((im*255).astype('uint8'))
    path = os.path.join(path, name) + ".png"
    im.save(path)
# inputs, real_images, fake_images: pytorch variable with images (Straight from network)
# name: .....
# i: index of image to get out 
def save_result_images(inputs, real_images, fake_images, name, path="results", i=0):
    inputs = reshape_image(inputs[i])
    real_images = reshape_image(real_images[i])
    fake_images = reshape_image(fake_images[i], True)
    im = np.concatenate((inputs, real_images, fake_images), 1)
    save_image(im, name, path)
    
    
def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)
        

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return Variable(images)
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images
    
    