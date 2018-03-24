from PIL import Image
#import Image
import os

def crop(Path1,Path2, input, height, width, imNum):
    im = Image.open(input)
    imgwidth, imgheight = im.size
    imageCount = 1
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            if(imageCount%2==1):
                a.save(os.path.join(Path1,"%s.png" % imNum))
            else:
                a.save(os.path.join(Path2,"%s.png" % imNum))
            imageCount += 1

imageCount = 0
for i in range(1097):
    if (i == 0):
        continue
    imageCount = imageCount + 1
    crop("./input/data","./target/data", "./maps/train/" + str(i) + ".jpg", 600, 600, imageCount)
for i in range(1099):
    if (i == 0):
        continue
    imageCount = imageCount + 1
    crop("./input/data","./target/data", "./maps/val/" + str(i) + ".jpg", 600, 600, imageCount)