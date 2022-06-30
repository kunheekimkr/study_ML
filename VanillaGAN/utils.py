import io
from PIL import Image, ImageDraw, ImageOps
import imageio as io


def saveresults(epochs):
    path= "./result/generated_img_epoch:"
    images=[]

    for epoch in range(1, epochs+1):
        file = Image.open(path+str(epoch)+".png")
    
        img = ImageOps.expand(file, border=10, fill='white')

        draw_object = ImageDraw.Draw(img)
        draw_object.text((130, 289), "Epoch: {}".format(epoch), fill="black")
    
        img.save(path+str(epoch)+".png")
        images.append(img)

    io.mimsave('./result/result.gif', images, duration =0.3)
