from unittest import result
import torch

from torchvision.utils import save_image
from dataloader import Dataset, DataLoader
from loss import Loss
from model import Generator, Discriminator
from utils import saveresults

import numpy as np
from matplotlib.pyplot import imsave

dataset = Dataset()
data_loader = DataLoader(dataset)

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("mps") #Apple Silicon Support

G = Generator().to(device)
D = Discriminator().to(device)

G_optim = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optim = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion= Loss()

#Parameters
epochs=50
batch_size = data_loader.dataloader.batch_size
step =0
k = 10 # Train Discirminator for k steps, and train Genrator 1 step.

D_labels= torch.ones(batch_size,1).to(device)
D_fakes= torch.zeros(batch_size,1).to(device)

for epoch in range(epochs):
    for idx, (images,_) in enumerate(data_loader.dataloader):
        # Train Discriminator
        x = images.view(images.size(0), -1).to(device)
        x_outputs = D(x)
        D_x_loss = criterion(x_outputs, D_labels)


        z = torch.randn(batch_size, 100).to(device)
        z_outputs = D(G(z))
        D_z_loss = criterion(z_outputs, D_fakes)
        
        D_loss = D_x_loss + D_z_loss

        D.zero_grad()
        D_loss.backward()
        D_optim.step()

        if step % k == 0 :
            # Train Generator
            z = torch.randn(batch_size, 100).to(device)
            z_outputs = D(G(z))
            G_loss = criterion(z_outputs, D_labels)

            G.zero_grad()
            G_loss.backward()
            G_optim.step()
        if step % 500 == 0:
            print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}'.format(epoch, epochs, step, D_loss.item(), G_loss.item()))  

        step += 1
    
    G.eval()
    z = torch.randn(100, 100).to(device)
    temp = G(z).view(100,28,28)
    result = temp.cpu().data.numpy()
    fake_img = np.zeros([280,280]) # 10x10 grid of images
    for i in range(10):
        fake_img[i*28:(i+1)*28] = np.concatenate ([j for j in result[i*10 : (i+1)*10]],axis=-1) 
    imsave('result/generated_img_epoch:{}.png'.format(epoch+1), fake_img, cmap='gray')
    G.train()


saveresults(epochs)
