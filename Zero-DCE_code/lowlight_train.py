import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader_1
import model
import Myloss
import numpy as np
from torchvision import transforms
import tensorflow as tf

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)





def train(config):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	if config.channel=="RGB":
		DCE_net = model.enhance_net_nopool_3().cuda()
	elif config.channel=="HSV":
		DCE_net = model.enhance_net_nopool_1_3().cuda()
	elif config.channel=="HLS":
		DCE_net = model.enhance_net_nopool_1_2().cuda()
	elif config.channel=="YCbCr" or  config.channel=="YUV" or  config.channel=="LAB" or  config.channel=="Luv":
		DCE_net = model.enhance_net_nopool_1_1().cuda()
	
	DCE_net.apply(weights_init)
	"""
	if config.load_pretrain == True:
	    DCE_net.load_state_dict(torch.load(config.pretrain_dir))
	"""
	
	train_dataset = dataloader_1.lowlight_loader(config.lowlight_images_path, config.channel)	
	print(len(train_dataset))	
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)



	L_color = Myloss.L_color()
	L_spa = Myloss.L_spa()

	L_exp = Myloss.L_exp(16,0.6)
	L_TV = Myloss.L_TV()


	optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	data_ter = len(train_loader)
	DCE_net.train()

	for epoch in range(config.num_epochs):
		train_summary_writer = tf.summary.create_file_writer("log/train_loss_"+config.channel)
		Loss_TV = []
		loss_spa = []
		loss_col = []
		loss_exp = []
		loss_tot = []
		for iteration, img_lowlight in enumerate(train_loader):
			img_lowlight = img_lowlight.cuda()

			enhanced_image_1,enhanced_image,A  = DCE_net(img_lowlight)

			Loss_TV.append(L_TV(A))
			print("Loss_TV:",Loss_TV[iteration].item())
			
			loss_spa.append(torch.mean(L_spa(enhanced_image, img_lowlight)))
			print("loss_spa:",loss_spa[iteration].item())

			loss_col.append(torch.mean(L_color(enhanced_image)))
			print("loss_col:",loss_col[iteration].item())

			loss_exp.append(torch.mean(L_exp(enhanced_image)))
			print("loss_exp:",loss_exp[iteration].item())
			
			
			# best_loss
			loss =  200*Loss_TV[iteration] + loss_spa[iteration] + 5*loss_col[iteration] + 10*loss_exp[iteration]
			loss_tot.append(loss)
			#
			
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(DCE_net.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				print("Loss at iteration", iteration+1, ":", loss.item())
			if ((iteration+1) % config.snapshot_iter) == 0:
				
				torch.save(DCE_net.state_dict(), config.snapshots_folder + config.channel  + '.pth') #+ "Epoch" + str(epoch)
		with train_summary_writer.as_default():
			tf.summary.scalar('Illumination Smoothness Loss', sum(Loss_TV).item()/data_ter, step=epoch)
			tf.summary.scalar('Spatial Loss', sum(loss_spa).item()/data_ter, step=epoch)
			tf.summary.scalar('Color Loss', sum(loss_col).item()/data_ter, step=epoch)
			tf.summary.scalar('Exposure Control Loss', sum(loss_exp).item()/data_ter, step=epoch)
			tf.summary.scalar('Total Loss', sum(loss_tot).item()/data_ter, step=epoch)
	train_summary_writer.close()




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
	parser.add_argument("--channel", type=str, default="RGB") # HSV, YCbCr, LAB, Luv, HLS, YUV
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch99.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)








	
