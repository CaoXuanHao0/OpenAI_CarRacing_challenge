import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F

from demonstrations import load_demonstrations
from network import ClassificationNetwork
from network2 import RegreessionNetwork

def train(data_folder, trained_network_file):
    """
    Function for training the network.
    """
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # infer_action = ClassificationNetwork().to(device)
    infer_action = RegreessionNetwork().to(device)
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=0.0001)     # Tried lr=0.1, but the loss wouldn't go down... 1e-4 is a good start.
    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.MSELoss()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    observations, actions = load_demonstrations(data_folder)

    batches = data_aug(observations, actions, infer_action)
    batches = [(batch[0],infer_action.actions_to_classes(batch[1])) for batch in batches]

    # observations = [torch.Tensor(observation)/255 for observation in observations]
    # observations = torch.tensor(observations)
    # actions = [torch.Tensor(action) for action in actions]
    # if len(infer_action.actions_to_classes(actions))!=len(observations):
    #     print('Some actions are not in the action_classes!')
    # batches = [batch for batch in zip(observations,
    #                                   infer_action.actions_to_classes(actions))]

    nr_epochs = 10
    batch_size = 128
    start_time = time.time()
    total_loss2=0

    print('start training')
    for epoch in range(nr_epochs):
        random.shuffle(batches)

        total_loss = 0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(batches):
            batch_in.append(batch[0].to(device))
            if type(batch[1])==list:  # ???why??? there is a list???
                batch_gt.append(batch[1][0].to(device))  # get rid of the list
            else: batch_gt.append(batch[1].to(device))

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                # batch_in = torch.permute(torch.reshape(torch.cat(batch_in, dim=0),
                #                          (-1, 96, 96, 3)),(0,3,1,2))
                batch_in = torch.stack(batch_in,dim=0)  # (batch_size,3,96,96)
                # batch_gt = torch.reshape(torch.cat(batch_gt, dim=0),
                #                          (-1,))
                batch_gt = torch.stack(batch_gt,dim=0)

                batch_out = infer_action(batch_in)
                loss = loss_function(batch_out, batch_gt)
   
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss

                batch_in = []
                batch_gt = []

        lr_scheduler.step()  # Learning rate scheduling
        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))
        
        # if total_loss<20:
            # break
        # if total_loss2-total_loss<3 and total_loss<20:
        #     break
        # total_loss2=total_loss

    torch.save(infer_action, trained_network_file)
    print('save model to: ', trained_network_file)

def data_aug(observations, actions, infer_action):
    
    batches = [(torch.tensor(batch[0]),torch.tensor(batch[1])) for batch in zip(observations, actions)]

    transforms = torch.nn.Sequential(
        T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 3)),
        # T.ColorJitter(contrast = 0.5),
        T.ColorJitter(brightness=(0.2,0.35)),
        # T.Grayscale(), 
        # T.RandomInvert(),
        T.RandomSolarize(threshold=192.0),
        T.RandomAdjustSharpness(sharpness_factor=2),
        # T.RandomPosterize(bits=2),
        # crop_img(3,3,96,96)
    )   

    # data aug -- flip image and action
    for act, obs in zip(actions,observations):
        if act[0] != 0:
            # 84 here not helpful
            new_obs = torch.tensor(obs).clone()
            new_obs[:,:84,:] = torchvision.transforms.RandomHorizontalFlip(p=1)(torch.tensor(obs[:,:84,:]))
            new_act = torch.tensor(act)
            new_act[0] = -act[0]
            batches.append((new_obs, new_act))   

            if act[0] != 0 and act[2]!=0:
                # not enough data of [-1. ,  0. ,  0.8] and [1. ,  0. ,  0.8], so we need to augment them more
                for i in range(len(transforms)):
                    new_obs2 = torch.tensor(new_obs).clone()
                    new_obs2[:,:84,:] = transforms[i](torch.tensor(new_obs[:,:84,:]))
                    batches.append((new_obs2,torch.tensor(act)))

                    obs2 = torch.tensor(obs).clone()
                    obs2[:,:84,:] = transforms[i](torch.tensor(obs[:,:84,:]))                   
                    batches.append((obs2,torch.tensor(new_act)))

        elif act[0]==0 and act[-1]==0.8:  # augment [ 0. ,  0. ,  0.8]
            for i in range(len(transforms)-2):
                    new_obs = torch.tensor(obs).clone()
                    new_obs[:,:84,:] = transforms[i](torch.tensor(obs[:,:84,:]))
                    # new_obs = transforms[i](torch.tensor(obs))
                    batches.append((new_obs,torch.tensor(act)))

    # data augmentation -- simple transform images                               
    for i in range(len(transforms)):
        # shuffle the data
        np.random.seed(i)
        index = np.random.choice(range(observations.shape[0]),observations.shape[0],False)
        observations = observations[index]
        actions = actions[index]
        # ranodmly aug data
        for batch in zip(observations[:400],actions[:400]):
            # only transform part of image: batch[0][:,:84,:], because in NN.extract_sensor_values, we need the other part unchange to extract some infomation
            new_obs = torch.tensor(batch[0]).clone()
            new_obs[:,:84,:] = transforms[i](torch.tensor(batch[0][:,:84,:]))
            # new_obs = transforms[i](torch.tensor(batch[0]))
            batches.append((new_obs,torch.tensor(batch[1])))

    return batches

# def data_aug(observations, actions, infer_action):
    
#     batches = [(torch.tensor(batch[0]),torch.tensor(batch[1])) for batch in zip(observations, actions)]

#     transforms = torch.nn.Sequential(
#         T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 3)),
#         # T.ColorJitter(contrast = 0.5),
#         T.ColorJitter(brightness=(0.2,0.35)),
#         # T.Grayscale(), 
#         # T.RandomInvert(),
#         T.RandomSolarize(threshold=192.0),
#         T.RandomAdjustSharpness(sharpness_factor=2),
#         # T.RandomPosterize(bits=2),
#         # crop_img(3,3,96,96)
#     )   

    

#     # data aug -- flip image and action
#     for act, obs in zip(actions,observations):
#         if act[0] != 0:
#             new_obs = torch.tensor(obs).clone()
#             new_obs[:,:84,:] = torchvision.transforms.RandomHorizontalFlip(p=1)(torch.tensor(obs[:,:84,:]))
#             new_act = torch.tensor(act)
#             new_act[0] = -act[0]
#             batches.append((new_obs, new_act))   

#             if act[0] != 0 and act[2]!=0:
#                 # not enough data of [-1. ,  0. ,  0.8] and [1. ,  0. ,  0.8], so we need to augment them more
#                 for i in range(len(transforms)):
#                     new_obs2 = torch.tensor(new_obs).clone()
#                     new_obs2[:,:84,:] = transforms[i](torch.tensor(new_obs[:,:84,:]))
#                     batches.append((new_obs2,torch.tensor(act)))

#                     obs2 = torch.tensor(obs).clone()
#                     obs2[:,:84,:] = transforms[i](torch.tensor(obs[:,:84,:]))                   
#                     batches.append((obs2,torch.tensor(new_act)))

#             else:
#                 # randomly apply one transform
#                 index = np.random.randint(0,len(transforms)-1)
#                 new_obs2 = torch.tensor(new_obs).clone()
#                 new_obs2[:,:84,:] = transforms[index](torch.tensor(new_obs[:,:84,:]))
#                 batches.append((new_obs2,torch.tensor(new_act)))
                
#         elif act[0]==0 and act[-1]==0.8:  # augment [ 0. ,  0. ,  0.8]
#             for i in range(len(transforms)-2):
#                     new_obs = torch.tensor(obs).clone()
#                     new_obs[:,:84,:] = transforms[i](torch.tensor(obs[:,:84,:]))
#                     batches.append((new_obs,torch.tensor(act)))
                    
#     # for act, obs in zip(actions,observations):
#     #     if act[0] != 0:
#     #         # if act[0] != 0 and act[2]!=0:
#     #             # not enough data of [-1. ,  0. ,  0.8] and [1. ,  0. ,  0.8], so we need to augment them more
#     #             for i in range(len(transforms)):
#     #                 new_obs = torch.tensor(obs).clone()
#     #                 new_obs[:,:84,:] = transforms[i](torch.tensor(obs[:,:84,:]))
#     #                 batches.append((new_obs,torch.tensor(act)))
#     #         # else:
#     #         #     # randomly apply one transform
#     #         #     index = np.random.randint(0,len(transforms)-2)
#     #         #     new_obs = torch.tensor(obs).clone()
#     #         #     new_obs[:,:84,:] = transforms[index](torch.tensor(obs[:,:84,:]))
#     #         #     batches.append((new_obs, torch.tensor(act))) 
#     #     elif act[0]==0 and act[-1]==0.8:  # augment [ 0. ,  0. ,  0.8]
#     #         for i in range(len(transforms)-2):
#     #                 new_obs = torch.tensor(obs).clone()
#     #                 new_obs[:,:84,:] = transforms[i](torch.tensor(obs[:,:84,:]))
#     #                 batches.append((new_obs,torch.tensor(act)))

#     # data augmentation -- simple transform images                               
#     for i in range(len(transforms)):
#         # shuffle the data
#         np.random.seed(i)
#         index = np.random.choice(range(observations.shape[0]),observations.shape[0],False)
#         observations = observations[index]
#         actions = actions[index]
#         # ranodmly aug data
#         for batch in zip(observations[:300],actions[:300]):
#             # only transform part of image: batch[0][:,:84,:], because in NN.extract_sensor_values, we need the other part unchange to extract some infomation
#             new_obs = torch.tensor(batch[0]).clone()
#             new_obs[:,:84,:] = transforms[i](torch.tensor(batch[0][:,:84,:]))
#             batches.append((new_obs,torch.tensor(batch[1])))

#     return batches
