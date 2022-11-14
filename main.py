import os
import sys
import numpy as np
import torch
import gym
import argparse
from pyvirtualdisplay import Display

from training import train
from demonstrations import record_demonstrations

def evaluate(args, trained_network_file):
    """
    """
    infer_action = torch.load(trained_network_file)
    infer_action.eval()
    if args.cluster:
        display = Display(visible=0, size=(800,600))
        display.start()

    # Here I change the CarRacing-v0 to CarRacing-v2, otherwise I can't run it in my local machine...
    # So is the code below from line 38-40.
    env = gym.make('CarRacing-v2')  
    
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    infer_action = infer_action.to(device)

    avg_reward = 0
    for episode in range(5):
        observation = env.reset()

        reward_per_episode = 0
        for t in range(500):
            
            # print(np.ascontiguousarray(observation[None]).shape)  
            env.render()
            if len(np.ascontiguousarray(observation[0]).shape) == 3:
                action_scores = infer_action(torch.Tensor(
                    np.ascontiguousarray(observation[0])).to(device))  
            else:
                action_scores = infer_action(torch.Tensor(
                    np.ascontiguousarray(observation[None])).to(device))    # shape of (1, 96, 96, 3)
            # print(infer_action.action_classes)

            steer, gas, brake = infer_action.scores_to_action(action_scores)
            observation, reward, done, info, done  = env.step([steer, gas, brake])   # 5 output in new env
            reward_per_episode += reward
            
            # print(steer, gas, brake)

        print('episode %d \t reward %f' % (episode, reward_per_episode))
        avg_reward += reward_per_episode

    print('average reward: ', avg_reward/5)
    env.close()
    if args.cluster:
	    display.stop()

def calculate_score_for_leaderboard(args, trained_network_file):
    """
    Evaluate the performance of the network. This is the function to be used for
    the final ranking on the course-wide leader-board, only with a different set
    of seeds. Better not change it.
    """
    infer_action = torch.load(trained_network_file)
    infer_action.eval()
    if args.cluster:
        display = Display(visible=0, size=(800,600))
        display.start()
    env = gym.make('CarRacing-v0')

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seeds = [22597174, 68545857, 75568192, 91140053, 86018367,      # change seed to evaluate in new env
            49636746, 66759182, 91294619, 84274995, 31531469]

    total_reward = 0

    for episode in range(8):
        env.seed(seeds[episode])
        observation = env.reset()

        reward_per_episode = 0
        for t in range(600):
            env.render()
            action_scores = infer_action(torch.Tensor(
                np.ascontiguousarray(observation[None])).to(device))

            steer, gas, brake = infer_action.scores_to_action(action_scores)
            observation, reward, done, info = env.step([steer, gas, brake])
            reward_per_episode += reward

        print('episode %d \t reward %f' % (episode, reward_per_episode))
        total_reward += np.clip(reward_per_episode, 0, np.infty)

    print('---------------------------')
    print(' total score: %f' % (total_reward / 10))
    print('---------------------------')
    env.close()
    if args.cluster:
	    display.stop()

if __name__ == "__main__":
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument(
        "--train",
        action="store_true",
    )
    main_parser.add_argument(
        "--test",
        action="store_true",
    )
    main_parser.add_argument(
        "--score",
        action="store_true",
    )
    main_parser.add_argument(
        "--collect",
        action="store_true",
    )
    main_parser.add_argument(
        "--agent_load_path",
        type=str,
        default="model/train.t7",
        # default="/home/stud100/ex_01_imitation_learning/model/train.t7",
        # default="/home/ubuntu/SDC_exercises/ex_01_imitation_learning/model/train.t7",
        help="Path to the .t7 file of the trained agent."
    )
    main_parser.add_argument(
        "--agent_save_path",
        type=str,
        default="model/train.t7",
        help="Save path of the trained model."
    )
    main_parser.add_argument(
        "--training_data_path",
        type=str,
        # default="/mnt/beegfs/home/stud100/ex_01_imitation_learning/data",
        default="data",
        help="Path of the training data (save and load)."
    )
    main_parser.add_argument(
        "--cluster",
        action="store_true",
	)


    args = main_parser.parse_args()

    if args.collect:
        print('Collect: You can collect training data now.')
        record_demonstrations(args.training_data_path)
    elif args.train:
        print('Train: Training your network with the collected data.')
        train(args.training_data_path, args.agent_save_path)
    elif args.test:
        print('Test: Your trained model will be tested now.')
        evaluate(args, args.agent_load_path)
    elif args.score:
        calculate_score_for_leaderboard(args, args.agent_load_path)

