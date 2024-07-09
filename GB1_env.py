import argparse
import torch
import gym
import itertools
import numpy as np
import copy
import random
import time
import csv
from contextlib import contextmanager
import pandas as pd
import sys, os
from transformers import AutoTokenizer,AutoModel,EsmForMaskedLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import logger
from stable_baselines3.common.env_checker import check_env
cwd = os.path.dirname(os.path.abspath(__file__))

collected_seqs_set = set()
# path_96 or path_192 or path_288
path_96 = './data/96/GB1_96.csv'
AMINO_ACIDS = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
class GB1Env(gym.Env):
    def __init__(self,
                 action_space: gym.spaces,
                 observation_space: gym.spaces,
                 args: dict,
                 max_len: int = 58,
                 ):
        super(GB1Env, self).__init__()
        self.action_space = action_space
        self.observation_space = observation_space

        self.reward = float("-inf")
        self.reward_list = []
        self.max_step = args.max_step
        self.score_stop_criteria = args.score_stop_criteria
        self.k = [0,6]
        self.len_step = 0
        self.max_len = max_len

        # datas = pd.read_csv('GB1-384.csv', names=['Variants', 'HD', 'Count_input', 'Count_selected', 'Fitness'], header=0)
        datas = pd.read_csv(path_96, names=['AACombo', 'Fitness'], header=0)
        self.gb1 = []
        self.gb1_fitness = []
        self.gb1_protein = []
        self.num2seq = {}
        for i in range(len(datas)):
            # print(datas["Variants"][i][0:3])
            protein = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNG___EWTYDDATKTFT_TE"
            protein = protein.replace("___", datas["AACombo"][i][0:3])
            protein = protein.replace("_", datas["AACombo"][i][3])
            tokens = tokenizer(protein, return_tensors="pt").to(device)
            self.gb1_protein.append(datas["AACombo"][i])
            self.gb1.append(tokens['input_ids'].squeeze(0).cpu().numpy())
            self.gb1_fitness.append(float(datas['Fitness'][i]))

        print("finish build")

    def init_seq(self):
        index = random.randint(0, len(self.gb1)-1)
        # seq = random.choice(self.gb1)
        self.initial_seq = self.gb1[index]
        self.score_stop_criteria = self.gb1_fitness[index]
        collected_seqs_set.add(self.gb1_protein[index])
        return self.initial_seq

    def reset(self):

        self.state = self.init_seq()

        self.len_step = 0

        self.reward_list = []

        self.k = [0,6]

        return self.state

    def check_terminal(self, score, stirng, have):
        if have == True:
            collected_seqs_set.add(stirng)
        if score >= self.score_stop_criteria or self.len_step >= self.max_step:
            return True
        else:
            return False

    def _get_reward(self,seq):
        seq = torch.from_numpy(seq).unsqueeze(0).to(device)
        protein = [tokenizer.decode(r) for r in seq][0][6:117][::2]
        string = protein[38] + protein[39] + protein[40] + protein[53]
        flag = False
        for i in string:
            if i not in AMINO_ACIDS:
                flag = True
        have = False
        if string in fitness.keys():
            score_truth = fitness[string]
            have = True
        elif flag == True:
            score_truth = -100
        else:
            score_truth = -1

        self.reward = score_truth

        terminal = self.check_terminal(self.reward, string, have)

        return self.reward, terminal, score_truth

    def _edit_sequence(self, seq, actions):
        protein = seq
        position = actions[0]

        protein[position] = actions[1]

        return protein

    def step(self, actions: torch.Tensor):
        ### take action
        new_seqs = self._edit_sequence(self.state, actions)

        self.len_step += 1
        term_reward, terminal, score_truth = self._get_reward(new_seqs)
        self.reward_list.append(term_reward)
        if len(self.reward_list)>=2 and self.reward_list[-1] > self.reward_list[-2]:
            self.k[1] = max(self.k[1] - 1,6)
            self.k[0] = max(self.k[0] - 1,0)
        # Check if the last value is increasing compared to the previous one
        if len(self.reward_list)>=3 and self.reward_list[-2] >= self.reward_list[-1] and self.reward_list[-3] >= self.reward_list[-2]:
            self.k[0] = min(self.k[0] + 1, 14)
            self.k[1] = min(self.k[1] + 1, 20)
        info = {}
        info['terminal'] = str(terminal)
        info['action'] = ",".join([str(actions[i]) for i in range(2)])
        info['old_seq'] = tokenizer.decode(self.state[0])
        info['new_seq'] = tokenizer.decode(new_seqs[0])
        info['init_seq'] = self.initial_seq if self.initial_seq is not None else "None"
        info['rewards'] = float(term_reward)
        info['score_truth'] = float(score_truth)
        info['k'] = self.k
        logger.record("state/reward", term_reward)
        logger.record("state/fitness", score_truth)

        self.state = new_seqs

        return self.state, term_reward, terminal, info

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

if __name__ == '__main__':
    import sys, os
    from stable_baselines3.ppo import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
    from ESM_GB1 import PolicyNet
    import pickle
    import torch
    import warnings

    tensorboard_log = "./tensorboard_logs/"

    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help="path to save results", default="./checkpoints")

    # ppo algorithms
    parser.add_argument('--gamma', type=float, default=0.99, help="discount_factor")
    parser.add_argument('--steps', type=int, default=30000, help="total time steps")
    parser.add_argument('--ent_coef', type=float, default=0.2, help="encourage exploration")

    parser.add_argument('--clip', type=float, default=0.2, help="")
    # parser.add_argument('--kl_target', type=float, default=0.1, help="")
    parser.add_argument('--max_len', type=int, default=58)

    # environment
    parser.add_argument('--num_envs', type=int, default=10, help="number of environments")
    parser.add_argument('--n_steps', type=int, default=20, help="number of roll out steps")
    parser.add_argument('--max_step', type=int, default=20, help="maximum number of steps")
    parser.add_argument('--score_stop_criteria', type=float, default=0, help="stop_criteria")

    args = parser.parse_args()
    path = args.path
    t1 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    action_space = gym.spaces.multi_discrete.MultiDiscrete([4,33])
    observation_space = gym.spaces.MultiDiscrete([33]*args.max_len)

    fitness = {}
    # PredictedFitness_96 or PredictedFitness_192 or PredictedFitness_288
    for row in csv.reader(open("./reward/GB1/PredictedFitness_96.csv")):
        if row[0] == 'AACombo':
            continue
        fitness[row[0]] = float(row[1])

    ground = {}
    for row in csv.reader(open("./data/GB1.csv")):
        if row[0] == 'Variants':
            continue
        ground[row[0]] = float(row[4])
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

    m_env_kwargs = {"action_space": action_space, "observation_space": observation_space, "args": args}


    m_env = make_vec_env(GB1Env, n_envs=args.num_envs, env_kwargs=m_env_kwargs)


    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=path + '/', name_prefix='rl_model')

    model = PPO(PolicyNet, m_env, learning_rate=1e-4, verbose=1, n_steps=args.n_steps, ent_coef=args.ent_coef,
                gamma=args.gamma, clip_range=args.clip,tensorboard_log=tensorboard_log, device=device,batch_size=64)

    print_trainable_parameters(model.policy)

    model.learn(total_timesteps=args.steps, callback=checkpoint_callback)
    t2 = time.time()

    print("finish training in %.4f" % (t2 - t1))
    print("saving model.....")
    model.save(path=path + "/ppo")

    collected_seqs_list = list(collected_seqs_set)
    seq_fitness = {}
    for i in collected_seqs_list:
        seq_fitness[i] = fitness[i]
    sorted_dct = dict(sorted(seq_fitness.items(), key=lambda kv: kv[1], reverse=True))
    seq = list(sorted_dct.keys())
    values = list(sorted_dct.values())

    gb1_protein = []
    datas = pd.read_csv(path_96, names=['AACombo', 'Fitness'], header=0)
    for i in range(len(datas)):
        gb1_protein.append(datas["AACombo"][i])

    total_protein = []
    total_fitness = []
    predict = []
    for index, sequence in enumerate(seq):
        if sequence not in gb1_protein:
            total_protein.append(sequence)
            total_fitness.append(ground[sequence])
            predict.append(values[index])

    seq_96 = total_protein[:96]
    values_96 = total_fitness[:96]
    df_96 = pd.DataFrame({"AACombo": seq_96, "Fitness": values_96})
    df_96.to_csv(r"./output_384_training_seqs/GB1/96_GB1.csv", index=False)
    seq_384 = total_protein[:384]
    values_384 = total_fitness[:384]
    df_384 = pd.DataFrame({"AACombo": seq_384, "Fitness": values_384})
    df_384.to_csv(r"./output_384_training_seqs/GB1/384_GB1.csv", index=False)
    seq_288 = total_protein[:288]
    values_288 = total_fitness[:288]
    df_288 = pd.DataFrame({"AACombo": seq_288, "Fitness": values_288})
    df_288.to_csv(r"./output_384_training_seqs/GB1/288_GB1.csv", index=False)
