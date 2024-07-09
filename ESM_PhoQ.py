from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Any
import collections

import gym
from gym import spaces
import torch
from torch import nn
import numpy as np
import math

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy,BasePolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    StateDependentNoiseDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    BernoulliDistribution,
    Distribution)
from transformers import AutoTokenizer,AutoModel,EsmForMaskedLM,EsmModel
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
protein = "SYMVWSWFIYVLSANLLLVIPLLWVAAWWSLRPIEALAKEVRELEEHNRELLNPATTRELTSLVRNLNRLLKSERERYDKYRTTLTDLTHSLKTPL<mask><mask>LQ<mask><mask>LRSLRSEKMSVSDAEPVMLEQISRISQQIGYYLHRASMRGGTLLSRELHPVAPLLDNLTSALNKVYQRKGVNISLDISPEISFVGEQ"
pos = {0:96,1:97,2:100,3:101}
re_pos = {96:0,97:1,100:2,101:3}
AMINO_ACIDS = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class ActionNet(nn.Module):

    def __init__(self,
                 latent_dim: int,
                 action_space: spaces,
                 ):
        super(ActionNet, self).__init__()
        self.action_space = action_space
        self.latent_dim = latent_dim

        self.flatten = nn.Flatten()
        self.pos_action_net = nn.Linear(latent_dim, 4)
        self.pos_dist = self._build_dist(4)

    def forward(self, embedding) -> torch.Tensor:
        output = self.flatten(embedding)
        pos_pd = self.pos_action_net(output)

        return pos_pd

    def _build_dist(self, dim: int) -> Distribution:
        return CategoricalDistribution(dim)

class ValueNet(nn.Module):

    def __init__(self,latent_dim: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.latent_dim = latent_dim
        self.summary = nn.Linear(self.latent_dim, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.out = nn.Linear(512, 1)
        # self.first_dropout = nn.Dropout(0.2)

    def forward(self, hidden_states):
        output = self.flatten(hidden_states)
        # output = self.first_dropout(output)
        output = self.summary(output)
        output = gelu(output)
        output = self.layer_norm(output)
        output = self.out(output)

        return output
import pandas as pd

# Load the Excel file
file_path = './data/amino_acid_knowledge_graph_normalized.xlsx'
distance_df = pd.read_excel(file_path, index_col=0)

# Function to find amino acids with distance less than K from a given amino acid
def find_AA(amino_acid, K):

    # Check if the amino acid is in the DataFrame
    if amino_acid not in distance_df.columns:
        return "Amino acid not found."

    # Filter amino acids with distance less than K
    return distance_df.index[(distance_df[amino_acid] <= K[1]) & (distance_df[amino_acid] >= K[0])].tolist()

class PolicyNet(BasePolicy):
    def __init__(
        self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            use_sde: bool = False,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Disable orthogonal initialization
        super(PolicyNet, self).__init__(observation_space, action_space)
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5
        self.use_sde = use_sde

        self.action_dist = make_proba_distribution(4)
        self._build(lr_schedule)
        self.amino_loss = nn.CrossEntropyLoss()
        self.pos_loss = nn.CrossEntropyLoss()

    def _get_data(self) -> Dict[str, Any]:
        data = dict()

        data.update(
            dict(
                observation_space=self.observation_space,
                action_space=self.action_space,
                use_sde=self.use_sde,
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
            )
        )
        return data
    def _build_mlp_extractor(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.mlp_extractor = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        # for name, param in self.mlp_extractor.named_parameters():
        #     param.requires_grad = False
        self.ref_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        # for name, param in self.ref_model.named_parameters():
        #     param.requires_grad = False
        self.tokens = self.tokenizer(protein, return_tensors="pt").to(device)
        self.AA2token = {}
        for j in AMINO_ACIDS:
            self.AA2token[j] = self.tokenizer(j, return_tensors="pt").to(device)['input_ids'][0][1]
    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        # self.lm_head = EsmLMHead()

        self.action_net = ActionNet(480*191, self.action_space)

        self.value_net = ValueNet(480*191)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: torch.Tensor,k:int, deterministic: bool = False) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """

        # pdb.set_trace()
        hidden_states = self._get_latent(obs)
        self.ref = self.ref_model(**self.tokens)[0]
        hidden_states = hidden_states/self.ref
        # Evaluate the values for the given observations
        self.pos_prob = self.action_net(hidden_states)  # (batch, seq_len)
        distribution = self._get_action_dist_from_latent(self.pos_prob)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        act = torch.tensor([])
        for i, action in enumerate(actions):
            ori_aa = ''
            ori_hidden_state = 0
            similarity = {}
            position = action.item()
            position = pos[position]
            for j in AMINO_ACIDS:
                id = self.AA2token[j]
                if id == obs[i][position]:
                    ori_aa = j
                    with torch.no_grad():
                        ori_hidden_state = self.mlp_extractor(obs)[0][0][position]
                    break
            AA_k = find_AA(ori_aa, k)
            for j in AA_k:
                id = self.AA2token[j]
                if id == obs[i][position]:
                    continue
                obs[i][position] = id
                with torch.no_grad():
                    last_hidden_state = self.mlp_extractor(obs)[0][0][position]
                similarity[ori_aa+' '+j] = torch.cosine_similarity(ori_hidden_state.unsqueeze(0), last_hidden_state.unsqueeze(0)).item()
            similarity = sorted(similarity.items(), key=lambda x: x[1], reverse=True)
            predicted_token_id = self.AA2token[similarity[0][0][2]]
            if min(act.shape) == 0:
                act = torch.tensor([position, predicted_token_id]).unsqueeze(0)
            else:
                act = torch.cat((act,torch.tensor([position, predicted_token_id]).unsqueeze(0)), dim=0)
        # with torch.no_grad():
        #     logits = self.mlp_extractor(obs,output_hidden_states=True)[0]
        #     # ref = self.ref_model(**self.tokens)[0]
        #     # last_hidden_state = last_hidden_state/ref
        #     # logits = self.lm_head(last_hidden_state)
        # act = torch.tensor([])
        # for i, action in enumerate(actions):
        #     position = action.item()
        #     position = pos[position]
        #     predicted_token_id = logits[i, position].argmax(axis=-1)
        #     if min(act.shape) == 0:
        #         act = torch.tensor([position, predicted_token_id]).unsqueeze(0)
        #     else:
        #         act = torch.cat((act,torch.tensor([position, predicted_token_id]).unsqueeze(0)), dim=0)
        # # pos = self.p_head(hidden_states).argmax(-1)
        values = self.value_net(hidden_states).squeeze(-1)
        return act, values, log_prob

    def _get_latent(self, obs: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Preprocess the observation if needed
        features = self.mlp_extractor(input_ids=obs,
            output_attentions=True,
            output_hidden_states=True)

        return features.hidden_states[-1]

    def _get_action_dist_from_latent(self, mean_actions: torch.Tensor, latent_sde: Optional[torch.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        """

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        hidden_states = self._get_latent(observation.long())
        self.pos_prob = self.action_net(hidden_states)
        distribution = self._get_action_dist_from_latent(self.pos_prob)
        return distribution.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        hidden_states = self._get_latent(obs.long())
        # distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        hidden_states = hidden_states / self.ref
        self.pos_prob = self.action_net(hidden_states)  # (batch, seq_len)
        distribution = self._get_action_dist_from_latent(self.pos_prob)
        act = []
        for i in actions[:,0]:
            act.append(re_pos[i.item()])
        act = torch.tensor(act).to(actions[0,0].device)
        log_prob = distribution.log_prob(act)
        entropy = distribution.entropy()

        values = self.value_net(hidden_states).squeeze(-1)
        return values, log_prob, entropy

def make_proba_distribution(
    dim: int = 4
) -> Distribution:
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    return CategoricalDistribution(dim)
