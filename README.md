# Knowledge-aware Reinforced Language Models for Protein Directed Evolution

The official implementation of the ICML'2024 paper Knowledge-aware Reinforced Language Models for Protein Directed Evolution

# Environments

To set up the environment for running KnowRLM, use the command `pip install -r requirements.txt`. For executing the code in Step1 and Step2, please follow the specific environment setup instructions provided in their respective libraries. It is recommended to create separate virtual environments for each step to ensure compatibility and avoid conflicts.

# Getting started

## Step1: Obtain the Initial Candidate Library

Acquire the initial set of 96 candidate sequences using the CLADE package. These sequences will be added to the candidate sequence library. The CLADE package is available [here](https://github.com/WeilabMSU/CLADE). Refer to the paper: [Qiu Yuchi, Jian Hu, Guo-Wei Wei, "Cluster learning-assisted directed evolution" Nature Computational Science (2021)](https://www.nature.com/articles/s43588-021-00168-y).

## Step2: Train the Reward Model

Train the reward function using the candidate sequence library.The reward function outputs predictions for the entire protein space, which are saved as a CSV file. The reward value for each protein can be directly retrieved from this table. The reward function can be found [here](https://github.com/fhalab/MLDE). Refer to the paper:[Wittmann, Bruce J., Yisong Yue, and Frances H. Arnold. "Informed training set design enables efficient machine learning-assisted directed protein evolution." Cell Systems (2021).](https://www.cell.com/cell-systems/fulltext/S2405-4712(21)00286-6?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2405471221002866%3Fshowall%3Dtrue)

## Step3: Obtain Candidate Sequences Through Reinforcement Learning

Run the script from the script folder using: `python GB1_env.py or PhoQ_env.py` to obtain the top 96 candidate sequences. Add these sequences to the candidate sequence library. Repeat Steps 2 and 3 for n iterations, where n∈[1,3], based on the experimental setup. The data sources for the amino acid knowledge graph used in this project can be found [here](https://www.biorxiv.org/content/10.1101/2023.08.03.551768v1.supplementary-material?versioned=true). Refer to the paper:[Breimann, S., Kamp, F., Steiner, H., and Frishman, D. "AAontology: An ontology of amino acid scales for interpretable machine learning." bioRxiv(2023)](https://www.biorxiv.org/content/10.1101/2023.08.03.551768v1)

Following the steps outlined above, the candidate sequence libraries obtained from our experimental runs are stored in `/data/96`, `/data/192`, `/data/288`, and `/data/384`, with the numbers indicating the quantity of candidate sequences. If you find the environment setup too cumbersome, you may consider using these pre-generated libraries directly.

## Step4: Predict Candidate Sequences Using the Predictor

The predictor and reward model are the same. Train the predictor using the candidate sequence library obtained from the previous steps. Predict the globally optimal 96 candidate sequences and add them to the candidate sequence library. The final library will contain 96 + 96×n + 96 sequences.  The experimental results are stored in the `results` directory. These results can also be directly used as the output of the reward function in Step 2 to provide rewards for the next round of reinforcement learning.

## Step5: Evaluation

Run the script from the script folder using: `python mean_max_3.py or NDCG.py`.

**Note:** This method involves data sampling rather than training, leading to randomness. It is recommended to conduct multiple experiments. The results reported in the paper represent the optimal outcomes.