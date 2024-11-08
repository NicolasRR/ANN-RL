import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import argparse
import torch.nn.init as init
import random
from collections import deque
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size,dropout_rate, initilization = False):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # BatchNorm1d layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        if initilization:
            # Xavier initialization
            init.xavier_uniform_(self.fc1.weight)
            init.xavier_uniform_(self.fc2.weight)
            init.xavier_uniform_(self.fc3.weight)
            init.xavier_uniform_(self.fc4.weight)


        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)  
    
        return self.fc4(x)

class ReplayBuffer():
    def __init__(self, replay_size):
        self.state_buffer = None
        self.action_buffer = None
        self.reward_buffer = None
        self.next_state_buffer = None
        self.importance_buffer = None  
        self.terminal_buffer = None  
        self.replay_size = replay_size

    def add(self, state, action, reward, next_state, terminal):
        if self.state_buffer is None:
            self.state_buffer = np.array(state)
            self.action_buffer = np.array([action])
            self.reward_buffer = np.array([reward])
            self.next_state_buffer = np.array(next_state)
            self.terminal_buffer = np.array([terminal])
            self.importance_buffer = np.array([1], dtype=np.float32)
        else:
            self.state_buffer = np.vstack((self.state_buffer[-self.replay_size:], state)).astype(np.float32)
            self.action_buffer = np.hstack((self.action_buffer[-self.replay_size:], action))
            self.reward_buffer = np.hstack((self.reward_buffer[-self.replay_size:], reward))
            self.terminal_buffer = np.hstack((self.terminal_buffer[-self.replay_size:], terminal))
            self.next_state_buffer = np.vstack((self.next_state_buffer[-self.replay_size:], next_state)).astype(np.float32)
            self.importance_buffer = np.hstack((self.importance_buffer[-self.replay_size:], np.max(self.importance_buffer)))
    
    
    def update(self,importance, idx):

        self.importance_buffer[idx] = importance

    def sample(self, batch_size, alpha):
        if self.state_buffer is None or self.state_buffer.shape[0] < self.replay_size:
            return None
        else:
            if alpha>1e-5:
                idx = np.random.choice(len(self.state_buffer), batch_size, replace=False, p = self.importance_buffer**alpha/np.sum(self.importance_buffer**alpha))
            else:
                idx = np.random.choice(len(self.state_buffer), batch_size, replace=False)

            return [self.state_buffer[idx],self.action_buffer[idx],self.reward_buffer[idx],self.next_state_buffer[idx],self.terminal_buffer[idx],idx]

class DQNAgent:
    def __init__(
        self,
        state_size : int,
        action_size : int,
        hidden_size:int,
        replay_size: int,
        learning_rate: float,
        initial_epsilon: float,
        final_epsilon: float,
        epsilon_decay: float,
        discount_factor: float = 0.95,
        dropout_rate: float = 0.1,
        weight_decay: float = 0.01,
        scheduler=False,
        target_network = False,
        target_network_update = 100,
        alpha = 1,
        amsgrad = False,
        device = "cpu"
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.device = device
        self.qnetwork = QNetwork(state_size=state_size, action_size=action_size, hidden_size=hidden_size, dropout_rate=dropout_rate).to(self.device)
      
        if target_network: 
            self.target_network = QNetwork(state_size=state_size, action_size=action_size, hidden_size=hidden_size, dropout_rate=dropout_rate).to(self.device)
            self.target_network.load_state_dict(self.qnetwork.state_dict())
            self.target_network.eval()
        else:
            self.target_network = None
        self.replay_buffer = ReplayBuffer(replay_size=replay_size)
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.optimizer = torch.optim.AdamW(self.qnetwork.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=amsgrad)
        if scheduler:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=self.optimizer, max_lr=args.lr, total_steps=args.iterations, 
                                                            pct_start=args.warmup_percent, anneal_strategy="linear", 
                                                            cycle_momentum=False, div_factor=1e2, final_div_factor=.1)
        else:
            self.scheduler = None
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.target_network_update = target_network_update
        self.criterion = nn.SmoothL1Loss()
        self.alpha = alpha


    def get_action(self, obs, env) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment

        if np.random.random() < self.epsilon:
            action = env.action_space.sample() 
            return action

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            with torch.no_grad():
                q_values = self.qnetwork(torch.tensor(obs,device=self.device).unsqueeze(0))
                action = int(torch.argmax(q_values))
                return action

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        next_obs: tuple[int, int, bool],
        terminal: bool,
        target_count: int,
        batch_size: int = 32,
    ):
        
        self.replay_buffer.add(obs, action, reward, next_obs, terminal)

        sample_replay = self.replay_buffer.sample(batch_size, self.alpha)
        if sample_replay is None:
            return None, target_count
        idx = sample_replay[-1]
        replay_reward = torch.tensor(sample_replay[2]).to(self.device)
        actions = torch.tensor(sample_replay[1]).to(self.device)
        replay_obs = torch.tensor(sample_replay[0]).to(self.device)
        replay_next_obs = torch.tensor(sample_replay[3]).to(self.device)
        non_terminal = torch.tensor(~sample_replay[4]).to(self.device)
        
        self.qnetwork.train()
        q_values_obs = self.qnetwork(replay_obs)

        with torch.no_grad():
            q_values_next_obs = torch.zeros_like(q_values_obs, device=self.device)

            if self.target_network is not None:            
                q_values_next_obs[non_terminal,:] = self.target_network(replay_next_obs[non_terminal,:])
            else:
                q_values_next_obs[non_terminal,:] = self.qnetwork(replay_next_obs[non_terminal,:])
            

        self.optimizer.zero_grad()

        expected = replay_reward + self.discount_factor * torch.max(q_values_next_obs,dim=1).values
        current = torch.gather(q_values_obs, 1, actions.unsqueeze(1)).squeeze(1)   
        if self.alpha > 1e-5:
            delta = torch.abs(expected - current).detach().cpu().numpy()
            self.replay_buffer.update(importance=delta, idx=idx)
            
        loss = self.criterion(current,expected)

        loss.backward()
        torch.nn.utils.clip_grad_value_(self.qnetwork.parameters(), 100)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.qnetwork.eval()
        if self.target_network is not None and target_count == self.target_network_update:
            self.target_network.load_state_dict(self.qnetwork.state_dict())
            self.target_network.eval()
            target_count = 0
        else:
            target_count += 1

        return loss.item(), target_count


    def decay_epsilon(self):
        if self.replay_buffer.reward_buffer.shape[0] > self.replay_buffer.replay_size:
            self.epsilon = max(self.final_epsilon, self.epsilon*self.epsilon_decay)


def run(args, env):
    learning_rate = args.learning_rate
    n_episodes = args.n_episodes
    start_epsilon = args.start_epsilon
    final_epsilon = args.final_epsilon
    epsilon_decay = args.epsilon_decay
    # reduce the exploration over time
    batch_size = args.batch_size
    discount_factor = args.discount_factor
    replay_size = args.replay_size
    hidden_size=args.hidden_size
    dropout_rate=  args.dropout_rate
    weight_decay=args.weight_decay
    target_network = args.target_network,
    target_network_update = int(args.target_network_update)
    alpha = args.alpha
    np.random.seed(args.seed)
    intermediate_reward = args.intermediate_reward
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    agent = DQNAgent(
        learning_rate=learning_rate,
        state_size=2,
        action_size=3,
        discount_factor=discount_factor,
        final_epsilon=final_epsilon,
        hidden_size=hidden_size,
        epsilon_decay=epsilon_decay,
        initial_epsilon=start_epsilon,
        replay_size=replay_size,
        dropout_rate=dropout_rate,
        target_network=target_network,
        weight_decay=weight_decay,
        target_network_update=target_network_update,
        alpha=alpha,
        amsgrad=args.amsgrad,
        device = 'cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu'

    )
    if args.wandb:
        wandb.init(project='ANN-1', config={"learning_rate": learning_rate, "n_episodes": n_episodes, "start_epsilon": start_epsilon, "final_epsilon": final_epsilon, "epsilon_decay": epsilon_decay, "batch_size": batch_size, "discount_factor": discount_factor, "replay_size": replay_size, "hidden_size": hidden_size, "dropout_rate": dropout_rate, "weight_decay":weight_decay, "target_network":target_network, "alpha":alpha,"target_network_update":target_network_update,"intermediate_reward":intermediate_reward, "w_position":args.w_position, "w_velocity":args.w_velocity, "amsgrad":args.amsgrad}, name='DQN')


    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
    with tqdm(total=n_episodes, desc=f"Episode 0/{n_episodes}") as pbar:
        target_count = 0
        finished = 0
        empty = True
        cumulative_auxiliary_reward = 0
        cumulative_env_reward = 0
        for episode in tqdm(range(n_episodes)):
            obs, info = env.reset()
            done = False
            # play one episode
            t = 0
            episode_auxiliary_reward = 0
            episode_env_reward = 0
            episode_loss = 0
            
            while not done:
                action = agent.get_action(obs, env)
                next_obs, env_reward, terminated, truncated, info = env.step(action)

                # update if the environment is done and the current obs
                done = terminated or truncated
                aux_reward =intermediate_reward*np.min((args.w_position*(next_obs[0]-obs[0])/(0.5+1.2) + args.w_velocity*(np.abs(next_obs[1]))/0.7, 1))

                loss, target_count = agent.update(obs, action, env_reward+aux_reward, next_obs, batch_size=batch_size, target_count=target_count, terminal=terminated)
                    
                if loss is not None:
                    episode_auxiliary_reward += aux_reward
                    episode_env_reward += env_reward
                    episode_loss+=loss
                obs = next_obs
                t+=1

            pbar.set_description(f"Episode {episode + 1}/{n_episodes}")
            pbar.set_postfix(train_loss=episode_loss, epsilon=agent.epsilon, target_count=target_count, episode_steps=t, episode_auxiliary_reward=episode_auxiliary_reward, episode_env_reward=episode_env_reward)
            pbar.update(1)
            pbar.refresh() 
            if not empty:
                finished += terminated
                cumulative_auxiliary_reward += episode_auxiliary_reward
                cumulative_env_reward += episode_env_reward

                agent.decay_epsilon()
                if args.wandb:
                    wandb.log({"train_loss": episode_loss, "epsilon": agent.epsilon, "episode_steps": t, "finished": finished, "episode_env_reward":episode_env_reward, "episode_aux_reward":episode_auxiliary_reward, "cumulative_env_reward":cumulative_env_reward, "cumulative_aux_reward":cumulative_auxiliary_reward})

                    
            if loss is not None:
                empty = False




if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Script for pretraining a language model")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--n_episodes", type=int, default=5_000)
    parser.add_argument("--start_epsilon", type=float, default=0.9)
    parser.add_argument("--final_epsilon", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--discount_factor", type=float, default=0.95)
    parser.add_argument("--replay_size", type=int, default=10_000)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--target_network_update", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--intermediate_reward", type=float, default=0)
    parser.add_argument("--w_position", type=float, default=1.0)
    parser.add_argument("--w_velocity", type=float, default=1.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--target_network", action="store_true")
    parser.add_argument("--gpu", action="store_true")

    args = parser.parse_args()
    env = gym.make('MountainCar-v0')
    env.action_space.seed(args.seed)
    observation, info = env.reset(seed=args.seed)
    run(args, env)