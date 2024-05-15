import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size,dropout_rate, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # BatchNorm1d layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer

    def forward(self, state):
        if state.shape[0] == 1:
            state = torch.unsqueeze(state, 0)
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))

        else:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))

         # Apply BatchNorm1d after fc1
        # x = self.dropout(x)  # Apply Dropout after BatchNorm1d
        
        return self.fc4(x)

class ReplayBuffer():
    def __init__(self, replay_size):
        self.state_buffer = None
        self.action_buffer = None
        self.reward_buffer = None
        self.next_state_buffer = None
        self.importance_buffer = None  
        self.replay_size = replay_size

    def add(self, state, action, reward, next_state):
        if self.state_buffer is None:
            self.state_buffer = np.array(state)
            self.action_buffer = np.array([action])
            self.reward_buffer = np.array([reward])
            self.next_state_buffer = np.array(next_state)
            self.importance_buffer = np.array([1], dtype=np.float32)
        else:
            self.state_buffer = np.vstack((self.state_buffer[-self.replay_size:], state)).astype(np.float32)
            self.action_buffer = np.hstack((self.action_buffer[-self.replay_size:], action))
            self.reward_buffer = np.hstack((self.reward_buffer[-self.replay_size:], reward))
            self.next_state_buffer = np.vstack((self.next_state_buffer[-self.replay_size:], next_state)).astype(np.float32)
            self.importance_buffer = np.hstack((self.importance_buffer[-self.replay_size:], np.max(self.importance_buffer)))
    
    
    def update(self,importance, idx):

        self.importance_buffer[idx] = importance

    def sample(self, batch_size, alpha):
        if self.state_buffer.shape[0] < self.replay_size:
            return None
        else:
            if alpha>1e-5:
                idx = np.random.choice(len(self.state_buffer), batch_size, replace=False, p = self.importance_buffer**alpha/np.sum(self.importance_buffer**alpha))
            else:
                idx = np.random.choice(len(self.state_buffer), batch_size, replace=False)

            return [self.state_buffer[idx],self.action_buffer[idx],self.reward_buffer[idx],self.next_state_buffer[idx],idx]

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
        seed: int = 42,
        weight_decay: float = 0.01,
        scheduler=None,
        target_network = False,
        target_network_update = 100,
        alpha = 1,
        amsgrad = False
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
        self.qnetwork = QNetwork(state_size=state_size, action_size=action_size, hidden_size=hidden_size, dropout_rate=dropout_rate,seed=seed).to(DEVICE)
      
        if target_network: 
            self.target_network = QNetwork(state_size=state_size, action_size=action_size, hidden_size=hidden_size, dropout_rate=dropout_rate,seed=seed).to(DEVICE)
            self.target_network.load_state_dict(self.qnetwork.state_dict())
            self.target_network.eval()
        else:
            self.target_network = None
        self.replay_buffer = ReplayBuffer(replay_size=replay_size)
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.optimizer = torch.optim.AdamW(self.qnetwork.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=amsgrad)
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.scheduler = scheduler
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
                q_values = self.qnetwork(torch.tensor(obs,device=DEVICE).unsqueeze(0))
                action = int(torch.argmax(q_values))
                return action

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        next_obs: tuple[int, int, bool],
        target_count: int,
        batch_size: int = 32,
    ):
        
        self.replay_buffer.add(obs, action, reward, next_obs)

        sample_replay = self.replay_buffer.sample(batch_size, self.alpha)
        if sample_replay is None:
            return None, target_count
        idx = sample_replay[-1]
        replay_reward = torch.tensor(sample_replay[2]).to(DEVICE)
        actions = torch.tensor(sample_replay[1]).to(DEVICE)
        replay_obs = torch.tensor(sample_replay[0]).to(DEVICE)
        replay_next_obs = torch.tensor(sample_replay[3]).to(DEVICE)
        non_terminal = ~torch.isnan(replay_next_obs)[:,0]
        
        self.qnetwork.train()
        q_values_obs = self.qnetwork(replay_obs)

        with torch.no_grad():
            q_values_next_obs = torch.zeros_like(q_values_obs, device=DEVICE)

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
            
        # loss = self.criterion(current,expected)
        loss = torch.sum((current - expected)**2)


        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.qnetwork.parameters(), 100)

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
    logging_interval = args.logging_interval
    hidden_size=args.hidden_size
    dropout_rate=  args.dropout_rate
    weight_decay=args.weight_decay
    target_network = args.target_network,
    target_network_update = int(args.target_network_update)
    alpha = args.alpha
    np.random.seed(args.seed)
    auxiliary = args.auxiliary
    intermediate_reward = args.intermediate_reward
    final_reward = args.final_reward

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
        seed=args.seed,
        amsgrad=args.amsgrad,
    )
    if args.wandb:
        wandb.init(project='ANN', config={"learning_rate": learning_rate, "n_episodes": n_episodes, "start_epsilon": start_epsilon, "final_epsilon": final_epsilon, "epsilon_decay": epsilon_decay, "batch_size": batch_size, "discount_factor": discount_factor, "replay_size": replay_size, "hidden_size": hidden_size, "dropout_rate": dropout_rate, "weight_decay":weight_decay, "target_network":target_network, "alpha":alpha,"target_network_update":target_network_update, "auxiliary":auxiliary, "final_reward":final_reward, "intermediate_reward":intermediate_reward, "amsgrad":args.amsgrad}, name='DQN')


    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
    with tqdm(total=n_episodes, desc=f"Episode 0/{n_episodes}") as pbar:
        losses = []
        rewards = []
        target_count = 0
        finished = []
        episode_steps = []
        for episode in tqdm(range(n_episodes)):
            obs, info = env.reset()
            done = False
            # play one episode
            t = 0
            episode_reward = 0
            episode_loss = 0
            
            while not done:
                action = agent.get_action(obs, env)
                next_obs, reward, terminated, truncated, info = env.step(action)

                # update if the environment is done and the current obs
                done = terminated or truncated
                # reward+=0.5*intermediate_reward*(1.8-(0.6-next_obs[0]))/1.8+0.5*intermediate_reward*(np.abs(next_obs[1]))/0.7
                reward+=intermediate_reward*(np.abs(next_obs[1]))/0.7

                if terminated:
                    next_obs = (None, None)
                    reward+=final_reward

                loss, target_count = agent.update(obs, action, reward, next_obs, batch_size=batch_size, target_count=target_count)
                if loss is not None:
                    episode_reward += reward
                    episode_loss+=loss
                obs = next_obs
                t+=1
            finished.append(terminated)
            episode_steps.append(t)
            rewards.append(episode_reward)
            losses.append(episode_loss)
            agent.decay_epsilon()
            pbar.set_description(f"Episode {episode + 1}/{n_episodes}")
            pbar.set_postfix(train_loss=loss, epsilon=agent.epsilon, target_count=target_count)
            pbar.update(1)
            pbar.refresh() 
            if episode % logging_interval == 0 and episode>100:
                if args.wandb:
                    wandb.log({"train_loss": np.mean(losses), "epsilon": agent.epsilon, "episode_steps": np.mean(episode_steps), "finished": np.sum(finished), "mean_reward": np.mean(rewards)})
                losses = []
                rewards = []
                finished = []
                episode_steps = []




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
    parser.add_argument("--logging_interval", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--target_network_update", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--intermediate_reward", type=float, default=0)
    parser.add_argument("--final_reward", type=float, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--auxiliary", action="store_true")
    parser.add_argument("--target_network", action="store_true")

    args = parser.parse_args()
    env = gym.make('MountainCar-v0')

    run(args, env)