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
from dqn import DQNAgent, QNetwork, ReplayBuffer

class DQNAgentV2(DQNAgent):
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
        reward_hidden_size :int,
        discount_factor: float = 0.95,
        dropout_rate: float = 0.1,
        weight_decay: float = 0.01,
        scheduler=False,
        target_network = False,
        target_network_update = 100,
        alpha = 1,
        amsgrad = False,
        device = "cpu",
        reward_factor = 1/5,
        running_window = 100,

    ):
        super().__init__(state_size=state_size, action_size=action_size, hidden_size=hidden_size, replay_size=replay_size, learning_rate=learning_rate, initial_epsilon=initial_epsilon, final_epsilon=final_epsilon, epsilon_decay=epsilon_decay, discount_factor=discount_factor, dropout_rate=dropout_rate, weight_decay=weight_decay, scheduler=scheduler, target_network=target_network, target_network_update=target_network_update, alpha=alpha, amsgrad=amsgrad, device=device)        
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.predictor = QNetwork(state_size=state_size, action_size=1, hidden_size=reward_hidden_size, dropout_rate=0.0,initilization=True).to(self.device)
        self.target_predictor = QNetwork(state_size=state_size, action_size=1, hidden_size=reward_hidden_size, dropout_rate=0.0, initilization=True).to(self.device)
        for param in self.target_predictor.parameters():
            param.requires_grad = False
        self.criterion_intrinsic = nn.MSELoss()
        self.mse_buffer = None
        self.reward_factor = reward_factor
        self.optimizer_intrinsic = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=amsgrad)
        self.target_predictor.eval()
        self.predictor.train()
        self.running_window = running_window

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
        
        #################
        self.predictor.train()
        self.optimizer_intrinsic.zero_grad()
        predicted_intrinsic = self.predictor(torch.tensor(next_obs, device=self.device).unsqueeze(0))
        target_intrinsic = self.target_predictor(torch.tensor(next_obs, device=self.device).unsqueeze(0))
        intrinsic_loss = self.criterion_intrinsic(predicted_intrinsic, target_intrinsic)

        #################

        sample_replay = self.replay_buffer.sample(batch_size, self.alpha)
        if sample_replay is None:
            if self.mse_buffer is None:
                self.mse_buffer = np.array([intrinsic_loss.item()])
            else:
                self.mse_buffer = np.hstack((self.mse_buffer[-self.running_window:], intrinsic_loss.item()))
            return None, target_count, None, None
        
        intrinsic_loss.backward()
        self.optimizer_intrinsic.step()
        #################
        with torch.no_grad():
            self.predictor.eval()
            means = np.mean(self.replay_buffer.state_buffer[-self.running_window:], axis=0)
            SD = np.std(self.replay_buffer.state_buffer[-self.running_window:], axis=0)
            normalized_obs = torch.tensor((next_obs-means)/SD, device=self.device).unsqueeze(0)
            predicted_intrinsic = self.predictor(normalized_obs)
            target_intrinsic = self.target_predictor(normalized_obs)
            loss = self.criterion_intrinsic(predicted_intrinsic, target_intrinsic)
            RND = self.reward_factor*np.clip((loss.item()-np.mean(self.mse_buffer[-self.running_window:]))/np.std(self.mse_buffer[-self.running_window:]), -5,5)
            self.mse_buffer = np.hstack((self.mse_buffer[-self.running_window:], RND))
        
        reward+=RND
        #################
        idx = sample_replay[-1]
        replay_reward = torch.tensor(sample_replay[2]).to(self.device)
        actions = torch.tensor(sample_replay[1]).to(self.device)
        replay_obs = torch.tensor(sample_replay[0]).to(self.device)
        replay_next_obs = torch.tensor(sample_replay[3]).to(self.device)
        non_terminal = ~torch.isnan(replay_next_obs)[:,0]
        
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

        return loss.item(), target_count, RND, intrinsic_loss.item()


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
        seed=args.seed,
        amsgrad=args.amsgrad,
        device = 'cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu'

    )
    if args.wandb:
        wandb.init(project='ANN', config={"learning_rate": learning_rate, "n_episodes": n_episodes, "start_epsilon": start_epsilon, "final_epsilon": final_epsilon, "epsilon_decay": epsilon_decay, "batch_size": batch_size, "discount_factor": discount_factor, "replay_size": replay_size, "hidden_size": hidden_size, "dropout_rate": dropout_rate, "weight_decay":weight_decay, "target_network":target_network, "alpha":alpha,"target_network_update":target_network_update,"reward_factor":args.reward_factor, "amsgrad":args.amsgrad, "reward_hidden_size":args.reward_hidden_size}, name='DQN')


    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
    with tqdm(total=n_episodes, desc=f"Episode 0/{n_episodes}") as pbar:
        losses = []
        rewards = []
        target_count = 0
        finished = []
        episode_steps = []
        empty = True
        
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
                if terminated:
                    next_obs = (None, None)
                loss, target_count, RND, predictor_intrinsic_loss = agent.update(obs, action, reward, next_obs, batch_size=batch_size, target_count=target_count)
                    
                if loss is not None:
                    episode_reward += reward+RND
                    episode_loss+=loss
                obs = next_obs
                t+=1

            pbar.set_description(f"Episode {episode + 1}/{n_episodes}")
            pbar.set_postfix(train_loss=loss, epsilon=agent.epsilon, target_count=target_count)
            pbar.update(1)
            pbar.refresh() 
            if not empty:
                finished.append(terminated)
                episode_steps.append(t)
                rewards.append(episode_reward)
                losses.append(episode_loss)
                agent.decay_epsilon()
                if episode % logging_interval == 0 :
                    if args.wandb:
                        wandb.log({"train_loss": np.mean(losses), "epsilon": agent.epsilon, "episode_steps": np.mean(episode_steps), "finished": np.sum(finished), "mean_reward": np.mean(rewards)})
                    losses = []
                    rewards = []
                    finished = []
                    episode_steps = []
                    
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
    parser.add_argument("--logging_interval", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--reward_hidden_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--reward_factor", type=float, default=1/5)
    parser.add_argument("--target_network_update", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--target_network", action="store_true")
    parser.add_argument("--gpu", action="store_true")

    args = parser.parse_args()
    env = gym.make('MountainCar-v0')

    run(args, env)