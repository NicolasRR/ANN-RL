import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import argparse
from scipy.special import softmax
from tqdm import tqdm
import wandb

class DynaAgent:
    def __init__(self, discr_step=np.array([0.025, 0.005]), gamma=0.99, decay= 0.99, start_epsilon=0.9, min_epsilon=0.05, k=5, replay_size=10_000, alpha=0,env=gym.make('MountainCar-v0')):
    
        self.born_inf=env.observation_space.low
        self.born_sup=env.observation_space.high
        self.discr_step = discr_step
        self.n_states = ((self.born_sup - self.born_inf) / self.discr_step).astype(int)+1
        self.n_actions=env.action_space.n
        self.gamma = gamma
        self.decay=decay
        self.epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.k = k
        self.replay_size = replay_size
        
        # Initialize model components
        self.P_hat = np.ones((self.n_states[0], self.n_states[1], self.n_actions, self.n_states[0], self.n_states[1]))
        self.R_hat = np.zeros((self.n_states[0], self.n_states[1], self.n_actions))
        self.count_matrix=np.zeros_like(self.R_hat)
        self.Q = np.zeros((self.n_states[0], self.n_states[1], self.n_actions))
        self.delta_Q = []
        self.replay_buffer = None
        self.alpha = alpha
        
        
    def discretize_state(self, state):
        discr_state = (state - self.born_inf) / self.discr_step
        return discr_state.astype(int)
    
    def update_model(self, discr_state, action, reward, discr_next_state):
        # Update transition probabilities

        self.P_hat[discr_state[0], discr_state[1], action, discr_next_state[0], discr_next_state[1]] += 1
        
        # Update rewards
        self.R_hat[discr_state[0], discr_state[1], action] += reward
        self.count_matrix[discr_state[0], discr_state[1], action]+=1
    
    
    def update_q_value(self, discr_state, action, idx=None):
    
        # Precompute max Q-values for all next states
        max_next_q_values = np.max(self.Q, axis=-1)
        if len(discr_state.shape)>1:
            position = discr_state[:,0]
            velocity = discr_state[:,1]
        else:
            position = discr_state[0]
            velocity = discr_state[1]
  
        if len(discr_state.shape)>1:
            discounted_rewards = self.gamma * np.sum(self.P_hat[position, velocity, action, :,:] / np.sum(self.P_hat[position, velocity, action, :,:],axis=(-1,-2)).reshape(-1,1,1)*max_next_q_values, axis=(-1,-2))
        else:
            discounted_rewards = self.gamma * np.sum(self.P_hat[position, velocity, action, :,:] / np.sum(self.P_hat[position, velocity, action, :,:],axis=(-1,-2))*max_next_q_values, axis=(-1,-2))
    
        # Compute the Q-value update
        update_value = self.R_hat[position, velocity, action] / self.count_matrix[position, velocity, action] + discounted_rewards
        delta = update_value - self.Q[position, velocity, action]
        # Update Q-value
        if len(discr_state.shape)>1:
            self.delta_Q.extend(delta)
        else:
            self.delta_Q.append(delta)

        self.Q[position, velocity, action] = update_value
        # FIXME: do we have to reset the count matrix after updating the Q-value?
        if self.alpha>1e-5 and idx is not None:
            self.importance_buffer[idx] = np.abs(delta)

        return delta
        
    def update(self, state, action, reward, next_state):
        discr_state = self.discretize_state(state)
        discr_next_state = self.discretize_state(next_state)
        # FIXME: add none when too small replay buffer
        self.update_model(discr_state, action, reward, discr_next_state)
        _ = self.update_q_value(discr_state, action)
        
        # Store experience in replay buffer
        if self.replay_buffer is None:
            self.replay_buffer = np.array([(discr_state[0], discr_state[1], action)])
            self.importance_buffer = np.array([1], dtype=np.float32)
        else:
            self.replay_buffer = np.vstack((self.replay_buffer[-self.replay_size:], (discr_state[0], discr_state[1], action)))
            self.importance_buffer = np.hstack((self.importance_buffer[-self.replay_size:], np.max(self.importance_buffer)),dtype=np.float32)
        
        # Sample from replay buffer for further updates
        # Randomly sample from replay buffer
        if len(self.replay_buffer) >= self.replay_size:
            if self.alpha>1e-5:
                rand_idx = np.random.choice(len(self.replay_buffer), self.k, replace=False, p = self.importance_buffer**self.alpha/np.sum(self.importance_buffer**self.alpha))
            else:
                rand_idx = np.random.choice(len(self.replay_buffer), self.k, replace=False)

            rand_experience = self.replay_buffer[rand_idx]
            return np.mean(self.update_q_value(rand_experience[:,0:2], rand_experience[:,-1], idx=rand_idx))
        return None
        
    def select_action(self, state, env):
        if np.random.random() < self.epsilon:
            action = env.action_space.sample() 
            return action
        else:
            discr_state = self.discretize_state(state)
            return np.argmax(self.Q[discr_state[0], discr_state[1], :])
    
    def decay_epsilon(self):
        if len(self.replay_buffer) >= self.replay_size:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

def plot_max_Q(Q_values, t):

    data = np.max(Q_values, axis=-1).T
    plt.figure()

    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title(f'Max Q-value at episode {t}')
    plt.colorbar()
    return plt
    
    
def run(args):
    # Parameters
    seed = args.seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    discr_step = [args.discr_pos, args.discr_vel]
    k = args.k
    alpha = args.alpha
    decay = args.decay
    discount = args.discount_factor
    replay_size = args.replay_size
    start_epsilon=  args.start_epsilon
    n_episodes = args.n_episodes
    final_epsilon= args.final_epsilon
    snapshot_interval = args.snapshot_interval
    # Create the environment
    env = gym.make('MountainCar-v0')
    env.action_space.seed(seed)

    observation, info = env.reset(seed=seed)
    # Create the DynaAgent
    agent = DynaAgent(decay=decay, start_epsilon=start_epsilon, gamma=discount, discr_step=discr_step, k=k,alpha=alpha, replay_size=replay_size,env=env, min_epsilon=final_epsilon)
    if args.wandb:
        wandb.init(project='ANN-1', config={"seed":seed,"n_episodes": n_episodes, "start_epsilon": start_epsilon, "final_epsilon": final_epsilon, "epsilon_decay": decay, "batch_size": k, "discount_factor": discount, "replay_size": replay_size,"alpha":alpha, "discr_pos":args.discr_pos, "discr_vel":args.discr_vel}, name='dyna')

    # Train the agent
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
    with tqdm(total=n_episodes, desc=f"Episode 0/{n_episodes}") as pbar:
        finished = 0
        empty = True
        cumulative_env_reward = 0
        fig,ax=plt.subplots(1,2,figsize=(11,5))

        for episode in tqdm(range(n_episodes)):
            state, info = env.reset()
            done = False
            # play one episode
            t = 0
            episode_env_reward = 0
            episode_loss = 0
            x=[state[0]]
            v=[state[1]]
            while not done:
                action = agent.select_action(state, env)
                next_state, reward, terminated, truncated, _ = env.step(action)

                loss = agent.update(state, action, reward, next_state)
                done = terminated or truncated
                state = next_state
                if loss is not None:
                    episode_env_reward += reward
                    episode_loss+=loss           
                t+=1
                if episode % snapshot_interval == 0:
                    x.append(state[0])  
                    v.append(state[1])
            
            agent.decay_epsilon()

            pbar.set_description(f"Episode {episode + 1}/{n_episodes}")
            pbar.set_postfix(train_loss=episode_loss, epsilon=agent.epsilon, episode_steps=t, episode_env_reward=episode_env_reward, finished=finished, cumulative_env_reward=cumulative_env_reward)
            pbar.update(1)
            pbar.refresh() 
            if not empty:
                finished += terminated
                cumulative_env_reward += episode_env_reward

                agent.decay_epsilon()
                if args.wandb:
                    wandb.log({"train_loss": episode_loss, "epsilon": agent.epsilon, "episode_steps": t, "finished": finished, "episode_env_reward":episode_env_reward, "cumulative_env_reward":cumulative_env_reward})

            if (episode // snapshot_interval >=1 and episode % snapshot_interval == 0)  or episode == n_episodes - 1:
                max_q = plot_max_Q(agent.Q, episode)
                color = f"{0.9*(1-(episode+1)/n_episodes)}"
                ax[0].plot(range(t+1),x, c=color, zorder = 1)
                ax[1].plot(range(t+1),v, c=color, zorder = 1)                
                if args.wandb:
                    wandb.log({"max_Q": wandb.Image(max_q,caption=f'Max Q-value at episode {episode}')})

            if loss is not None:
                empty = False


        env.close()
        ax[0].set_xlabel('Episodes')
        ax[0].set_ylabel('Position')
        ax[1].set_xlabel('Position')
        ax[1].set_ylabel('Velocity')
        plt.legend()
        if args.wandb:
            wandb.log({"trajectories": wandb.Image(fig,caption=f'Trajectories')})


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Script for pretraining a language model")
    parser.add_argument("--n_episodes", type=int, default=10_000)
    parser.add_argument("--snapshot_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--discr_pos", type=float, default=0.05)
    parser.add_argument("--discr_vel", type=float, default=0.005)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--replay_size", type=int, default=10_000)
    parser.add_argument("--start_epsilon", type=float, default=0.9)
    parser.add_argument("--final_epsilon", type=float, default=0.05)
    parser.add_argument("--wandb", action='store_true')

    args = parser.parse_args()

    run(args)

