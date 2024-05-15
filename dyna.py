import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import argparse


class DynaAgent:
    def __init__(self, discr_step=np.array([0.025, 0.005]), gamma=0.99, decay= 0.99, epsilon=0.9, min_epsilon=0.05, k=5, env=gym.make('MountainCar-v0')):
    
        self.born_inf=env.observation_space.low
        self.born_sup=env.observation_space.high
        self.discr_step = discr_step
        self.n_states = ((self.born_sup - self.born_inf) / self.discr_step).astype(int)+1
        self.n_actions=env.action_space.n
        self.gamma = gamma
        self.decay=decay
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.k = k
        
        # Initialize model components
        self.n_states_tot=self.n_states[0]*self.n_states[1]
        self.P_hat = np.ones(shape=(self.n_states_tot, self.n_actions, self.n_states_tot) )
        self.R_hat = np.zeros(shape=(self.n_states_tot,self. n_actions))
        self.count_matrix=np.zeros_like(self.R_hat)
        self.Q = np.zeros(shape=(self.n_states_tot, self.n_actions))
        self.delta_Q=[]
        self.replay_buffer = deque(maxlen=10000)
        
        
    def discretize_state(self, state):
        discr_state = (state - self.born_inf) / np.array(self.discr_step)
        return tuple(discr_state.astype(int))
        
    def encode_state(self,state):
     
        return state[0]+self.n_states[0]*state[1]
    
    def update_model(self, state, action, reward, next_state):
        discr_state = self.discretize_state(state)
        discr_next_state = self.discretize_state(next_state)
        
        # Update transition probabilities

        self.P_hat[self.encode_state(discr_state)][action][self.encode_state(discr_next_state)] += 1
        
        # Update rewards
        self.R_hat[self.encode_state(discr_state)][action] += reward
        self.count_matrix[self.encode_state(discr_state)][action]+=1
    
    
    def update_q_value(self, state, action):
        discr_state = self.discretize_state(state)
        current = self.encode_state(discr_state)
    
        # Precompute max Q-values for all next states
        max_next_q_values = np.max(self.Q, axis=1)
    
        # Compute the second term of the Q-value update equation
        second_term = self.gamma * np.dot(self.P_hat[current, action, :] / np.sum(self.P_hat[current, action, :]), max_next_q_values)
    
        # Compute the Q-value update
        update = self.R_hat[current, action] / self.count_matrix[current, action] + second_term
    
        # Update Q-value
        self.delta_Q.append(update - self.Q[current, action])
        self.Q[current, action] = update
    
        
    def update(self, state, action, reward, next_state):
        
        self.update_model(state, action, reward, next_state)
        self.update_q_value(state, action)
        
        # Store experience in replay buffer
        self.replay_buffer.append((state, action))
        
        # Sample from replay buffer for further updates
        for _ in range(self.k):
            # Randomly sample from replay buffer
            rand_experience = random.choice(self.replay_buffer)
            rand_state, rand_action = rand_experience
            
            self.update_q_value(rand_state, rand_action)
      
        
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            discr_state = self.discretize_state(state)
            return np.argmax(self.Q[self.encode_state(discr_state)])
    
    def decay_epsilon(self, episode):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def plot_delta_Q(self):
        plt.plot(self.delta_Q)
        plt.title('Q_value update step')
        plt.show()
        
    def plot_max_Q(self):
        max_Q_values = np.zeros((self.n_states[0], self.n_states[1]))
    
        for i in range(self.n_states[0]):
            for j in range(self.n_states[1]):
                state = (self.born_inf[0] + i * self.discr_step[0], self.born_inf[1] + j * self.discr_step[1])
                discr_state = self.discretize_state(state)
                max_Q_values[i][j] = np.max(self.Q[self.encode_state(discr_state)])
    
        plt.imshow(max_Q_values.T, origin='lower', extent=[self.born_inf[0], self.born_sup[0], self.born_inf[1], self.born_sup[1]])
        plt.colorbar(label='Max Q-value')
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.title('Max Q-values after learning')
        plt.show()
    

def run(args):
    
    # Create the environment
    env = gym.make('MountainCar-v0')
    # Create the DynaAgent
    dyna_agent = DynaAgent(decay=0.9, discr_step=[0.25, 0.05], env=env)

    # Train the agent
    n_episodes = 10001
    episode_rewards=[]
    episode_durations = []

    fig,ax=plt.subplots(1,2,figsize=(11,5))

    for episode in range(n_episodes):
        state,_ = env.reset()
        total_reward = 0
        solved=False
        done = False
        x=[state[0]]
        v=[state[1]]
        n_itr=0
        while not done:
            action = dyna_agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            dyna_agent.update(state, action, reward, next_state)
            solved=terminated
            done= terminated or truncated
            state = next_state
            total_reward += reward
            n_itr+=1
            x.append(state[0])  
            v.append(state[1])

        dyna_agent.decay_epsilon(episode)
        
        episode_rewards.append(total_reward) 
        episode_durations.append(n_itr)
        
        if (episode%2000==0):
            ax[0].plot(list(range(n_itr+1)),x, label=f'episode {episode}')
            ax[1].plot(x,v, label=f'episode {episode}')
            print(f'episode {episode} : solved? : {solved}')

    env.close()
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Position')
    ax[1].set_xlabel('Position')
    ax[1].set_ylabel('Velocity')
    plt.legend()
    plt.show()

    dyna_agent.plot_delta_Q()
    dyna_agent.plot_max_Q()

    plt.plot(episode_durations, label='episode durations')
    plt.plot(episode_rewards,label='episode rewards')
    plt.title('Duration and total reward for each episode' )
    plt.xlabel('Episodes')
    plt.legend()
    plt.savefig('dyna.png')

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

    run(args)

