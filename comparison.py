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
from dqn import DQNAgent
from dqnv2 import DQNAgentV2
from dyna import DynaAgent
from matplotlib import pyplot as plt
from dyna import plot_max_Q
import itertools

def run_dqn(agent, n_episodes,args, training, env, rnd, color_type, ax):
    X = np.arange(env.observation_space.low[0], env.observation_space.high[0], args.discr_pos)
    Y = np.arange(env.observation_space.low[1], env.observation_space.high[1], args.discr_vel)
    discr_space = torch.tensor(np.array(list(itertools.product(X, Y))), dtype=torch.float32).requires_grad_(False)
    with tqdm(total=n_episodes, desc=f"Episode 0/{n_episodes}") as pbar:
        target_count = 0
        finished = 0
        empty = True
        cumulative_auxiliary_reward = 0
        cumulative_env_reward = 0

        for episode in tqdm(range(n_episodes)):
            if training:
                obs, info = env.reset()
            else:
                obs, info = env.reset(seed=episode)
            x=[obs[0]]
            v=[obs[1]]
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
                if rnd:
                    env_reward*=args.env_reward
                else:
                    aux_reward = args.intermediate_reward*np.min((args.w_position*(next_obs[0]-obs[0])/(0.5+1.2) + args.w_velocity*(np.abs(next_obs[1]))/0.7, 1))

                if training:
                    if rnd:
                        loss, target_count, aux_reward,intrinsic_loss = agent.update(obs, action, env_reward, next_obs, batch_size=args.batch_size, target_count=target_count, terminal=terminated)
                    else:
                        loss, target_count = agent.update(obs, action, env_reward+aux_reward, next_obs, batch_size=args.batch_size, target_count=target_count, terminal=terminated)
                else: 
                    loss = None
                    episode_env_reward += env_reward
                
                if episode % args.snapshot_interval == 0 or episode == (n_episodes - 1):
                    x.append(obs[0])  
                    v.append(obs[1])

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
                if training:
                    agent.decay_epsilon()
                wandb.log({"train_loss": episode_loss, "epsilon": agent.epsilon, "episode_steps": t, "finished": finished, "episode_env_reward":episode_env_reward, "episode_aux_reward":episode_auxiliary_reward, "cumulative_env_reward":cumulative_env_reward, "cumulative_aux_reward":cumulative_auxiliary_reward})

            if training and ((episode // args.snapshot_interval >=1 and episode % args.snapshot_interval == 0)  or episode == (n_episodes - 1)):
                q_values = torch.max(agent.qnetwork(torch.tensor(discr_space)), axis=-1).values.detach().numpy().reshape(len(X),len(Y))
                max_q = plot_max_Q(q_values, episode, (args.discr_pos, args.discr_vel), env.observation_space.low, maximum=True)        
                wandb.log({"max_Q": wandb.Image(max_q,caption=f'Max Q-value at episode {episode}')})
                    
            if not training:
                finished += terminated
                cumulative_env_reward += episode_env_reward    
                wandb.log({"episode_steps": t, "finished": finished, "episode_env_reward":episode_env_reward, "cumulative_env_reward":cumulative_env_reward}) 
            if not training and ((episode // args.snapshot_interval >=1 and episode % args.snapshot_interval == 0)  or episode == (n_episodes - 1)):
                color_intensity = 0.9*(1-(episode+1)/n_episodes)
                cmap = plt.cm.get_cmap(color_type)
                color = cmap(color_intensity)                
                ax[0].plot(range(t+1),x, c=color, zorder = 1)
                ax[1].plot(range(t+1),v, c=color, zorder = 1)        
                
            if loss is not None:
                empty = False

def run_dyna(agent, n_episodes, snapshot_interval,discr_step, training, ax, env, color_type):
    with tqdm(total=n_episodes, desc=f"Episode 0/{n_episodes}") as pbar:
        finished = 0
        empty = True
        cumulative_env_reward = 0

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
                if training:
                    loss = agent.update(state, action, reward, next_state)
                else:
                    loss = None
                    episode_env_reward += reward

                done = terminated or truncated
                state = next_state
                if loss is not None:
                    episode_env_reward += reward
                    episode_loss+=loss           
                t+=1
                if episode % snapshot_interval == 0 or episode == (n_episodes - 1):
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
                wandb.log({"train_loss": episode_loss, "epsilon": agent.epsilon, "episode_steps": t, "finished": finished, "episode_env_reward":episode_env_reward, "cumulative_env_reward":cumulative_env_reward})

            if training and (episode // snapshot_interval >=1 and episode % snapshot_interval == 0)  or episode == (n_episodes - 1):
                max_q = plot_max_Q(agent.Q, episode, discr_step, agent.born_inf)             
                wandb.log({"max_Q": wandb.Image(max_q,caption=f'Max Q-value at episode {episode}')})
            if not training:
                finished += terminated
                cumulative_env_reward += episode_env_reward
                wandb.log({"episode_steps": t, "finished": finished, "episode_env_reward":episode_env_reward, "cumulative_env_reward":cumulative_env_reward})
            if not training and ((episode // args.snapshot_interval >=1 and episode % args.snapshot_interval == 0)  or episode == (n_episodes - 1)):
                color_intensity = 0.9*(1-(episode+1)/n_episodes)
                cmap = plt.cm.get_cmap(color_type)
                color = cmap(color_intensity)
                ax[0].plot(range(t+1),x, c=color, zorder = 1)
                ax[1].plot(range(t+1),v, c=color, zorder = 1)    

            if loss is not None:
                empty = False


        env.close()



def experiment(args, env):
    n_episodes = args.n_episodes
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    agent = DQNAgent(
        learning_rate=args.learning_rate,
        state_size=2,
        action_size=3,
        discount_factor=args.discount_factor,
        final_epsilon=args.final_epsilon,
        hidden_size=args.hidden_size,
        epsilon_decay=args.epsilon_decay,
        initial_epsilon=args.start_epsilon,
        replay_size=args.replay_size,
        dropout_rate=args.dropout_rate,
        target_network=args.target_network,
        weight_decay=args.weight_decay,
        target_network_update=args.target_network_update,
        alpha=args.alpha,
        amsgrad=args.amsgrad,
        device = 'cpu'

    )
    fig,ax=plt.subplots(1,2,figsize=(11,5))


    wandb.init(project='ANN-1', config=vars(args), name=f'DQN_training')
    run_dqn(agent, n_episodes, args, training=True, env=env, rnd=False, ax=ax, color_type='Reds')
    wandb.finish()

    wandb.init(project='ANN-1', config=vars(args), name=f'DQN_testing')
    agent.epsilon = 0
    # agent.qnetwork.eval()
    run_dqn(agent, 1_000, args, training=False, env=env, rnd=False, ax=ax, color_type='Reds')

    wandb.finish()

    agent = DQNAgentV2(
        learning_rate=args.learning_rate,
        state_size=2,
        action_size=3,
        discount_factor=args.discount_factor,
        final_epsilon=args.final_epsilon,
        hidden_size=args.hidden_size,
        epsilon_decay=args.epsilon_decay,
        initial_epsilon=args.start_epsilon,
        replay_size=args.replay_size,
        dropout_rate=args.dropout_rate,
        target_network=False,
        weight_decay=args.weight_decay,
        target_network_update=args.target_network_update,
        alpha=args.alpha,
        amsgrad=args.amsgrad,
        reward_hidden_size=args.reward_hidden_size,
        reward_factor=args.reward_factor,
        running_window=args.running_window,
        predictor_learning_rate=args.predictor_learning_rate,
        predictor_weight_decay=args.predictor_weight_decay
    )

    wandb.init(project='ANN-1', config=vars(args), name=f'DQN_training_v2')

    run_dqn(agent, n_episodes, args, training=True, env=env, rnd=True, ax=ax, color_type='Greens')
    wandb.finish()

    wandb.init(project='ANN-1', config=vars(args), name=f'DQN_testing_v2')
    agent.epsilon = 0
    # agent.qnetwork.eval()
    run_dqn(agent, 1_000, args, training=False, env=env, rnd=True,ax=ax, color_type='Greens')

    wandb.finish()

    wandb.init(project='ANN-1', config=vars(args), name=f'Dyna_training')
    agent = DynaAgent(decay=args.epsilon_decay, start_epsilon=args.start_epsilon, gamma=args.discount_factor, discr_step=(args.discr_pos, args.discr_vel), k=args.batch_size,alpha=args.alpha, replay_size=args.replay_size,env=env, min_epsilon=args.final_epsilon, init_val=args.init_val)
    run_dyna(agent, n_episodes,args.snapshot_interval, (args.discr_pos, args.discr_vel),training=True,ax=ax, color_type="Blues",env=env)
    wandb.finish()
    agent.epsilon = 0
    wandb.init(project='ANN-1', config=vars(args), name=f'Dyna_testing')
    run_dyna(agent, 1000,args.snapshot_interval, (args.discr_pos, args.discr_vel),training=False,ax=ax, color_type="Blues",env=env)

    ax[0].set_xlabel('Steps')
    ax[0].set_ylabel('Position')
    ax[1].set_xlabel('Steps')
    ax[1].set_ylabel('Velocity')
    wandb.log({"trajectories": wandb.Image(fig,caption=f'Trajectories')})



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Script for pretraining a language model")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--n_episodes", type=int, default=3_000)
    parser.add_argument("--start_epsilon", type=float, default=0.9)
    parser.add_argument("--final_epsilon", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--replay_size", type=int, default=10_000)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--target_network_update", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--intermediate_reward", type=float, default=0.999)
    parser.add_argument("--w_position", type=float, default=1.0)
    parser.add_argument("--w_velocity", type=float, default=1.0)
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--target_network", action="store_true")
    parser.add_argument("--discr_pos", type=float, default=0.1)
    parser.add_argument("--discr_vel", type=float, default=0.01)
    parser.add_argument("--init_val", type=float, default=0.01)    
    parser.add_argument("--snapshot_interval", type=int, default=500)
    parser.add_argument("--reward_hidden_size", type=int, default=128)
    parser.add_argument("--predictor_learning_rate", type=float, default=1e-3)
    parser.add_argument("--reward_factor", type=float, default=5)
    parser.add_argument("--running_window", type=int, default=1000)
    parser.add_argument("--predictor_weight_decay", type=float, default=1e-4)
    parser.add_argument("--env_reward", action="store_false")

    args = parser.parse_args()
    env = gym.make('MountainCar-v0')
    env.action_space.seed(args.seed)
    observation, info = env.reset(seed=args.seed)
    experiment(args, env)