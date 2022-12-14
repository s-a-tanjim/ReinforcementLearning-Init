'''
  https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/mountaincar.py
'''
import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers
import pickle

pos_space = np.linspace(-1.2, 0.6, 12)
vel_space = np.linspace(-0.07, 0.07, 20)

def get_state(observation):
    pos, vel =  observation
    pos_bin = int(np.digitize(pos, pos_space))
    vel_bin = int(np.digitize(vel, vel_space))

    return (pos_bin, vel_bin)

def max_action(Q, state, actions=[0, 1, 2]):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)

    return action

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000
    alpha = 0.1
    gamma = 0.99
    eps = 1.0

    action_space = [0, 1, 2]

    states = []
    for pos in range(21):
        for vel in range(21):
            states.append((pos, vel))

    Q = {}
    for s in states:
        for a in action_space:
            Q[s, a] = 0

    # pickle_in = open('mountaincar.pkl', 'rb')
    # Q = pickle.load(pickle_in)
    # env = wrappers.Monitor(env, "tmp/mountaincar",
                            # video_callable=lambda episode_id: True, force=True)

    n_games = 50000
    total_rewards = np.zeros(n_games)
    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()
        s = get_state(obs)
        
        
        while not done:
            a = np.random.choice([0,1,2]) if np.random.random() < eps else max_action(Q, s)
            obs_, reward, done, info = env.step(a)
            s_ = get_state(obs_)
            score += reward
            a_ = max_action(Q, s_)
            Q[s, a] = Q[s, a] + alpha*(reward + gamma*Q[s_, a_] - Q[s, a])
            s = s_
        total_rewards[i] = score
        eps = eps - 2/n_games if eps > 0.01 else 0.01

        if i % 100 == 0 and i > 0:
            print('episode ', i, 'score ', score, 'epsilon %.3f' % eps)

    mean_rewards = np.zeros(n_games)
    for t in range(n_games):
        mean_rewards[t] = np.mean(total_rewards[max(0, t-50):(t+1)])
    plt.plot(mean_rewards)
    plt.show()
    # plt.savefig('mountaincar.png')

    #f = open("mountaincar.pkl","wb")
    #pickle.dump(Q,f)
    #f.close()