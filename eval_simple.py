import tensorflow as tf
import numpy as np
import gym
from gym.spaces import Discrete, Box
import nautlabs.shipperf as ns

def eval_model(env_name='gym_shipping:Shipping-v0', env_params=(2019,6, 2,0), hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name, start_datetime=(env_params))
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('models/tf_simple_pg.ckpt.meta')
        saver.restore(sess, 'models/tf_simple_pg.ckpt')

        graph = tf.get_default_graph()
        # for op in graph.get_operations():
        #     #if op.type == 'Placeholder':
        #     print(op.name, op.type)

		#tensors are named scope/name:0
		# ops are named: scope/subscope/.../operation
        obs_ph = graph.get_tensor_by_name('ph/observation:0')
        actions = graph.get_tensor_by_name('ph/actions:0')

        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths
        info = {}               # for capturing the episode termination reason

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        obs, done, ep_rews = env.reset(), False, []

        while True:
                        # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            #act = sess.run(actions, {obs_ph: obs.reshape(1,-1)})[0]
            #sess.run(result, feed_dict={obs_ph: obs})
            #logit = sess.run([tanh], feed_dict={obs_ph: obs.reshape(1,-1)})
            act = sess.run([actions], feed_dict={obs_ph : obs.reshape(1,-1)})[0]

            obs, rew, done, info = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                break

        return batch_rets, batch_lens, batch_obs, batch_acts, info

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='gym_shipping:Shipping-v0')
    parser.add_argument('--env_params', '--param', type=str, default=(2018, 6, 7, 0))
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    best_return = -1e9
    best_index = 0
    for i in np.arange(50):
        br, bl, bo, ba, info = eval_model(env_name=args.env_name, env_params=args.env_params, render=args.render, lr=args.lr)
        if np.mean(br) > best_return:
            best_return = np.mean(br)
            best_index = i
            print('Evaluation: %3d \t return: %.3f \t ep_len: %.3f \t reason: %s' %
                    (i, np.mean(br), np.mean(bl), info['reason']))
            ns.save_states_to_file(bo, 'outputs/tf_val_simple.csv')

    print('Best return index: ', best_index)
