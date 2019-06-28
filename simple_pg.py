import tensorflow as tf
import numpy as np
import gym
from gym.spaces import Discrete, Box
import nautlabs.shipperf as ns

def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network.
    with tf.name_scope('mlp'):
        for size in sizes[:-1]:
            x = tf.layers.dense(x, units=size, activation=activation, name='dense_'+ str(size))
        return tf.layers.dense(x, units=sizes[-1], activation=output_activation,name='mlp_final')

def train(env_name='gym_shipping:Shipping-v0', env_params=(2019,6, 2,0), hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name, start_datetime=(env_params))
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    with tf.name_scope('ph'):
        # make core of policy network
        obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32, name='observation')
        logits = mlp(obs_ph, sizes=hidden_sizes+[n_acts])

		# make action selection op (outputs int actions, sampled from policy)
        actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1, name='actions')

        # make loss function whose gradient, for the right data, is policy gradient
        weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
        act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
        action_masks = tf.one_hot(act_ph, n_acts)
        log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
        loss = -tf.reduce_mean(weights_ph * log_probs)

        # make train op
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = sess.run(actions, {obs_ph: obs.reshape(1,-1)})[0]

            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                # if len(batch_obs) > batch_size:
                #     break
                break

        # take a single policy gradient update step
        batch_loss, _ = sess.run([loss, train_op],
                                 feed_dict={
                                    obs_ph: np.array(batch_obs),
                                    act_ph: np.array(batch_acts),
                                    weights_ph: np.array(batch_weights)
                                 })
        return batch_loss, batch_rets, batch_lens, batch_obs, batch_acts

    def seeded_train():
        import nautlabs.shipperf as ns
        import gym_shipping.envs.shipping_env as se

        # make some empty lists for logging.
        batch_obs = []  # for observations
        batch_acts = []  # for actions
        batch_weights = []  # for R(tau) weighting in policy gradient
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()  # first obs comes from starting distribution
        done = False  # signal from environment that episode is over
        ep_rews = []  # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            # act = sess.run(actions, {obs_ph: obs.reshape(1, -1)})[0]
            currloc = obs[7:9]
            dest = obs[1:3]

            rh = ns.get_random_along_geodesic_from_loc_to_dest(currloc, dest)

            # Get the closes action number, e.g. say action 7
            act = se.get_nearest_action_from_rpm_heading(rh)

            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                # if len(batch_obs) > batch_size:
                #     break
                break

        # take a single policy gradient update step
        batch_loss, _ = sess.run([loss, train_op],
                                    feed_dict={
                                        obs_ph: np.array(batch_obs),
                                        act_ph: np.array(batch_acts),
                                        weights_ph: np.array(batch_weights)
                                    })
        return batch_loss, batch_rets, batch_lens, batch_obs, batch_acts

	# Run here
    # graph = tf.Graph()
    # with graph.as_default():
    best_rets = -1e9
    warm_up = 10
    saver = tf.train.Saver()
    # training loop
    for i in range(epochs):
        if i < warm_up:
            batch_loss, batch_rets, batch_lens, batch_obs, batch_acts = seeded_train()
            print('Seed epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
                    (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
        else:
            batch_loss, batch_rets, batch_lens, batch_obs, batch_acts = train_one_epoch()
            print('Train epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

        if np.mean(batch_rets) > best_rets:
            best_rets = np.mean(batch_rets)
            print('Saving best model thus far')
            saver.save(sess, 'models/tf_simple_pg.ckpt')
            ns.save_states_to_file(batch_obs, 'outputs/tf_op.csv')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='gym_shipping:Shipping-v0')
    parser.add_argument('--env_params', '--param', type=str, default=(2019, 6, 2, 0))
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, env_params=args.env_params, render=args.render, lr=args.lr)