import random
import sys
import gym
import tensorflow as tf
import numpy as np

learning_rate = 0.0001
momentum = 0.9
batch_size = 20

n_episode = 300
training_epsilon = 0.5
epsilon_decay = 0.99
test_epsilon = 0.01
memory_length = 1000
gamma = 0.9
max_step = 500

tf.reset_default_graph()

state = tf.placeholder(tf.float32, shape=[None, 4])
W1 = tf.get_variable('W1', shape=[4, 64])
b1 = tf.get_variable('b1', shape=[64])
W2 = tf.get_variable('W2', shape=[64, 64])
b2 = tf.get_variable('b2', shape=[64])
hidden = tf.nn.relu(tf.matmul(state, W1) + b1)
hidden = tf.nn.relu(tf.matmul(hidden, W2) + b2)
Q = tf.layers.dense(hidden, 2, kernel_initializer=tf.random_uniform_initializer)

actions = tf.placeholder(tf.float32, shape=[None, 2])

nextQ = tf.placeholder(tf.float32, shape=[None])
loss = tf.reduce_sum(tf.square(tf.reduce_sum(Q * actions, axis=1) - nextQ))
tf.summary.scalar('loss', loss)

optim_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)

init = tf.global_variables_initializer()

average_loss_var = tf.placeholder(tf.float32)
reward_var = tf.placeholder(tf.float32)
merged = tf.summary.merge([
    tf.summary.scalar('average loss', average_loss_var),
    tf.summary.scalar('reward', reward_var)    
])

saver = tf.train.Saver()

env = gym.make('CartPole-v1')

with tf.Session() as sess:
    try:
        saver.restore(sess, tf.train.latest_checkpoint('model'))
    except:
        print('No existing model. Starting with Xavier init.')
    train_writer = tf.summary.FileWriter('log')
    sess.run(init)
    if sys.argv[1] == 'train':
        print('Training...')

        memory = []

        for episode in range(n_episode):
            observation = env.reset()
            done = False
            total_reward = 0
            total_loss = 0.0
            total_step = 0
            while not done and total_step < max_step:
                # Perform action
                if random.random() < training_epsilon:
                    action = random.randint(0, 1)
                else:
                    action = np.argmax(sess.run(Q, feed_dict={state: [observation]})[0])
                next_observation, next_reward, next_done, _ = env.step(action)
                training_epsilon *= epsilon_decay
                # Append to memory
                if len(memory) >= memory_length:
                    memory.pop()
                memory.append((observation, action, next_reward, next_observation, next_done))
                observation = next_observation
                done = next_done
                total_reward += next_reward
                # Train Q-network
                # 1. Fetch minibatch
                minibatch = random.sample(memory, min(batch_size, len(memory)))
                # 2. Preprocess minibatch
                Xs = []
                ys = []
                mask = []
                for sample in minibatch:
                    training_observation, training_action, training_next_reward, training_next_observation, training_next_done = sample
                    y = training_next_reward
                    if not training_next_done:
                        y += gamma * np.max(sess.run(Q, feed_dict={state: [training_next_observation]}))
                    if training_action == 0:
                        mask.append([1, 0])
                    else:
                        mask.append([0, 1])
                    Xs.append(training_observation)
                    ys.append(y)
                Xs = np.array(Xs)
                ys = np.array(ys)
                mask = np.array(mask)
                # 3. Gradient Descent
                step_loss, _ = sess.run([loss, optim_op], feed_dict={state: Xs, nextQ: ys, actions: mask})
                total_loss += step_loss
                total_step += 1
            print('Episode', episode)
            print('Reward:', total_reward)
            print('Average loss:', total_loss / total_step)
            summary = sess.run(merged, feed_dict={average_loss_var: total_loss / total_step, reward_var: total_reward})
            train_writer.add_summary(summary, episode)
            saver.save(sess, 'model/model', global_step=episode)
        
        print('Training done!')
    
        input('Press Enter to start testing!')
    
    print('Testing...')
    for _ in range(5):
        observation = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            if random.random() < test_epsilon:
                action = random.randint(0, 1)
            else:
                q = sess.run(Q, feed_dict={state: [observation]})[0]
                print(q)
                action = np.argmax(q)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
        print('Total reward:', total_reward)
env.close()