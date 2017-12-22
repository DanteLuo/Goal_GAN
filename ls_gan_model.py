import tensorflow as tf
import numpy as np
import os


class LSGAN():
    def __init__(self):
        self.z_dim = 4
        self.h1_dim_g = 128
        self.h2_dim_g = 128

        self.goal_dim = 2
        self.h1_dim_d = 256
        self.h2_dim_d = 256

        self.scaling = 6
        self.iteration = 0

        self.X = tf.placeholder(tf.float64, shape=[None, self.goal_dim],name='GAN_X')
        self.Y = tf.placeholder(tf.float64, shape=[None, 1],name='GAN_Y')
        self.Z = tf.placeholder(tf.float64, shape=[None, self.z_dim],name='GAN_Z')

        with tf.variable_scope('G'):
            W1 = tf.get_variable(name='W1', shape=[self.z_dim, self.h1_dim_g], dtype=tf.float64,
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable(name='b1', shape=[self.h1_dim_g], dtype=tf.float64,
                                 initializer=tf.initializers.truncated_normal())
            W2 = tf.get_variable(name='W2', shape=[self.h1_dim_g, self.h2_dim_g], dtype=tf.float64,
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable(name='b2', shape=[self.h2_dim_g], dtype=tf.float64,
                                 initializer=tf.initializers.truncated_normal())
            W3 = tf.get_variable(name='W3', shape=[self.h2_dim_g, self.goal_dim], dtype=tf.float64,
                                 initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.get_variable(name='b3', shape=[self.goal_dim], dtype=tf.float64,
                                 initializer=tf.initializers.truncated_normal())
            self.theta_G = [W1, W2, W3, b1, b2, b3]

        with tf.variable_scope('D'):
            W1 = tf.get_variable(name='W1', shape=[self.goal_dim, self.h1_dim_d], dtype=tf.float64,
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable(name='b1', shape=[self.h1_dim_d], dtype=tf.float64,
                                 initializer=tf.initializers.truncated_normal())
            W2 = tf.get_variable(name='W2', shape=[self.h1_dim_d, self.h2_dim_d], dtype=tf.float64,
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable(name='b2', shape=[self.h2_dim_d], dtype=tf.float64,
                                 initializer=tf.initializers.truncated_normal())
            W3 = tf.get_variable(name='W3', shape=[self.h2_dim_d, 1], dtype=tf.float64,
                                 initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.get_variable(name='b3', shape=[1], dtype=tf.float64,
                                 initializer=tf.initializers.truncated_normal())
            self.theta_D = [W1, W2, W3, b1, b2, b3]

    def sample_z(self,m):
        return np.random.uniform(-1., 1., size=[m,self.z_dim])

    def generator(self,z):
        with tf.variable_scope('G'):
            h1 = tf.nn.relu(tf.matmul(z,self.theta_G[0])+self.theta_G[3])
            h2 = tf.nn.relu(tf.matmul(h1,self.theta_G[1])+self.theta_G[4])
            y = tf.nn.sigmoid(tf.matmul(h2,self.theta_G[2])+self.theta_G[5])
            goals = y*self.scaling
        return goals

    def discriminator(self,x):
        with tf.variable_scope('D'):
            h1 = tf.nn.relu(tf.matmul(x, self.theta_D[0]) + self.theta_D[3])
            h2 = tf.nn.relu(tf.matmul(h1, self.theta_D[1]) + self.theta_D[4])
            out = tf.matmul(h2, self.theta_D[2]) + self.theta_D[5]
        return out

    def build_model(self, lr=0.001, sess=None):
        self.G_sample = self.generator(self.Z)

        self.D_real = self.discriminator(self.X)
        self.D_fake = self.discriminator(self.G_sample)

        self.D_loss = 0.5 * (tf.reduce_mean(tf.multiply(self.Y,(self.D_real-1)**2)+
                                       tf.multiply((1-self.Y),(self.D_real+1)**2))+
                             tf.reduce_mean((self.D_fake+1)**2))
        self.G_loss = 0.5 * (tf.reduce_mean(self.D_fake**2))

        self.D_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.D_loss, var_list=self.theta_D)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.G_loss, var_list=self.theta_G)

        if sess == None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        # self.sess.run(tf.global_variables_initializer())

    def train(self, x_real, y_real, num_x_fake, num_iterations=200):
        for it in range(num_iterations):
            z_mb = self.sample_z(num_x_fake)

            _, D_loss_curr = self.sess.run([self.D_solver,self.D_loss],
                                            feed_dict={self.X: x_real,
                                                       self.Y: y_real,
                                                       self.Z: z_mb})
            _, G_loss_curr = self.sess.run([self.G_solver,self.G_loss],
                                           feed_dict={self.Z: z_mb})

            if self.iteration % 1000 == 0:
                print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'
                      .format(it, D_loss_curr, G_loss_curr))

                samples = self.sess.run(self.G_sample, feed_dict={self.Z: z_mb})
                print(samples)

            self.iteration += 1

    def sample_goals(self,num_goals):
        z_mb = self.sample_z(num_goals)
        goals = self.sess.run(self.G_sample,feed_dict={self.Z: z_mb})
        return goals

    def save_model(self, path=None):
        if path == None:
            path = os.getcwd()+"/model/"

        if not os.path.exists(path):
            os.makedirs(path)

        saver = tf.train.Saver()
        save_path = saver.save(sess=self.sess,save_path=path)
        print("Model saved in file: %s" % save_path)

    def close_session(self):
        self.sess.close()
