import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data as data
mnist = data.read_data_sets("TheMNIST/", one_hot=True)


class my_settings():
    def __init__(self, kargs):
        # Default settings
        settings = {
            'model_path' : os.path.join(os.getcwd(), 'my_rbm.ckpt'),
            'learning_rate' : 0.001,
            'cd_k' : 1,
            'n_hid' : 20,
            'n_label' : 10,
            'gauss' : True,
            'stddev' : .1,
            'rand_seed' : 123,
            'persistent' : False,
            'epochs' : 4000,
            'batch_size' : 200,
            'regtype' : 'L2',
            'regcoef' : 0
        }
        self.kargs = kargs
        settings.update(kargs)
        for key in settings.keys():
            setattr(self, key, settings[key])

    def _sample_prob(self, prob, rand):
        return tf.nn.relu(tf.sign(prob-rand))

    def __apply_regularization(self, theta):
        if self.regtype == 'L2':
            regterm = tf.nn.l2_loss(theta)
        elif self.regtype == 'L1':
            regterm = tf.reduce_sum(tf.abs(theta))
        else:
            regterm = 0
        return tf.abs(theta - self.regcoef * regterm)

    def _apply_regularization(self, thetas):
        res = []
        for theta in thetas:
            res.append(self.__apply_regularization(theta))
        return res

class Dataset():
    """docstring for Dataset"""
    def __init__(self, dataset, labels=None):
        if labels != None:
            dataset = np.concatenate([dataset, labels], -1)
        self.n_examples = dataset.shape[0]
        self.n_features = dataset.shape[-1]
        self.dataset = dataset

    def gen_batches(self, batch_size):
        dataset = np.random.permutation(self.dataset)
        if batch_size > 0:
            for i in range(0, self.n_examples, batch_size):
                yield dataset[i:i+batch_size]
        else:
            yield dataset
        
class RBM(my_settings):
    """docstring for RBM"""
    def __init__(self, **kargs):
        super(RBM, self).__init__(kargs)
        self.sess = tf.Session()
        tf.set_random_seed(self.rand_seed)
        np.random.seed(self.rand_seed)
        self.graph_built = False
        
    def _create_variables_placeholders(self):
        self.W = tf.Variable(tf.truncated_normal(shape=[self.n_vis, self.n_hid], stddev=self.stddev))
        self.b = tf.Variable(tf.constant(self.stddev, shape=[self.n_vis]))
        self.c = tf.Variable(tf.constant(self.stddev, shape=[self.n_hid]))
        self.input_data = tf.placeholder(tf.float32, shape=[None, self.n_vis])
        self.labels = tf.placeholder(tf.float32, shape=[None, self.n_label])
        self.hrand = tf.placeholder(tf.float32, shape=[None, self.n_hid])
        self.vrand = tf.placeholder(tf.float32, shape=[None, self.n_vis])

    def _create_feed_dict(self, data=None, labels=None):

        if data==None:
            data = np.zeros((1, self.n_vis))
        self.m_data = data.shape[0]
        if labels==None:
            labels = np.array([0]*self.n_label).reshape(1, -1)
        else:
            labels = np.array(labels).reshape(1, -1)

        return {
            self.input_data : data,
            self.labels : labels,
            self.hrand : np.random.rand(self.m_data, self.n_hid),
            self.vrand : np.random.rand(self.m_data, self.n_vis)
        }
    
    def _run_train_step(self, dataset, validation=None):

        for batch in dataset.gen_batches(self.batch_size):
            self.sess.run(self.update_theta, feed_dict=self._create_feed_dict(batch))

        if validation != None:
            energy = self.sess.run(self.energy, feed_dict=self._create_feed_dict(validation.dataset))
            return energy

    def build_graph(self):

        self._create_variables_placeholders()
        self.sess.run(tf.global_variables_initializer())

        self.h_gen = tf.add(self.c, tf.matmul(self.labels, self.W[-self.n_label:, ...]))
        self._generate = tf.add(self.b[:-self.n_label], tf.matmul(self.h_gen, tf.transpose(self.W[:-self.n_label, ...])))

        v0_state = self.input_data
        h0_prob, h0_state = self.infer_hid_from_vis(v0_state)

        hK_prob, hK_state = h0_prob, h0_state
        vK_prob, vK_state = self.infer_vis_from_hid(hK_prob)

        self._encode, self._reconstruct = hK_prob, vK_prob

        for _ in range(self.cd_k-1):
            hK_prob, hK_state = self.infer_hid_from_vis(vK_prob)
            vK_prob, vK_state = self.infer_vis_from_hid(hK_prob)

        def compute_association(h_prob, v_state):
            return tf.matmul(tf.transpose(h_prob), v_state)

        delta_W = compute_association(h0_prob, v0_state) - compute_association(hK_prob, vK_prob)
        delta_b = tf.reduce_mean(v0_state - vK_prob, axis=0)
        delta_c = tf.reduce_mean(h0_prob - hK_prob, axis=0)

        delta_theta = [(delta*self.learning_rate) for delta in [delta_W, delta_b, delta_c]]
        delta_theta = self._apply_regularization(delta_theta)

        self.update_theta = []
        theta = [self.W, self.b, self.c]

        for variable, delta in zip(theta, delta_theta):
            update_step = variable.assign_add(tf.transpose(delta))
            self.update_theta.append(update_step)

        tmp = tf.matmul(v0_state, tf.reshape(self.b, [-1, 1]))
        tmp += tf.matmul(h0_state, tf.reshape(self.c, [-1, 1]))
        tmp += tf.matmul(tf.matmul(h0_state, tf.transpose(self.W)), tf.transpose(v0_state))
        self.energy = tf.reduce_mean(tmp)

    def infer_vis_from_hid(self, hid):

        activation = tf.add(self.b, tf.matmul(hid, tf.transpose(self.W)))
        if self.gauss:
            v_prob = tf.truncated_normal(shape=tf.shape(activation), mean=activation, stddev=self.stddev)
        else:
            v_prob = tf.nn.sigmoid(activation)
        v_state = self._sample_prob(v_prob, self.vrand)

        return v_prob, v_state

    def infer_hid_from_vis(self, vis):

        h_prob = tf.nn.sigmoid(tf.add(self.c, tf.matmul(vis, self.W)))
        h_state = self._sample_prob(h_prob, self.hrand)

        return h_prob, h_state

    def fit(self, dataset, labels=None, validation=None):

        if not isinstance(dataset, Dataset):
            dataset = Dataset(dataset, labels)

        validation_label = validation[1]
        validation = validation[0]

        if (validation != None) and not isinstance(validation, Dataset):
            validation = Dataset(validation, validation_label)

        self.n_vis = dataset.n_features
        self.build_graph()
        self.saver = tf.train.Saver()

        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            if validation == None:
                self._run_train_step(dataset)
            else:
                energy = self._run_train_step(dataset, validation)
                pbar.set_description("Energy: {}".format(energy))
        print('Model saved in ', self.save())

    def encode(self, data):

        data = np.array(data)

        return self.sess.run(self._encode,
            feed_dict=self._create_feed_dict(data))

    def reconstruct(self, data, step=1):

        pbar = tqdm(range(step))
        data = np.array(data).reshape(1, -1)
        for _ in pbar:
            data = self.sess.run(self._reconstruct,
                feed_dict=self._create_feed_dict(data))

        return data

    def generate(self, label):
        self.n_label = len(label)
        label = np.array(label).reshape(1, -1)
        generated = self.sess.run(self._generate, 
            feed_dict=self._create_feed_dict(labels=label))
        print(generated.shape)
        plt.imshow(generated.reshape(28, 28))
        plt.show()


    def save(self):

        return self.saver.save(self.sess, self.model_path)

    def load(self, n_vis=784, path=None):

        if path == None:
            path = self.model_path
        self.n_vis = n_vis
        self.build_graph()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, path)

    def close(self):

        self.sess.close()

    def display_weight(self, n_vis=784):

        if not isinstance(self.sess, tf.Session):
            self.load(n_vis)
        weights = tf.transpose(self.W).eval(session=self.sess)
        for i, weight in enumerate(weights):
            disp_weight = weight.reshape(28, 28)
            print(disp_weight)
            plt.imshow(disp_weight)
            plt.imsave(str(i)+'.jpg', disp_weight)
            plt.show()

def mnist_images():

    return mnist.train.images

def mnist_labels():

    return mnist.train.labels.astype(np.float32)


def test():
    rbm = RBM()
    #rbm.fit(mnist_images(), mnist_labels())
    #rbm.load(784)
    rbm.fit(mnist_images(), validation=[mnist.test.images, mnist.test.labels], labels=mnist_labels())
    #rbm.display_weight()
    label = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for i in range(10):
        rbm.generate(np.roll(label, i))
    
    rbm.close()

if __name__ == "__main__":
    test()