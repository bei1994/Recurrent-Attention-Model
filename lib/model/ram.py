from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import rnn_decoder

import os
import sys
import logging
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append('../')
DATA_PATH = '/Users/lorenzostudyroom/Desktop/result/data/MNIST_data/'
SAVE_PATH = '/Users/lorenzostudyroom/Desktop/result/data/out/ram_centered/'
RESULT_PATH = '/Users/lorenzostudyroom/Desktop/result/data/out/test_image_centered/'

"""parse parameters from terminal."""
logging.getLogger().setLevel(logging.INFO)
flags = tf.flags
flags.DEFINE_bool("test", False, "Use trained models to test.")
flags.DEFINE_bool("train", False, "Train the RAM model.")
flags.DEFINE_bool("translate", False, "Use translated MNIST data.")
flags.DEFINE_bool("center", False, "Use center MINST data")

flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
flags.DEFINE_float("decay_factor", 0.97, "Learning rate decays by this much.")
flags.DEFINE_float("min_learning_rate", 1e-4, "Minimum learning rate.")
flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
flags.DEFINE_integer("num_steps", 100000, "Number of training steps.")
flags.DEFINE_integer("patch_base_size", 8, "Size of glimpse patch window.")
flags.DEFINE_integer("num_scales", 3, "Num of scales per glimpse.")
flags.DEFINE_integer("unit_pixel", 12, "Num of pixels for patch center to be most far.")
flags.DEFINE_integer("g_size", 128, "Size of theta_g^0.")
flags.DEFINE_integer("l_size", 128, "Size of theta_g^1.")
flags.DEFINE_integer("glimpse_output_size", 256, "Output size of Glimpse Network.")
flags.DEFINE_integer("cell_size", 256, "Size of LSTM cell.")
flags.DEFINE_integer("num_glimpses", 6, "Number of glimpses.")
flags.DEFINE_float("variance", 0.22, "Gaussian variance for Location Network.")
flags.DEFINE_integer("M", 10, "Monte Carlo sampling, see Eq(2).")
flags.DEFINE_integer("load", 100, "Load pretrained parameters with id.")
FLAGS = flags.FLAGS


def _weight_variable(shape):
  initial = tf.truncated_normal(shape=shape, stddev=0.01)
  return tf.Variable(initial)


def _bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)


def _log_likelihood(loc_means, locs, variance):
  """implementation of policy ascent.

  By calculating log likelihood to get gradient over theta of policy network.

  """

  loc_means = tf.stack(loc_means)  # [num_glimpses, batch_size, loc_dim]
  locs = tf.stack(locs)  # [num_glimpses, batch_size, loc_dim]
  gaussian = tf.distributions.Normal(loc=loc_means, scale=variance)
  log_prob = gaussian._log_prob(x=locs)  # [num_glimpses, batch_size, loc_dim]
  log_prob = tf.reduce_sum(log_prob, 2)  # [num_glimpses, batch_size]
  return tf.transpose(log_prob)  # [batch_size, num_glimpses]


class RetinaSensor(object):
  """RetinaSensor Network

  get retina representation ρ(x_t, l_t−1)

  Args:
    inputs means x_t: [batch_size, img_size*img_size]
    locs means l_t−1: [batch_size, 2]
    one scale means k=1
    img_size means current image size.
    patch_base_size means extraction patch base size


  """

  def __init__(self,
               config,
               is_translate=False):

    self.img_size = config.img_size
    self.patch_base_size = config.patch_base_size

    self.num_scales = config.num_scales
    self.unit_pixel = config.unit_pixel
    self.is_translate = is_translate
    self.trans_size = config.img_size

  def __call__(self, inputs, locs):
    # img: [batch_size, curr_height, curr_width, channels]
    with tf.name_scope('glimpse_sensor'):
      img = tf.reshape(inputs, [tf.shape(inputs)[0],
                                self.img_size,
                                self.img_size,
                                1])

      max_radius = int(self.patch_base_size * (2 ** (self.num_scales - 2)))
      inputs_pad = tf.pad(
          img,
          [[0, 0], [max_radius, max_radius], [max_radius, max_radius], [0, 0]],
          'CONSTANT')

      if self.is_translate:
        locs_adj = locs * 1.0 * self.unit_pixel / (self.trans_size / 2 + max_radius)
      else:
        locs_adj = locs * 1.0 * self.unit_pixel / (14 + max_radius)

      retina_reprsent = []
      for scale in range(0, self.num_scales):
        cur_patch_size = self.patch_base_size * (2 ** scale)  # current patch width
        # using extract_glimpse() function to get image glimpse
        # locs: [batch_size, 2]. between (-1, +1), normalized and centered
        # patch: [batch_size, glimpse_height, glimpse_width, channels]
        cur_patch = tf.image.extract_glimpse(input=inputs_pad,
                                             size=[cur_patch_size, cur_patch_size],
                                             offsets=locs_adj,
                                             centered=True,
                                             normalized=True,
                                             uniform_noise=True,
                                             name='glimpse_sensor',
                                             )

        # [batch_size, self.patch_base_size, self.patch_base_size, channels]
        cur_patch = tf.image.resize_images(images=cur_patch,
                                           size=[self.patch_base_size, self.patch_base_size],
                                           method=tf.image.ResizeMethod.BILINEAR,
                                           align_corners=False,
                                           )

        retina_reprsent.append(cur_patch)

      # [batch_size, self.patch_base_size, self.patch_base_size, channels*scale]
      self.retina_reprsent = tf.concat(retina_reprsent, axis=-1)
      # reshape retina_reprsent as 2D tensor: [batch_size, patch_base_size^2*scale]
      reshaped_retina_reprsent = tf.reshape(self.retina_reprsent,
                                            [tf.shape(locs)[0],
                                             self.patch_base_size ** 2 * self.num_scales])

      return reshaped_retina_reprsent


class GlimpseNetwork(object):
  """Glimpse network.

  Take glimpse location input and output features for RNN.

  Args:
    loc_dim: 2
    g_size means dimensionality of h_g. In this paper is 128.
    l_size means dimensionality of h_l. In this paper is 128.
    output_size means dimensionality of g_t. In this paper is 256.
    g_t: [batch_size, 256]

  """

  def __init__(self, config, is_translate=False):
    self.retina_sensor = RetinaSensor(config=config, is_translate=is_translate)

    # fully connected layer 1
    self.g1_w = _weight_variable((config.patch_base_size ** 2 * config.num_scales,
                                  config.g_size))
    self.g1_b = _bias_variable((config.g_size,))
    self.l1_w = _weight_variable((config.loc_dim, config.l_size))
    self.l1_b = _bias_variable((config.l_size,))

    # fully connected layer 2
    self.g2_w = _weight_variable((config.g_size, config.glimpse_output_size))
    self.g2_b = _bias_variable((config.glimpse_output_size,))
    self.l2_w = _weight_variable((config.l_size, config.glimpse_output_size))
    self.l2_b = _bias_variable((config.glimpse_output_size,))

  def __call__(self, inputs, locs):
    rhos = self.retina_sensor(inputs, locs)

    h_g = tf.nn.relu(tf.nn.xw_plus_b(rhos, self.g1_w, self.g1_b))
    linear_h_g = tf.nn.xw_plus_b(h_g, self.g2_w, self.g2_b)
    h_l = tf.nn.relu(tf.nn.xw_plus_b(locs, self.l1_w, self.l1_b))
    linear_h_l = tf.nn.xw_plus_b(h_l, self.l2_w, self.l2_b)
    g_t = tf.nn.relu(linear_h_g + linear_h_l)

    return g_t


class LocationNetwork(object):
  """

  This is one fully connected network only learns means of (x,y)
  Then use two component Gaussian model to sample next location.
  This is actor network(policy function)in deep reinforcement learning to
  produce actions for every observed state. The input is h_t as current state.

  Args:
    loc_dim: 2
    rnn_output_size means dimensionality of h_t
    locs: [batch_size, 2]
    mean: [batch_size, 2]
    cell_output: [batch_size, cell_size]


  """

  def __init__(self, config, rnn_output_size, is_sampling=False):
    self.loc_dim = config.loc_dim
    self.variance = config.variance
    self.w = _weight_variable((rnn_output_size, config.loc_dim))
    self.b = _bias_variable((config.loc_dim,))
    self.is_sampling = is_sampling

  def __call__(self, cell_output):
    mean = tf.nn.xw_plus_b(cell_output, self.w, self.b)
    mean = tf.clip_by_value(mean, -1., 1.)
    mean = tf.stop_gradient(mean)

    if self.is_sampling:
      locs = mean + tf.random_normal(
          shape=[tf.shape(cell_output)[0], self.loc_dim],
          stddev=self.variance)
      locs = tf.clip_by_value(locs, -1., 1.)
    else:
      locs = mean

    locs = tf.stop_gradient(locs)
    return locs, mean


class RecurrentAttentionModel(object):
  """recurrent attention network

  build up a whole recurrent attention network.


  """

  def __init__(self,
               config,
               decay_step,
               is_training=False,
               is_translate=False):

    # image means feed-in images: batch_size * img_size^2
    # label: labels of images not one hot representation
    self.config = config
    self.decay_step = decay_step
    self.is_training = is_training
    self.is_translate = is_translate

    # input data placeholders
    self.image = tf.placeholder(tf.float32, [None, config.input_img_size * config.input_img_size])
    self.label = tf.placeholder(tf.int64, [None])

    # translate MNIST data if need
    if self.is_translate:
      img = tf.reshape(self.image, [tf.shape(self.image)[0],
                                    config.input_img_size,
                                    config.input_img_size,
                                    1])

      self.proc_image = self._translate_image(img)
      # reshape into 2D tensor: [batch_size, img_size^2]
      # new_img_size = self.proc_image.get_shape().as_list()
      # print(new_img_size)
      self.proc_image = tf.reshape(self.proc_image, [tf.shape(self.image)[0],
                                                     config.img_size * config.img_size])
    else:
      self.proc_image = self.image

    self.global_step = tf.Variable(0, trainable=False)

    # define learning rate
    self.learning_rate = tf.maximum(tf.train.exponential_decay(config.learning_rate,
                                                               self.global_step,
                                                               decay_step,
                                                               config.decay_factor,
                                                               staircase=True),
                                    config.min_learning_rate)

    cell = BasicLSTMCell(config.cell_size)

    with tf.variable_scope('GlimpseNetwork'):
      glimpse_network = GlimpseNetwork(config=config,
                                       is_translate=self.is_translate)

    with tf.variable_scope('LocationNetwork'):
      location_network = LocationNetwork(config=config,
                                         rnn_output_size=cell.output_size,
                                         is_sampling=self.is_training)

    # Core LSTM Network
    with tf.name_scope('RNN'):
      batch_size = tf.shape(self.image)[0]
      init_locs = tf.random_uniform(shape=[batch_size, config.loc_dim],
                                    minval=-1,
                                    maxval=1)
      init_state = cell.zero_state(batch_size, tf.float32)

      # transfer glimpse network output into 2D list
      # rnn_inputs: 3D list [[batch_size, 256], ...]
      init_glimpse = glimpse_network(self.proc_image, init_locs)
      rnn_inputs = [init_glimpse]
      rnn_inputs.extend([0] * config.num_glimpses)

      self.locs, loc_means, self.retina_reprsent = [], [], []

      def loop_function(prev, _):
        loc, loc_mean = location_network(prev)
        self.locs.append(loc)
        loc_means.append(loc_mean)
        glimpse = glimpse_network(self.proc_image, loc)
        self.retina_reprsent.append(glimpse_network.retina_sensor.retina_reprsent)
        return glimpse

      rnn_outputs, _ = rnn_decoder(rnn_inputs,
                                   init_state,
                                   cell,
                                   loop_function=loop_function)

    # Time independent baselines is actually state value function V(s).
    # This is network part to estimate state value function V(s) given actor pi.
    # This network is parallel to actor network and they have the same input h_t
    # as current state.
    # The result is a scalar.
    with tf.variable_scope('Baseline'):
      baseline_w = _weight_variable((cell.output_size, 1))
      baseline_b = _bias_variable((1,))
    # core net does not trained through baseline loss
    stopped_rnn_outputs = tf.stop_gradient(rnn_outputs)
    baselines = []
    for step_id in range(config.num_glimpses):
      output = stopped_rnn_outputs[step_id + 1]
      baseline = tf.nn.xw_plus_b(output, baseline_w, baseline_b)  # [batch_size, 1]
      baseline = tf.squeeze(baseline)  # [batch_size]
      baselines.append(baseline)  # [[batch_size]*num_glimpses]

    baselines = tf.stack(baselines)  # [num_glimpses, batch_size]
    baselines = tf.transpose(baselines)  # [batch_size, num_glimpses]

    # Classification. Take the last step only.
    rnn_last_output = rnn_outputs[-1]
    with tf.variable_scope('Classification'):
      logit_w = _weight_variable((cell.output_size, config.num_classes))
      logit_b = _bias_variable((config.num_classes,))
    logits = tf.nn.xw_plus_b(rnn_last_output, logit_w, logit_b)  # [batch_size, num_classes]
    self.prediction = tf.argmax(logits, 1)  # [batch_size]
    self.softmax = tf.nn.softmax(logits)

    if is_training:

      # classification loss
      self.cl_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=self.label,
          logits=logits))

      # RL reward
      reward = tf.cast(tf.equal(self.prediction, self.label), tf.float32)  # [batch_size]
      rewards = tf.expand_dims(reward, 1)  # [batch_size, 1]
      rewards = tf.tile(rewards, (1, config.num_glimpses))  # [batch_size, num_glimpses]
      self.reward = tf.reduce_mean(reward)  # a scalar

      # core net does not trained through REINFORCE reward
      # actor net does not trained through state value net
      advantages = rewards - tf.stop_gradient(baselines)  # [batch_size, num_glimpses]
      log_prob = _log_likelihood(loc_means, self.locs, config.variance)  # [batch_size, num_glimpses]
      REINFORCE_reward = tf.reduce_mean(log_prob * advantages)  # gradient over theta of policy
      self.advantage = tf.reduce_mean(advantages)  # a scalar

      # baseline loss to train state value function V(s)
      # self.baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
      self.baselines_mse = tf.losses.mean_squared_error(labels=rewards,
                                                        predictions=baselines)

      # hybrid loss: classification loss, RL reward, baseline loss
      self.loss = -REINFORCE_reward + self.cl_loss + self.baselines_mse

      var_list = tf.trainable_variables()
      gradients = tf.gradients(self.loss, var_list)
      # collect histogram of gradient
      # [tf.summary.histogram('gradient/' + var.name, grad,
      #                       collections=[tf.GraphKeys.SUMMARIES])
      #  for grad, var in zip(gradients, var_list)]

      clipped_gradients, norm = tf.clip_by_global_norm(gradients, config.max_gradient_norm)
      self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
          zip(clipped_gradients, var_list), global_step=self.global_step)

    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=99999999)

  def _translate_image(self, input_im):
    """
    Generate translate images for 60*60 translated MNIST.

    Args:
        input_im: [batch_size, in_height, in_width, channel]

    """
    with tf.name_scope('ImageTranslation'):
      trans_offset = int((self.config.img_size - 28) / 2)
      pad_im = tf.pad(
          input_im,
          paddings=tf.constant(
              [[0, 0],
               [trans_offset, trans_offset],
               [trans_offset, trans_offset],
               [0, 0]]),
          mode='CONSTANT',
          name='pad_im',
          constant_values=0
      )

      batch_size = tf.shape(input_im)[0]
      translations = tf.random_uniform(
          (batch_size, 2), minval=-trans_offset, maxval=trans_offset)  # [batch_size, 2]
      trans_im = tf.contrib.image.translate(
          pad_im,
          translations,
          interpolation='NEAREST',
          name=None
      )

      return trans_im


class Trainer(object):
  def __init__(self, model, train_data):
    self.model = model
    self.config = model.config
    self.train_data = train_data

    self.train_op = model.train_op
    self.loss = model.loss
    self.cl_loss = model.cl_loss
    self.reward = model.reward
    self.advantage = model.advantage
    self.baselines_mse = model.baselines_mse
    self.lr_rate = model.learning_rate

    self.softmax = model.softmax
    self.sample_locs = model.locs
    self.predics = model.prediction

  def train_epoch(self, sess, saver, name):
    self.model.is_training = True
    for step in range(self.config.num_steps):
      # images:[batch_size, 784]
      images, labels = self.train_data.train.next_batch(self.config.batch_size)
      # Duplicate M times
      images = np.tile(images, [self.config.M, 1])  # [batch_size*FLAGS.M, 784]
      labels = np.tile(labels, [self.config.M])  # [batch_size*FLAGS.M]

      output_feed = [self.train_op, self.loss, self.cl_loss, self.reward,
                     self.advantage, self.baselines_mse, self.lr_rate]

      _, loss, cl_loss, reward, advantage, baselines_mse, lr = sess.run(
          output_feed,
          feed_dict={self.model.image: images, self.model.label: labels})

      if step and step % 100 == 0:
        logging.info(
            'step {}: lr = {:3.6f}\tloss = {:3.4f}\tcl_loss = {:3.4f}\treward = {:3.4f}\tadvantage = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
                step, lr, loss, cl_loss, reward, advantage, baselines_mse))

        saver.save(sess,
                   '{}ram-{}-mnist-step-{}'
                   .format(SAVE_PATH, name, self.config.num_glimpses),
                   global_step=(step // 100))

      # validate and test
      if step and step % self.model.decay_step == 0:
        for dataset in [self.train_data.validation, self.train_data.test]:
          steps_per_epoch = dataset.num_examples // self.config.batch_size
          correct_num = 0
          num_samples = steps_per_epoch * self.config.batch_size
          for test_step in range(steps_per_epoch):
            images, labels = dataset.next_batch(self.config.batch_size)
            labels_backup = labels
            # Duplicate M times
            images = np.tile(images, [self.config.M, 1])
            labels = np.tile(labels, [self.config.M])
            softmax = sess.run(self.softmax,
                               feed_dict={
                                   self.model.image: images,
                                   self.model.label: labels
                               })
            # softmax: [FLAGS.M, batch_size, num_classes]
            softmax = np.reshape(softmax, [self.config.M, -1, 10])
            softmax = np.mean(softmax, 0)  # [batch_size, num_classes]
            prediction = np.argmax(softmax, 1).flatten()  # [batch_size]
            correct_num += np.sum(prediction == labels_backup)
          accuracy = correct_num / num_samples
          if dataset == self.train_data.validation:
            logging.info('valid accuracy = {}'.format(accuracy))
          else:
            logging.info('test accuracy = {}'.format(accuracy))

    # if summary_writer is not None:
    #   s = tf.Summary()
    #   s.value.add(tag='train_loss', simple_value=loss)
    #   s.value.add(tag='train_accuracy', simple_value=accuracy)
    #   summary_writer.add_summary(s, step)
    #   summary_writer.add_summary(cur_summary, step)

  def test_batch(self, sess, batch_data, unit_pixel, size, scale, save_path=''):
    def draw_bbx(ax, x, y):
      rect = patches.Rectangle(
          (x, y), cur_size, cur_size, edgecolor='r', facecolor='none', linewidth=2)
      ax.add_patch(rect)

    self.model.is_training = False
    test_im, test_lb = batch_data.next_batch(self.config.batch_size)
    loc_list, pred, input_im, glimpses = sess.run(
        [self.sample_locs, self.predics, self.model.proc_image,
         self.model.retina_reprsent],
        feed_dict={self.model.image: test_im,
                   self.model.label: test_lb,
                   })

    # reshape input_im into 4D: [batch_size, curr_height, curr_width, channel]
    batch_size = input_im.shape[0]
    input_im = np.reshape(input_im, [batch_size,
                                     self.config.img_size,
                                     self.config.img_size,
                                     1])

    pad_radius = size * (2 ** (scale - 2))  # size: glimpse base size
    im_size = self.config.img_size
    loc_list = np.clip(np.array(loc_list), -1.0, 1.0)
    # print(type(im_size))
    # print(type(pad_radius))
    # print(type(unit_pixel))
    # print(type(input_im))
    # print(input_im.shape)
    loc_list = loc_list * 1.0 * unit_pixel / (im_size / 2 + pad_radius)
    loc_list = (loc_list + 1.0) * 1.0 / 2 * (im_size + pad_radius * 2)
    offset = pad_radius
    correct_num = np.sum(pred == test_lb)

    # print prediction results
    print("true labels: ", test_lb)
    print("pred labels: ", pred)
    print("predict accuracy: {:.4f}".format(correct_num / self.config.batch_size))

    for step_id, cur_loc in enumerate(loc_list):
      im_id = 0
      glimpse = glimpses[step_id]
      for im, loc, cur_glimpse in zip(input_im, cur_loc, glimpse):
        im_id += 1
        fig, ax = plt.subplots(1)
        ax.imshow(np.squeeze(im), cmap='gray')
        for scale_id in range(0, scale):
          cur_size = size * 2 ** scale_id
          side = cur_size * 1.0 / 2
          x = loc[1] - side - offset
          y = loc[0] - side - offset
          draw_bbx(ax, x, y)

        for i in range(0, scale):
          scipy.misc.imsave(
              os.path.join(save_path, 'im_{}_glimpse_{}_step_{}.png').format(im_id, i, step_id),
              np.squeeze(cur_glimpse[:, :, i]))
        plt.savefig(os.path.join(
            save_path, 'im_{}_step_{}_pred_{}.png').format(im_id, step_id, pred[im_id - 1]))
        plt.close(fig)

    self.model.is_training = True


class CenterConfig(object):
  """config for center MNIST."""

  input_img_size = 28  # MNIST: 28 * 28
  img_size = 28  # don't translate
  loc_dim = 2   # (x,y)
  num_classes = 10
  learning_rate = 1e-3
  decay_factor = 0.97
  min_learning_rate = 1e-4
  max_gradient_norm = 5.0
  batch_size = 32
  num_steps = 100000
  patch_base_size = 8
  num_scales = 1
  unit_pixel = 12
  g_size = 128
  l_size = 128
  glimpse_output_size = 256
  cell_size = 256
  num_glimpses = 6
  variance = 0.22
  M = 10  # Monte Carlo sampling


class TranslatedConfig(object):
  """config for translated MNIST."""

  input_img_size = 28  # MNIST: 28 * 28
  img_size = 60  # translated size: 60 * 60
  loc_dim = 2   # (x,y)
  num_classes = 10
  learning_rate = 1e-3
  decay_factor = 0.97
  min_learning_rate = 1e-4
  max_gradient_norm = 5.0
  batch_size = 32
  num_steps = 100000
  patch_base_size = 12
  num_scales = 3
  unit_pixel = 26
  g_size = 128
  l_size = 128
  glimpse_output_size = 256
  cell_size = 256
  num_glimpses = 6
  variance = 0.22
  M = 10  # Monte Carlo sampling


class CustomConfig(object):
  """custom config from FLAGs."""

  input_img_size = 28  # MNIST: 28 * 28
  img_size = 60  # translated size: 60 * 60
  loc_dim = 2   # (x,y)
  num_classes = 10

  learning_rate = FLAGS.learning_rate
  decay_factor = FLAGS.decay_factor
  min_learning_rate = FLAGS.min_learning_rate
  max_gradient_norm = FLAGS.max_gradient_norm
  batch_size = FLAGS.batch_size
  num_steps = FLAGS.num_steps
  patch_base_size = FLAGS.patch_base_size
  num_scales = FLAGS.num_scales
  unit_pixel = FLAGS.unit_pixel
  g_size = FLAGS.g_size
  l_size = FLAGS.l_size
  glimpse_output_size = FLAGS.glimpse_output_size
  cell_size = FLAGS.cell_size
  num_glimpses = FLAGS.num_glimpses
  variance = FLAGS.variance
  M = FLAGS.M  # Monte Carlo sampling


def get_config():
  """Get model config."""
  config = None
  if FLAGS.center:
    name = 'centered'
    config = CenterConfig()
  elif FLAGS.translate:
    name = 'translated'
    config = TranslatedConfig()
  else:
    name = 'custom'
    FLAGS.translate = True
    config = CustomConfig()

  return config, name


def main(_):
  mnist = input_data.read_data_sets(DATA_PATH, one_hot=False)
  config, name = get_config()
  decay_step = mnist.train.num_examples // config.batch_size
  ram = RecurrentAttentionModel(config=config,
                                decay_step=decay_step,
                                is_training=True,
                                is_translate=FLAGS.translate)

  trainer = Trainer(ram, mnist)
  saver = tf.train.Saver(max_to_keep=99999999)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if FLAGS.train:
      trainer.train_epoch(sess, saver, name)

    if FLAGS.test:
      saver.restore(sess,
                    '{}ram-{}-mnist-step-6-{}'
                    .format(SAVE_PATH, name, FLAGS.load))

      batch_data = mnist.train
      trainer.test_batch(
          sess,
          batch_data,
          unit_pixel=config.unit_pixel,
          size=config.patch_base_size,
          scale=config.num_scales,
          save_path=RESULT_PATH)


if __name__ == "__main__":
  tf.app.run()
