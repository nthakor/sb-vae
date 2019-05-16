from __future__ import absolute_import, division, print_function, unicode_literals
'''!pip install imageio
!pip install tensorflow-gpu==2.0.0-alpha0
!pip install tfp-nightly --upgrade'''
import tensorflow as tf
import tensorflow_probability as tfp
import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
from tensorflow_probability.python.internal import dtype_util
from IPython import display

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

# Normalizing the images to the range of [0., 1.]
train_images /= 255.
test_images /= 255.

# Binarization
train_images[train_images >= .5] = 1.
train_images[train_images < .5] = 0.
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.

TRAIN_BUF = 60000
BATCH_SIZE = 100

TEST_BUF = 10000

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

class CVAE(tf.keras.Model):
  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.inference_net = tf.keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
          tf.keras.layers.Conv2D(
              filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
          tf.keras.layers.Conv2D(
              filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
          tf.keras.layers.Flatten(),
          # No activation
          tf.keras.layers.Dense(2,activation='softplus'),
      ]
    )

    self.generative_net = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
          tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
          tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
          tf.keras.layers.Conv2DTranspose(
              filters=64,
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
          tf.keras.layers.Conv2DTranspose(
              filters=32,
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
          # No activation
          tf.keras.layers.Conv2DTranspose(
              filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
        ]
    )

  def sample(self, alpha=None,beta=None):
    if alpha is None:
        alpha = tf.ones(shape=(100, 1))
    if beta is None:
        beta = tf.ones(shape=(100,1))*3
    Beta = tfp.distributions.Beta(alpha,beta)
    vi=[]
    for _ in range(self.latent_dim-1):
        v = Beta.sample()
        vi.append(v)
    vi = tf.transpose(tf.squeeze(tf.convert_to_tensor(vi)))
    pi = tfp.bijectors.IteratedSigmoidCentered().forward(vi)
    return self.decode(pi, apply_sigmoid=True)

  def encode(self, x):
    a, b = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
    return a, b

  def reparameterize(self, a, b):
    U = tfp.distributions.Uniform(low=tf.zeros(a.shape),high=tf.ones(a.shape))
    vi = []
    for _ in range(self.latent_dim-1):
        u = U.sample()
        v = (1-u**(1/b))**(1/a)
        vi.append(v)
    vi = tf.transpose(tf.squeeze(tf.convert_to_tensor(vi)))
    pi = tfp.bijectors.IteratedSigmoidCentered().forward(vi)
    return pi

  def decode(self, pi, apply_sigmoid=False):
    logits = self.generative_net(pi)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs

    return logits

optimizer = tf.keras.optimizers.Adam(0.0003,beta_1=0.95,beta_2=0.999)

def beta_fn(a,b):
    return tf.math.exp(tf.math.lgamma(a)+tf.math.lgamma(b)-tf.math.lgamma(a+b))

def compute_loss(model, x):
    a, b = model.encode(x)
    pi = model.reparameterize(a, b)
    x_logit = model.decode(pi)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    
    alpha = tf.ones(a.shape)
    beta = tf.ones(a.shape)*3
    gamma = 0.5772156649*tf.ones(a.shape)
    
    kl = (a-alpha)/alpha *(-gamma-tf.math.digamma(b)-(1/b)) 
    kl+= tf.math.log(a*b) + tf.math.log(beta_fn(alpha,beta)) - ((b-1)/b) 
    kl+= (beta-1)*b*(1/(1+a*b))*beta_fn(1/a,b)
    kl+= (beta-1)*b*(1/(2+a*b))*beta_fn(2/a,b)
    kl+= (beta-1)*b*(1/(3+a*b))*beta_fn(3/a,b)
    kl+= (beta-1)*b*(1/(4+a*b))*beta_fn(4/a,b)
    kl+= (beta-1)*b*(1/(5+a*b))*beta_fn(5/a,b)
    kl+= (beta-1)*b*(1/(6+a*b))*beta_fn(6/a,b)
    kl+= (beta-1)*b*(1/(7+a*b))*beta_fn(7/a,b)
    kl+= (beta-1)*b*(1/(8+a*b))*beta_fn(8/a,b)
    kl+= (beta-1)*b*(1/(9+a*b))*beta_fn(9/a,b)
    kl+= (beta-1)*b*(1/(10+a*b))*beta_fn(10/a,b)
    
    return -tf.reduce_mean(logpx_z-kl)

def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

epochs = 300
latent_dim = 10
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_alpha = tf.ones(shape=(num_examples_to_generate, 1))
random_beta = tf.ones(shape=(num_examples_to_generate, 1))*3
model = CVAE(latent_dim)

def generate_and_save_images(model, epoch, test_alpha,test_beta):
    predictions = model.sample(test_alpha,test_beta)
    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0], cmap='gray')
      plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

generate_and_save_images(model, 0, random_alpha,random_beta)
ELBo_track = []
for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    gradients, loss = compute_gradients(model, train_x)
    apply_gradients(optimizer, gradients, model.trainable_variables)
  end_time = time.time()

  if epoch % 1 == 0:
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
      loss(compute_loss(model, test_x))
    elbo = -loss.result()
    ELBo_tracka.append(elbo)
    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, '
          'time elapse for current epoch {}'.format(epoch,
                                                    elbo,
                                                    end_time - start_time))
    generate_and_save_images(
        model, epoch, random_alpha,random_beta)

plt.plot(ELBo_track)
plt.savefig('elbo_track.png')

(_, _), (test_images, test_class) = tf.keras.datasets.mnist.load_data()
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

# Normalizing the images to the range of [0., 1.]
test_images /= 255.

# Binarization
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.

BATCH_SIZE = 100

TEST_BUF = 10000

test_dataset = tf.data.Dataset.from_tensor_slices((test_images,test_class)).shuffle(TEST_BUF).batch(BATCH_SIZE)

from sklearn.manifold import TSNE as tsne
i = 0
ds = np.empty([0,10])
cls = []
for x,y in test_dataset:
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    z = z.numpy()
    ds = np.concatenate((ds,z))
    cls.append(y)
label = tf.convert_to_tensor(cls).numpy().flatten().astype(int)
ts = tsne(2)
mds = ts.fit_transform(ds)
plt.scatter(mds[:,0],mds[:,1],s=0.3,c=label)

for i,x in enumerate(test_dataset):
    img = x
    if(i==5):
        break
predictions=img[0][:16]
fig = plt.figure(figsize=(4,4))

for i in range(predictions.shape[0]):
  plt.subplot(4, 4, i+1)
  plt.imshow(predictions[i, :, :, 0], cmap='gray')
  plt.axis('off')

# tight_layout minimizes the overlap between 2 sub-plots
plt.savefig('original_data.png')
plt.show()