# Recurrent Attention Model

## Introduction

During last ten years, neurual network and deep learning have developed rapidly in various applications ranging from computer vision to natural language processing. With enormous improvement in computing, it is possible for people to train huge and deep neurual network to accomplish some specific tasks sush as image classification in Imagenet, image captioning by means of RNN, semantic segmentation, object detection, text generation and so on. There exist numerous diferent neurual network achitecures now. However traditional CNNS more or less face same problems: computing complexity, scalability, robusteness. 

At the same time, neurual network also was introduced into reinforcement learning and made a great coup in game playing. Milestones are
[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
and
[Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf).
These achievements made researchers consider the possibility of introducing reinforcement learning algorithms into CNN to add 'attention' to models.

This is the motivation of recurrent attention model, a mixture of CNN, RNN and REINFORCE algorithm. The original creative paper is [Recurrent Models of Visual Attention](https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf), which achieves great performence in MNIST dataset. This model can decrease computing hugely and ignore clutter in image.

I spent effort and time studying and inplementing this model in tensorflow.  This is what this repository is for.

## Model

Model from this paper is as follow: 
![model.png](https://github.com/bei1994/Recurrent-Attention-Model/blob/master/images/model.png)

Legends:

* Glimpse Sensor: given input images, glimpse locations and scale number to extract retina representation 
![rho expression](https://latex.codecogs.com/gif.latex?%5Crho%20%28x_t%2C%20l_%7Bt-1%7D%29).

* Glimpse Network ![glimpse network](https://latex.codecogs.com/gif.latex?f_g%28%5Ctheta%20_g%29): two fully connected layers to output glimpse representation given input images and glimpse locations.

* Action Network ![action network](https://latex.codecogs.com/gif.latex?f_a%28%5Ctheta%20_a%29): one fully connected layer doing image classification(only works at last step of LSTM).

* Location Network ![location network](https://latex.codecogs.com/gif.latex?f_l%28%5Ctheta%20_l%29):  one fully connected layer to act as actor in actor-critic algorithm to decide next glimpse location.

* Critic Network ![baseline network](https://latex.codecogs.com/gif.latex?f_b%28%5Ctheta%20_b%29): one fully connected layer to act as critic in actor-critic algorithm to decide next glimpse location.

* Core Network ![core network](https://latex.codecogs.com/gif.latex?f_h%28%5Ctheta%20_h%29): a LSTM network to output hidden states and cell states given glimpse representation inputs and previous hidden states.


## File List

```
.:
lib/model/ram.py
lib/helper/glimpse_gif.m
lib/helper/image_gif.m
```

## Implemenation

* The RAM mdoel is defined in [lib/model/ram.py](lib/model/ram.py).
* The gif generator is defined in [lib/helper/](lib/helper/).

Notes:

* Action network, LSTM network and glimpse network are trained through classification cross entropy error, not through location network and critic network. Thus, you need stopping gradient of location network and critic network when implementing.

* Location network is trained through [Actor-Critic Algorithm Based on Policy Gradient](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf). According to policy gradient expression, we can just treat it as a classic classification problem to train location network practically. Details can be found [here](https://www.youtube.com/watch?v=y8UPGr36ccI&t=0s&list=PLvSdMJxMoHLtRjW2QFp3mRfzU02TgzrSV&index=3) (Apologies for no english substitles in this video :( ).

* Critic network is trained through mean square error between reward and baseline. Baseline is value function ![value function](https://latex.codecogs.com/gif.latex?V%5E%5Cpi%28s%29). In this implementation reward is estimated by just one trajectory reward, which is not accurate though. But it is enough for MNIST dataset. For more complicated tasks, you should choose more fancier methods to estimate reward. These are some choices: 

   - [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
   - [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952?context=cs)
   - [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
   - [Parameter Space Noise for Exploration](https://arxiv.org/abs/1706.01905)
   - [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/pdf/1707.06887.pdf)
   - [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)
   - [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
   - [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
   - [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
 
 * To estimate reward, I use Monte Carlo methond to sample different trajectories.
 
 * I implement center MNIST data and translated MNIST data. Due to the time constraint, I don't experiment with cluttered MNIST data.
 
## Run the code
 
Clone [Recurrent-Attention-Model](https://github.com/bei1994/Recurrent-Attention-Model) from Github into your local machine. Running the following commands:

```
cd lib/model/
python3 ram.py --train=True --center=True
```

There are 3 supported model parameters configurations: 'center', 'translated' and 'custom'. You can set different hyperparameters to train your model using the same syntax in terminal. And you can also test your trained models. Details of hyperparameters can be found [here](lib/model/ram.py).

## Results

I experiment two different dataset: MNIST center dataset and MNIST translated dataset. After getting results, I use [code](lib/helper/) to generate whole image gif and glimpse gif for model testing process.For center MNIST dataset, training step is 100000, batch size is 32, glimpse number is 6, patch size is 8. The final valid error rate is 0.86%, test error rate is 0.84% which is much better than 1.29% in the original paper. For translated MNIST dataset, training step is 100000, batch size is 32, glimpse number is 6, patch size is 12, glimpse scale is 3. The final valid error rate is 1.24%, test error rate is 1.44% which is also much better than 1.84% in the original paper. The result data is in [file](). The gif are as follows:


#### Centered MNIST


#### Translated MNIST





 

