# Recurrent Attention Model

## Introduction

During last ten years, neurual network and deep learning have developed rapidly in various applications ranging from computer vision to natural language processing. With enormous improvement in computing, it is possible for people to train huge and deep neurual network to accomplish some specific tasks sush as image classification in Imagenet, image captioning by means of RNN, semantic segmentation, object detection, text generation and so on. There exist numerous diferent neurual network achitecures now. However traditional CNN more or less face same problems: computing complexity, scalability, robusteness. 

At the same time, neurual network also was introduced into reinforcement learning and made a great coup in game playing. Milestones are
[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
and
[Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf).
These achievements made researchers consider the possibility of introducing reinforcement learning algorithms into CNN to add 'attention' to models.

This is the motivation of recurrent attention model, a mixture of CNN, RNN and REINFORCE algorithm. The original creative paper is [Recurrent Models of Visual Attention](https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf), which achieves great performence. This model can decrease computing hugely and ignore clutter in image.

I spent effort and time studying and inplementing this model in tensorflow.  This is what this repository is for.
