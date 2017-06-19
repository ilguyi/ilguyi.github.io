---
layout: post
title: "Self-Normalizing Neural Networks"
categories:
- Paper Summary
tags:
- Deep Learning
- Regularization
- Normalization
---

# Self-Normalizing Neural Networks
* Paper summary
* Self-Normalizing Neural Networks [Klambauer et al. (2017)](https://arxiv.org/pdf/1706.02515.pdf)
* Günter Klambauer, Thomas Unterthiner, Andreas Mayr and Sepp Hochreiter

## Abstract
* Success of Standard feed-forward neural networks(FNN) is rare
	* FNN cannot exploit many levels of abstract representations
* Self-normalizing neural networks
	* enable high-level abstract representations
  * Scaled exponential linear units (SELUs)
  * Banach fixed-point theorem
		* activations will converge toward zero mean and unit variance
	* vanishing and exploding gradients are **impossible**
	* [Github link](https://github.com/bioinf-jku/SNNs)



## Introduction

* Deep learning has very success
	* CNN: vision and video task
		* self-driving, AlphaGo
		* Kaggle: the "Diabetic Retinopathy" and the "Right Whale" challenge
	* RNN: speech and natural language processing
* Kaggle challenges that are *not related* to vision or sequential tasks 
	* gradient boosting, random forests, SVMs are winning
	* very few cases where FNNs won, which are almost shallow
	* winning using FNN with at most 4 hidden layers
		* HIGGS challenge
		* Merck Molecular Activity challenge
		* Tox21 Data challenge
* Various normalization
	* Batch normalization [Ioffe et al. (2016)](http://proceedings.mlr.press/v37/ioffe15.pdf) [^fn1]
	* Layer normalization [Ba et al. (2016)](https://arxiv.org/pdf/1607.06450.pdf) [^fn2]
	* Weight normalization [Salimans et al. (2016)](https://papers.nips.cc/paper/6114-weight-normalization-a-simple-reparameterization-to-accelerate-training-of-deep-neural-networks.pdf) [^fn3]
* Training with normalization techniques is perturbed by
  * SGD, stochastic regularization (like dropout), the estimation of the normalization parameters
* RNNs, CNNs can stabilize learning via *weight sharing*
* FNNs trained with normalization suffer from these perturbations and have high variance in the training error
  * This high variance hinders learning and slows it down
  * Authors believe this sensitivity to perturbations is the reason that FNNs are less successful than RNNs and CNNs

![Fig1]({{ url }}/assets/2017-06-19/fig1.png){:width="100%"}

<p align="center"> Figure 1. The training error (y-axis) on left: MNIST, right: CIFAR-10. FNN with bn exhibit *high variance* due to perturbations. </p>



## Self-Normalizing Neural Networks (SNNs)

### Normalization and SNNs

![FNN]({{ url }}/assets/2017-06-19/fnn.jpeg){:width="100%"}

* **activation function**: $f$
* **weight matrix**: $\bf{W}$
* **activations in the lower layer**: $\bf{x}$
* **network inputs**: $\bf{z} = \bf{W} \bf{x}$
* **activations in the higher layer**: $\bf{y} = f(\bf{z})$
* activations $\bf{x}, \bf{y}$ and inputs $\bf{z}$ are random variables

#### Assume
* all activations $x_{i}$
  * mean $\mu := \mathbb{E}(x_{i})$ **across samples**
    * $\mathbb{E} := \sum^{N}$: $N$ is a sample size (**my notation**)
  * variance $\nu := \textrm{Var}(x_{i})$
* That means
  * $\mu := \mathbb{E}(x_{1}) = \mathbb{E}(x_{2}) = \cdots = \mathbb{E}(x_{n})$ 
  * $\nu := \textrm{Var}(x_{1}) = \textrm{Var}(x_{2}) = \cdots = \textrm{Var}(x_{n})$
  * $\mathbf{x} = (x_{1}, x_{2}, \cdots, x_{n})$
* single activation $y = f(z), z = \mathbf{w}^{T} \mathbf{x}$
  * mean $\tilde{\mu} := \mathbb{E}(y)$
  * variance $\tilde{\nu} := \textrm{Var}(y)$

#### Define
* $n$ times the mean of the weight vector
  * $\omega := \sum_{i=1}^{n} w_{i}$, for $\mathbf{w} \in \mathbb{R}^{n}$
* $n$ times the second moment of the weight vector
  * $\tau := \sum_{i=1}^{n} w_{i}^{2}$, for $\mathbf{w} \in \mathbb{R}^{n}$

#### mapping $g$
$$ \left( \begin{array}{c} \mu \\ \nu \end{array} \right) 
\mapsto \left( \begin{array}{c} \tilde{\mu} \\ \tilde{\nu} \end{array} \right) \quad : \quad
\left( \begin{array}{c} \tilde{\mu} \\ \tilde{\nu} \end{array} \right) 
= g \left( \begin{array}{c} \mu \\ \nu \end{array} \right) $$

* mapping $g$ keeps $(\mu, \nu)$ and $(\tilde{\mu}, \tilde{\nu})$ close to predefined values, typically $(0, 1)$
  * like most normalization techniques: batch, layer, or weight normalization

#### Notation summary
* relate to activations: $(\mu, \nu, \tilde{\mu}, \tilde{\nu})$
* relate to weight : $(\omega, \tau)$



### Definition 1 (Self-normalizing neural net)
> A neural network is **self-normalizing** if it possesses a mapping
> $g : \Omega \mapsto \Omega$ for each activation $y$ that maps mean and variance from one layer to the next
> and has a *stable* and *attracting* **fixed point** depending on $(\omega, \tau)$ in $\Omega$.
> Furthermore, the mean and the variance remain in the domain $\Omega$, that is $g(\Omega) \subseteq \Omega$,
> where $\Omega = {(\mu, \nu) | \mu \in [\mu_{\textrm{min}}, \mu_{\textrm{max}}], \nu \in [\nu_{\textrm{min}}, \nu_{\textrm{max}}]}$.
> When iteratively applying the mapping $g$, each point within $\Omega$ converges to this fixed point.

* if both their mean and their variance across samples are within predefined intervals
  * then activations are normalized.


### Constructing Self-normalizing Neural Networks
* Tow design choices
  1. the activation function
  2. the initialization of the weight

#### Scaled exponential linear units (SELUs)
$$ \textrm{selu}(x) = \lambda \left\{ \begin{array}{ll}
x & \textrm{if} \ x > 0 \\ 
\alpha e^{x} - \alpha & \textrm{if} \ x \leq 0 \end{array} \right. $$

1. negative and positive values for controlling the mean
2. saturation regions (derivatives approaching zero) to dampen the variance if it is too large in the lower layer
3. a slope larger than one to increase the variance if it is too small in the lower layer
4. a continuous curve. 

#### Weight initialization
* propose $\omega = 0$ and $\tau = 1$ for all units in the higher layer



### Deriving the Mean and Variance Mapping Function $g$

#### Assume
* $x_{i}$: independent from each other but share the same mean $\mu$ and variance $\nu$
  * $\mu := \mathbb{E}(x_{1}) = \mathbb{E}(x_{2}) = \cdots = \mathbb{E}(x_{n})$ 
  * $\nu := \textrm{Var}(x_{1}) = \textrm{Var}(x_{2}) = \cdots = \textrm{Var}(x_{n})$

#### some calculations
* $z = \mathbf{w}^{T} \mathbf{x} = \sum_{i=1}^{n} w_{i} x_{i}$
  * $\mathbb{E}(z) = \mathbb{E}( \sum_{i=1}^{n} w_{i} x_{i} ) = \sum_{i=1}^{n} w_{i} \mathbb{E}(x_{i}) = \mu \omega$
    * independent summation across dimension $(\sum^{n})$ and summation across samples $(\sum^{N})$
  * $\textrm{Var}(z) = \textrm{Var}( \sum_{i=1}^{n} w_{i} x_{i} ) = \nu \tau$
  * used the independence of the $x_{i}$
* Central limit theorem (CLT)
  * input $z$ is a weighted sum of i.i.d. variables $x_{i}$
  * $z$ approaches a normal distribution
  * $z \sim \mathcal{N} (\mu \omega, \sqrt{\nu \tau})$ with density $p_{N}(z; \mu \omega, \sqrt{\nu \tau})$

#### mapping $g$

$$ g : \left( \begin{array}{c} \mu \\ \nu \end{array} \right) 
\mapsto \left( \begin{array}{c} \tilde{\mu} \\ \tilde{\nu} \end{array} \right) : $$

$$ \begin{align}
\tilde{\mu} (\mu, \omega, \nu, \tau) &= \int_{-\infty}^{\infty} \textrm{selu}(z) p_{N}(z; \mu \omega, \sqrt{\nu \tau}) \textrm{d}z \\
\tilde{\nu} (\mu, \omega, \nu, \tau) &= \int_{-\infty}^{\infty} \textrm{selu}(z)^{2} p_{N}(z; \mu \omega, \sqrt{\nu \tau}) \textrm{d}z - (\tilde{\mu})^{2}
\end{align} $$


#### calculation of $g$

##### Remind SELUs
$$ \textrm{selu}(x) = \lambda \left\{ \begin{array}{ll}
x & \textrm{if} \ x > 0 \\ 
\alpha e^{x} - \alpha & \textrm{if} \ x \leq 0 \end{array} \right. $$

##### integration

$$ \tilde{\mu} = \int_{-\infty}^{0} \lambda \alpha (e^{z} - 1) p_{N}(z; \mu \omega, \sqrt{\nu \tau}) \textrm{d}z
  + \int_{0}^{\infty} \lambda z p_{N}(z; \mu \omega, \sqrt{\nu \tau}) \textrm{d}z $$

$$ \tilde{\xi} = \int_{-\infty}^{0} \lambda \alpha (e^{z} - 1)^{2} p_{N}(z; \mu \omega, \sqrt{\nu \tau}) \textrm{d}z
  + \int_{0}^{\infty} \lambda z^{2} p_{N}(z; \mu \omega, \sqrt{\nu \tau}) \textrm{d}z $$

$$ \tilde{\nu} = \tilde{\xi} - \tilde{\mu}^{2} $$


#### analytic form $\mu$ and $\nu$

![Eq1]({{ url }}/assets/2017-06-19/eq1.png){:width="100%"}

##### error function

$$ \begin{align}
\textrm{erf}(x) &= \frac{1}{\sqrt{\pi}} \int_{-x}^{x} e^{-t^{2}} \textrm{d} t \\
 &= \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^{2}} \textrm{d} t 
\end{align} $$

##### complementary error function

$$ \begin{align}
\textrm{erfc}(x) &= 1 - \textrm{erf}(x) \\
 &= \frac{2}{\sqrt{\pi}} \int_{x}^{\infty} e^{-t^{2}} \textrm{d} t 
\end{align} $$



### Stable and Attracting Fixed Point $(0, 1)$ for Normalized Weights

#### Assume
* $\mathbf{w}$ with $\omega = 0$ and $\tau = 1$
* choose a **fixed point** $(\mu, \nu) = (0, 1)$
  * $\mu = \tilde{\mu} = 0$ and $\nu = \tilde{\nu} = 1$

##### Jacobian of $g$

$$ \mathcal{J}(\mu, \nu) = \left( \begin{array}{cc}
  \frac{\partial \tilde{\mu}}{\partial \mu} & \frac{\partial \tilde{\mu}}{\partial \nu} \\
  \frac{\partial \tilde{\nu}}{\partial \mu} & \frac{\partial \tilde{\nu}}{\partial \nu} \\
\end{array} \right) $$

##### useful calculations
* $\mu = \tilde{\mu} = 0$ and $\nu = \tilde{\nu} = 1$
* $\omega = 0$ and $\tau = 1$
* $\textrm{erf}(0) = 0$ and $\textrm{erfc}(0) = 1$
* $\frac{\textrm{d}}{\textrm{d} x} \textrm{erf}(x) = \frac{2}{\sqrt{\pi}} e^{-x^{2}}$
  * $\left. \frac{\textrm{d}}{\textrm{d} x} \textrm{erf}(x) \right\|_{x=0} = \frac{2}{\sqrt{\pi}}$
* $\frac{\textrm{d}}{\textrm{d} x} \textrm{erfc}(x) = \frac{\textrm{d}}{\textrm{d} x} (1 - \textrm{erf}(x))
  =  - \frac{\textrm{d}}{\textrm{d} x} \textrm{erf}(x)$
  * $\left. \frac{\textrm{d}}{\textrm{d} x} \textrm{erfc}(x) \right\|_{x=0} = -\frac{2}{\sqrt{\pi}}$


##### insert $\mu = \tilde{\mu} = 0$,  $\nu = \tilde{\nu} = 1$,  $\omega = 0$ and $\tau = 1$ into Eq. (4) and (5)

$$ 0 = \frac{1}{2} \lambda \left( \alpha e^{1/2} \textrm{erfc}\left(\frac{1}{\sqrt{2}}\right)
      - \alpha + \sqrt{\frac{2}{\pi}} \right) $$

$$ 1 = \frac{1}{2} \lambda^{2} \left( 1 + \alpha^{2} \left( -2 e^{1/2} \textrm{erfc}\left(\frac{1}{\sqrt{2}}\right)
      + e^{2} \textrm{erfc}\left(\frac{2}{\sqrt{2}}\right) + 1 \right) \right) $$

$$ \therefore \alpha = -\sqrt{\frac{2}{\pi}} \left[ e^{1/2} \textrm{erfc} \left(\frac{1}{\sqrt{2}}\right) - 1 \right]^{-1} $$

$$ \therefore \lambda = \sqrt{2} \left[ 1 + \alpha^{2} \left( -2 e^{1/2} \textrm{erfc}\left(\frac{1}{\sqrt{2}}\right)
      + e^{2} \textrm{erfc}\left(\frac{2}{\sqrt{2}}\right) + 1 \right) \right]^{-1/2} $$

##### python code

```python
In [0]: from scipy.special import erfc
In [1]: import math
In [2]: alpha = -math.sqrt(2/math.pi) / (math.exp(0.5) * erfc(1/math.sqrt(2)) - 1)
In [3]: l = math.sqrt(2) / math.sqrt(1 + alpha**2 * (-2 * math.exp(0.5) * erfc(1/math.sqrt(2)) + math.exp(2) * erfc(2/math.sqrt(2)) + 1))
In [4]: alpha
Out[4]: 1.6732632423543778
In [5]: l
Out[5]: 1.0507009873554805
```






##### calculation of $\frac{\partial \tilde{\mu}}{\partial \mu}$

blackboard

##### calculation of $\frac{\partial \tilde{\mu}}{\partial \nu}$

blackboard










## References
[^fn1]: Ioffe, S. and Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings of The 32nd International Conference on Machine Learning, pages 448–456.
[^fn2]: Ba, J. L., Kiros, J. R., and Hinton, G. (2016). Layer normalization. arXiv preprint arXiv:1607.06450.
[^fn3]: Salimans, T. and Kingma, D. P. (2016). Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In Advances in Neural Information Processing Systems, pages 901–909. 




