---
layout: post
title: Basic formula in information theory
categories:
- Machine Learning
tags:
- Machine Learning
- Information Theory
- Entropy
- KL Divergence
---

## Self-information

$$ I(x) = - \log(P(x)) = \log \left( \frac{1}{P(x)} \right) $$


## Entropy (Shannon Entropy)
Expectation of self-information

$$ H(X) = \mathbb{E}_{X} [I(x)] $$

$$ H(X) = - \sum_{x} p(x) \log p(x) $$

$$ H(X) = - \int_{X} p(x) \log p(x) dx $$


## Joint entropy

$$ H(X, Y) = \mathbb{E}_{X, Y} [-\log p(x, y)] $$

$$ H(X, Y) = - \sum_{x,y} p(x, y) \log p(x, y) $$

$$ H(X, Y) = - \int_{X, Y} p(x, y) \log p(x, y) dx dy $$


## Cross entropy

$$ H(P, Q) = \mathbb{E}_{P} [-\log Q] = H(P) + D_{KL}(P || Q) $$

$$ H(P, Q) = - \sum_{x} p(x) \log q(x) $$

$$ H(P, Q) = - \int_{X} p(x) \log q(x) dx $$


## Mutual information

$$ I(X; Y) = \mathbb{E}_{X,Y} [SI(x,y)] $$

$$ I(X; Y) = \sum_{x,y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)} $$

$$ I(X; Y) = \int_{X,Y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)} dx dy $$

### Basic property of mutual information

$$ I(X; Y) = H(X) - H(X|Y) $$

$$ I(X; Y) = I(Y; X) = H(X) + H(Y) - H(X, Y) $$


## Kullback-Leibler divergence (information gain)

$$ D_{KL} (P || Q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)} $$

$$ D_{KL} (P || Q) = \int_{X} p(x) \log \frac{p(x)}{q(x)} dx $$


### Basic property of KL divergence
* The KL divergence is always non-negative

$$ D_{KL} (P || Q) \geq 0 $$

* The KL divergence is not symmetric

$$ D_{KL} (P || Q) \neq D_{KL} (Q || P) $$

* The relation between KL divergence and cross entropy

$$
\begin{eqnarray}
D_{KL} (P || Q) &=& - \sum_{x} p(x) \log q(x) &+& \sum_{x} p(x) \log p(x) \\
&=& H(P, Q) &-& H(P)
\end{eqnarray}
$$


## References
* Wikipedia


