---
date: '2024-10-22T15:52:04+08:00'
draft: false
tags: ["RL", "Math for ML"]
categories: ["Theories"]
title: "Examples of Estimators"
math: mathjax
---

## Introduction of estimators in ML

Estimations have played an important role in various scenarios in ML algorithms. Variational Auto-Encoder is one good example where one is required to sample a random variable $z \sim \mathcal{N}(0, 1)$ and reparameterize its mean $\mu$ and  deviation $\sigma$ to learn a desired Gaussian distribution $---$ the prior distribution of the latent vector. Other scenarios like Diffusion models and Reinforcement Learning also involve estimation on certain function of a random variable or random variables.  

Another scenario where estimators are especially useful is computing gradients for non-differential functions. As most if not all ML methods incorporate first-order optimizer (SGD, Adam, etc) to learn parameters, computing gradients for functions composing a network is involved as a must.  For those non-differential functions that are either inevitable ($max(\cdot)$,$Topk(\cdot)$) or beneficial since their discretized nature (VQ), estimators mange to approximate the derivatives of these functions. (To certain degree, the parameters in NNs are also random variables since the use the Stochastic Gradient Descent and its variants. Therefore approximating on the gradients of non-differential function on these parameters is also an estimation of random variables. ) 

This post is to exemplify some classic and empirically useful estimators for used in ML.

## Estimates in Random Process

### Monte Carlo Method

As what the ***the law of large numbers*** states, the average of independent samples of  a random variable converges to the expectation of which variable as the number of samples approaches infinity. In the realm of ML, Monte Carlo method is an very common practice, almost as a basis. For example, the very IID (independently identical distribution) prerequisite justify the generalization from train set to test set. In fact the dataset-side of the Scaling Law depicts the numerical relationship between generalization and dataset size. 

 ![overfitting](../images/scaling law(overfitting).jpg)

<center style="color:#C0C0C0;text-decoration:underline">Figure 1. The extent of overfitting depicted by the Scaling Law (https://arxiv.org/pdf/2001.08361) </center>

If, Training on train set and generalizing to test set can be considered as a ***one-shot*** Monte Carlo, sampling *episodes* in Reinforcement Learning is a ***continuous*** version of Monte Carlo. To explain a little bit, Reinforcement Learning is about interacting with environments where you collect data as input (state) and collect data as label (reward) from what you output (action) to the given environment. **Markov Chain Monte Carlo** (MCMC) is then introduced to justify how we sample in RL. What 





## Gradient Estimator

### Single Random Variable

Suppose we have $\mathbf{x}$ as a random variable with unknown distribution. We parameterize a function $f(\mathbf{x},\theta)$ with serve as goal to be optimized towards what we desired. Naturally, the scaler loss we optimize is the expectation of $f$, which is denoted as $\mathbb{E}\left[f(\mathbf{x})\right]$. This involves the calculation of $\frac{\partial}{\partial \theta}\mathbb{E}\left[f(\mathbf{x})\right]$. 

Depending on what we know (or set) for $\mathbf{x}$, different estimators can be used for this calculation. 

- The parameters is to depict the distribution, i.e. $\mathbf{x} \sim p(\cdot;\theta)$. We can use $\color{red}{\text{score function}}$ (SF) estimator: 

  $$\begin{equation} \begin{aligned}\frac{\partial}{\partial \theta} \mathbb{E}\left[f(\mathbf{x})\right] = \mathbb{E}\left[f(\mathbf{x})\frac{\partial}{\partial \theta} \log p(\mathbf{x};\theta)\right] \end{aligned}\end{equation}$$

  This is derived as follows:

  $$\begin{equation} \begin{aligned}  \frac{\partial}{\partial \theta}\mathbb{E}\left[f(\mathbf{x})\right] &= \frac{\partial}{\partial \theta}\int \text{d} \textbf{x} \  p(\textbf{x};\theta)f(\textbf{x}) \newline &= \int \text{d} \textbf{x} \ \frac{\partial}{\partial \theta} p(\textbf{x};\theta)f(\textbf{x}) \newline &= \int \text{d}\textbf{x} \  p(\textbf{x};\theta)\frac{\partial}{\partial \theta}\log p(\textbf{x};\theta)f(\textbf{x}) \newline &= \mathbb{E} \left[f(\textbf{x}) \frac{\partial}{\partial \theta} \log p(\textbf{x};\theta)\right] \end{aligned} \end{equation} $$

  The above equation holds (when SF estimator is valid) if and only if $p(\mathbf{x};\theta)$ is a continuous function of $\theta$ (not necessarily a continuous function of $\mathbf{x}$). 

- In some other cases, $\mathbf{x}$ may be a deterministic function of $\theta$ and another random variable $\mathbf{z}$, i.e. $\mathbf{x} = \mathbf{x}(\mathbf{z};\theta)$. We can then compute $\color{red}{\text{pathwise derivative}}$ (PD) estimator:

  $$\begin{equation}\begin{aligned} \frac{\partial}{\partial \theta} \mathbb{E} \left[f(\mathbf{x}(\mathbf{z};\theta))\right] = \mathbb{E}\left[\frac{\partial}{\partial \theta}f(\mathbf{x}(\mathbf{z;\theta}))\right]\end{aligned}\end{equation}$$ 

  This equation is valid (when we can swap the derivative and the expectation) if and only if $f(\mathbf{x}(\mathbf{z};\theta))$ is a continuous function of $\theta$ for all $\mathbf{z}$. 

- When $\theta$ appear both in the distribution $p(\cdot; \theta)$ and function $\textbf{x}(\textbf{z};\theta)$, we have estimator of two terms:

  $$\begin{equation}\begin{aligned}\frac{\partial}{\partial\theta} \mathbb{E}_{\textbf{z}\sim p(\cdot;\theta)}\left[f(\textbf{x}(\textbf{z};\theta))\right] = \mathbb{E} \left[\frac{\partial}{\partial\theta}f(\textbf{x}(\textbf{z};\theta)) + f(\textbf{x}(\textbf{z};\theta))\left(\frac{\partial}{\partial \theta} \log p(\textbf{z};\theta)\right)\right]\end{aligned}\end{equation}$$

  This formula is derived as follows:

  $$\begin{equation}\begin{aligned}\frac{\partial}{\partial \theta}\mathbb{E}\left[f(\textbf{x}(\textbf{z};\theta))\right] &= \frac{\partial}{\partial \theta}\int \text{d} \textbf{z} \  p(\textbf{z};\theta)f(\textbf{x}(\textbf{z};\theta)) \newline &= \int \text{d}\textbf{z} \  \left(\frac{\partial}{\partial \theta} p(\textbf{z};\theta)\right)f(\textbf{x}(\textbf{z};\theta)) + \left(\frac{\partial}{\partial \theta}f(\textbf{x}(\textbf{z};\theta))\right)p(\textbf{z};\theta) \newline &= \color{blue}{\int \text{d}\textbf{z} \  p(\textbf{z};\theta)\left(\frac{\partial}{\partial \theta}\log p(\textbf{z};\theta)\right)f(\textbf{x}(\textbf{z};\theta))} + \color{red}{\int \text{d} \textbf{z} \ p(\textbf{z};\theta)\frac{\partial}{\partial \theta}f(\textbf{x};\theta)} \newline &=\color{red}{\mathbb{E}\left[\frac{\partial}{\partial\theta} f(\textbf{x}(\textbf{z};\theta))\right]} + \color{blue}{\mathbb{E}\left[\left(\frac{\partial}{\partial \theta}\log p(\textbf{x};\theta)\right)f(\textbf{x}(\textbf{z};\theta))\right]}\end{aligned}\end{equation}  $$

  
