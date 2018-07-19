# Self-Normalizing Networks
Implementations based on "Self-normalizing networks"(SNNs) as suggested by Günter Klambauer, Thomas Unterthiner, Andreas Mayr - Purpose: Learning

### Objective : Understanding the core concept of Self-Normalizing NNs, their composition and detailed study of the research paper

Shortcomings of current deep learning architectures:
1.	FNNs i.e the feed-forward neural networks that perform well are typically shallow and, therefore cannot exploit many levels of abstract representations.
2.	Success stories of Deep Learning with standard feed-forward neural networks (FNNs) are rare.

## Solution proposed:
Introduction of self-normalising neural networks (SNNs) to enable high-level abstract representations. Neuron activations of SNNs automatically converge towards zero mean and unit variance. The activation function of SNNs are "scaled exponential linear units" (SELUs), which induce self-normalising properties. As proved in the appendix attached in the paper in disussion, activations close to zero mean and unit variance that are propagated through many network layers will converge towards zero mean and unit variance even if noise is present, providing the following advantages due to convergence property of SNNs:
•	Train deep networks with many layers
•	Employ strong regularization
•	Furthermore, for activations not close to unit variance, an upper and lower bound on the variance has been proved, thus, vanishing and exploding gradients are impossible.

## Implementation/ Analysis Notes:
1.	Analysed the implementations in the official repository of the paper in discussion and recognised differences by making changes in activation functions and combinations of fully connected and pooling layers.
2.	 Difference in various normalization techniques:
	i.  Batch normalization - to normalize neuron activations to zero mean and unit variance
	ii. Layer normalization - also ensures zero mean and unit variance
	iii. Weight normalization - ensures zero mean and unit variance if in the previous layer the activations have zero mean and unit variance
3.	 Training with normalization techniques is perturbed by stochastic gradient descent (SGD), stochastic regularization (like dropout), and the estimation of the normalization parameters. Both RNNs and CNNs can stabilize learning via weight sharing, therefore they are less prone to these perturbations. In contrast, FNNs trained with normalization techniques suffer from these perturbations and have high variance in the training error. Furthermore, strong regularization, such as dropout, is not possible as it would further increase the variance which in turn would lead to divergence of the learning process, thus leading to FNNs less success rate.
4.	 Normalization techniques like batch, layer, or weight normalization ensure a mapping g that keeps (µ, ν) and (˜µ, ν˜) close to predefined values, typically (0, 1).
5.	 A single activation y = f(z) has net input z = wT x. For n units with activation xi , 1 <= i<= n in the lower layer, we define n times the mean of the weight vector w ∈R^n as ω := sigma(wi) =1 wi and n times the second moment as τ := sigma(w^2) .Definition of Self-Normalizing Neural Net: A neural network is self-normalizing if it possesses a mapping g : Ω→Ω for each activation y that maps mean and variance from one layer to the next and has a stable and attracting fixed point depending on (ω, τ ) in Ω. Furthermore, the mean and the variance remain in the domain Ω, that is g(Ω) ⊆Ω, where Ω = {(µ, ν) | µ ∈ [µmin, µmax], ν ∈ [νmin, νmax]}. When iteratively applying the mapping g, each point within Ω converges to this fixed point.
6.	 Activations of a neural network to be normalized, if both their mean and their variance across samples are within predefined intervals. If mean and variance of x are already within these intervals, then also mean and variance of y remain in these intervals, i.e., the normalization is transitive across layers. Within these intervals, the mean and variance both converge to a fixed point if the mapping g is applied iteratively.
7.	 SNNs keep normalization of activations when propagating them through layers of the network. The normalization effect is observed across layers of a network: in each layer the activations are getting closer to the fixed point. The normalization effect can also observed be for two fixed layers across learning steps: perturbations of lower layer activations or weights are damped in the higher layer by drawing the activations towards the fixed point. If for all y in the higher layer, ω and τ of the corresponding weight vector are the same, then the fixed points are also the same. In this case we have a unique fixed point for all activations y. Otherwise, in the more general case, ω and τ differ for different y but the mean activations are drawn into [µmin, µmax] and the variances are drawn into [νmin, νmax].
8.	 The SELU activation function is given by selu(x) = λ _{ x : if x > 0, αe^x − α : if x <= 0
9.	Properties of SELU:
	 (1) negative and positive values for controlling the mean
         (2) saturation regions (derivatives approaching zero) to dampen the variance if it is too large in the lower layer
         (3) a slope larger than one to increase the variance if it is too small in the lower layer
         (4) a continuous curve
10.	 Activation function is made by multiplying the exponential linear unit (ELU) with λ > 1 to ensure a slope larger than one for positive net inputs.
11.	The net input z is a weighted sum of independent, but not necessarily identically distributed variables xi , for which the central limit theorem (CLT) states that z approaches a normal distribution: z ∼ N (µω, √ ντ ) with density pN(z; µω, √ ντ ). The function g maps the mean and variance of activations in the lower layer to the mean µ˜ = E(y) and variance ν˜ = Var(y) of the activations y in the next layer:
![image](https://user-images.githubusercontent.com/16400217/42956550-de083918-8b9d-11e8-9de7-c6aa92475fcf.png)

12.	 Given a set y=f(x) of n equations in n variables x1,….xn, written explicitly as
 ![image](https://user-images.githubusercontent.com/16400217/42956620-172910e6-8b9e-11e8-94a4-c62a6537e953.png)
or more explicitly as
![image](https://user-images.githubusercontent.com/16400217/42956659-31fdf0da-8b9e-11e8-9e39-b92fb1ab8078.png)
the Jacobian matrix, sometimes simply called "the Jacobian" (Simon and Blume 1994) is defined by
 ![image](https://user-images.githubusercontent.com/16400217/42956701-4ed9faaa-8b9e-11e8-8853-6c653a5ce231.png)
 
13.	 Stable and Attracting Fixed Point (0, 1) for Normalized Weights: µ˜ = µ = 0 and ν˜ = ν = 1: The analytical expressions for α and λ are generated from as per integration in pt. 11. The point of interest is whether the fixed point (µ, ν) = (0, 1) is stable and attracting. If the Jacobian of g has a norm smaller than 1 at the fixed point, then g is a contraction mapping and the fixed point is stable.  This calculation as shown in the paper proves it to be stable.
14.	 Stable and Attracting Fixed Points for Unnormalized Weights -  (Task to do- Not picked up yet)
15.	Central Limit Theorem: When independent random variables are added, their properly normalised sum tends toward a normal distribution (informally a "bell curve") even if the original variables themselves are not normally distributed. 
16.	Lyapunov CLT: In this variant of the central limit theorem the random variables Xi have to be independent, but not necessarily identically distributed. The theorem also requires that random variables.

Deduction verification via implementation:
Note: The reproduction of Figure 1 was done using help of some existing github repositories. The analysis of the code and its change when parameters change has been the main objective here.

Above mentioned drawbacks are curbed via Self-normalizing neural networks (SNNs) as they are robust to perturbations and do not have high variance in their training errors (Figure 1). SNNs push neuron activations to zero mean and unit variance thereby leading to the same effect as batch normalization, which enables to robustly learn many layers. SNNs are based on scaled exponential linear units “SELUs” which induce self-normalizing properties like variance stabilization which in turn avoids exploding and vanishing gradients.





## CONTENTS:
### KERAS CNN scripts:
- KERAS: Convolutional Neural Network on MNIST
- KERAS: Convolutional Neural Network on CIFAR10

### Basic python functions to implement SNNs
are provided here: selu.py

### In order to reproduce Figure1 in the paper
Code snippets are provided here: Figure1

### Basic Implementation
Referred various sources and tutorials of pytorch to manipulate and implement functions
