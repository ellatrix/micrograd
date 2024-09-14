---
layout: default
title: '3.2. makemore: Initialisation'
permalink: '/makemore-initialisation'
---

We want logits to be uniformly 0. Weights should be sampled from N(0, 1/sqrt(n)). Need entropy for symmetry breaking.

### Tanh too saturated

Lot's of -1 and 1 preactivations. tanh backward is 1 - tanh^2 so it stops the backpropagation.
Vanishing gradients.

Kaiming He, 2020.

We need a slight gain because the tanh is a squashing function. 5/3.
