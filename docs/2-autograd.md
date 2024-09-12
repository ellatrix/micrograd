---
layout: default
title: '2. Autograd'
permalink: 'autograd'
---

Manually figuring out gradients can be tedious. It's relatively easy for a
single layer neural network, but gets more complex as we add layers. Machine
Learning libraries all have an autograd engine: you can build out a mathematical
tree and it will automatically be able to figure out the gradients with respect
to te variables. We'll now build something similar, although just for the
operations we need. In the previous notebook, we just needed gradients for
`matMul` and `softmaxCrossEntropy`, so let's start with these operations.

<script>
