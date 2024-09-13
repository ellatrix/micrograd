---
layout: default
title: '2. Autograd'
permalink: 'autograd'
---

Manually figuring out gradients can be tedious. It's relatively easy for a
single layer neural network, but gets more complex as we add layers. Machine
Learning libraries all have an autograd engine: you can build out a
<em>mathematical expression</em> and it will automatically be able to figure out
the gradients with respect to the variables. We'll now build something similar,
although just for the operations we need. In the previous notebook, we just
needed gradients for `matMul` and `softmaxCrossEntropy`, so let's start with
these operations.

But first, let's build a simpler autograd engine with just two operations:
addition and multiplication of scalar numbers.

> [Auto]grad is everything you need to build neural networks, everything else is
just for efficiency. - Andrej Karpathy

## Building `Value`

To build the tree, we need a wrapper around values. Then we'll need the
gradients with respect to the variables so we know how much each variable
affects the result.

<script>
class Value {
    constructor(data) {
        this.data = data;
    }
}
export const a = new Value(2);
</script>

Nice! Now we can do some operations on these values and an example expression.

<script>
export class Value {
    constructor(data, _children = []) {
        this.data = data;
        this._prev = new Set( _children );
    }
    add(other) {
        const result = new Value(this.data + other.data, [ this, other ]);
        result._op = '+';
        return result;
    }
    mul(other) {
        const result = new Value(this.data * other.data, [ this, other ]);
        result._op = '*';
        return result;
    }
}
export const a = new Value(2);
export const b = new Value(-3);
export const c = new Value(10);
export const d = a.mul(b);
export const e = d.add(c);
export const f = new Value(-2);
export const L = e.mul(f);
</script>

Let's visualize the graph to better understand it. Ignore the graph code, it's
not so important, but all here for transparency.

<script>
export { instance } from 'https://esm.sh/@viz-js/viz';
export { Graph } from 'https://esm.sh/@dagrejs/graphlib';
export { default as graphlibDot } from 'https://esm.sh/graphlib-dot';
</script>

<script>
const ids = new WeakMap();

function createId(node) {
    let id = ids.get(node);
    if (!id) {
        id = createId.counter++;
        ids.set(node, id);
    }
    return id;
}

createId.counter = 0;

function trace( root ) {
	const nodes = new Set();
	const edges = new Set();

	function build( node ) {
		if ( ! nodes.has( node ) ) {
			nodes.add( node );

			for ( const child of node._prev ) {
				edges.add( [ child, node ] );
				build( child );
			}
		}
	}

	build( root );

	return { nodes, edges };
}

export async function drawDot(root) {
	const { nodes, edges } = trace( root );
	const graph = new Graph( { compound: true } );
    graph.setGraph({ rankdir: "LR" });

	for ( const node of nodes ) {
		node._id = createId(node);
		graph.setNode( node._id, {
            shape: 'record',
            // `grad` will be important later.
            label: [
                node.label,
                [ 'data', node.data ].join(': '),
                node.grad !== undefined ? [ 'grad', node.grad ].join(': ') : null
            ].filter( Boolean ).join(' | ')
        } );

		if ( node._op ) {
			graph.setNode( node._id + node._op, { label: node._op } );
			graph.setEdge( node._id + node._op, node._id );
		}
	}

	for ( const [ node, child ] of edges ) {
		graph.setEdge( node._id, child._id + child._op );
	}

	const viz = await instance();
    const dotString = graphlibDot.write(graph);
    return viz.renderSVGElement(dotString);
}

a.label = 'a';
b.label = 'b';
c.label = 'c';
d.label = 'd';
e.label = 'e';
f.label = 'f';
L.label = 'L';

export default await drawDot(L);
</script>

Great, we have now implemented the forward pass for multiplication and addition.

## Manual gradient calculation

derivative = sensitivity.

Let's add a "backward pass": calculating the gradients for each node from the
perspective of the output `L`. The gradient of L with respect to L is 1: if you
change L by a tiny amount, well, then L changes by that tiny amount.

So we should always start a backward pass setting the output's gradient to 1.

<script>
L.grad = 1;
export default await drawDot(L);
</script>

For multiplication `a*b=output`, the gradient of an input `a` with respect to
the `output` is the other input, `b`. `d(output)/da = b` and `d(output)/db = a`.
In other words, if you change `a` by a tiny amount, the result changes by `b`
times a tiny amount. If you change `b` by a tiny amount, the result changes by
`a` times a tiny amount.

If you remember your calculus, you might remember that the definition of the
derivative is the limit of `(f(x+h)-f(x))/h` as `h` goes to 0. So if we fill this in for multiplication, we get:

```
((a+h)*b - a*b)/h =
(ab + bh - ab)/h =
bh/h =
b
```

<script>
e.grad = f.data;
f.grad = e.data;
export default await drawDot(L);
</script>

Now we get the most important part. We need the gradient of `d` and `c` with
respect to `L`. We actually already have some information: the derivative of
`e` with respect to `L`. If we know the gradient of `d` with respect to `e`,
we can use the chain rule:

```
d(L)/d(e) * d(e)/d(c) = d(L)/d(c)
```

So what is the gradient of `d` with respect to `e`?

For addition, the gradient of an input `a` with respect to the `output` is 1.
`d(output)/da = 1` and `d(output)/db = 1`. In other words, if you change `a` by
a tiny amount, the result changes by 1 times a tiny amount. If you change `b` by
a tiny amount, the result changes by 1 times a tiny amount.

Again, following the definition of the derivative, we get:

```
((a+h) + b - (a + b))/h =
(a + h + b - a - b)/h =
h/h =
1
```

So `d(e)/d(c)` is 1.

And using the chain rule, we get now know that `d(L)/d(c)` is `d(L)/d(e) *
d(e)/d(c) = d(L)/d(e) * 1`. Plus nodes just pass the gradient along.

This is the core of backpropagation: as we propagate through the nodes, we can
simply multiply the gradients together to get the gradient of the input with
respect to the output.

<script>
c.grad = e.grad * 1;
d.grad = e.grad * 1;
export default await drawDot(L);
</script>

And finally, we can do the same thing for `a` and `b`. The gradient of `a` with
respect to `L` is `d(L)/d(a) = d(L)/d(d) * d(d)/d(a) = d(d) * b`. And the gradient
of `b` with respect to `L` is `d(L)/d(b) = d(L)/d(d) * d(d)/d(b) = d(d) * a`.

<script>
b.grad = d.grad * a.data;
a.grad = d.grad * b.data;
export default await drawDot(L);
</script>

We can also numerically check our gradients. If we nudge `a` by a small amount, let's say `0.001`, we should see the output change by the gradient times `0.001`.

<script>
function f(h) {
    const a = new Value(2);
    a.data += h;
    const b = new Value(-3);
    const c = new Value(10);
    const d = a.mul(b);
    const e = d.add(c);
    const f = new Value(-2);
    return e.mul(f);
}

const h = 0.001;
const L1 = f(0);
const L2 = f(h);
export const numericalGradient = (L2.data - L1.data) / h;
export const analyticalGradient = a.grad;
</script>

## Let's build a neuron

These gradients are important for training neural networks, because we will
define a loss function and then calculate the gradient of the loss with respect
to each parameter in the network, which we can then use to update the parameters
to improve the accuracy.

Now, we don't need to implement every single atomic operation. As long as we
know how to calculate the local derivative for a function, we can cluster it
together. `tanh(x)` is important for neural networks, because it squashes the
output of a neuron to a range between -1 and 1.

<script>
Value.prototype.tanh = function() {
    const x = this.data;
    const t = (Math.exp(2*x)-1)/(Math.exp(2*x)+1);
    const result = new Value(t, [this]);
    result._op = 'tanh';
    return result;
}

// Inputs x1, x2.
const x1 = new Value(2);
const x2 = new Value(0);
// Weights w1, w2.
const w1 = new Value(-3);
const w2 = new Value(1);
// Bias b.
const b = new Value(6.8813735870195432);
// Output.
const x1w1 = x1.mul(w1);
const x2w2 = x2.mul(w2);
const n = x1w1.add(x2w2).add(b);
const o = n.tanh();

x1.label = 'x1';
x2.label = 'x2';
w1.label = 'w1';
w2.label = 'w2';
b.label = 'bias';

// We should always set the output's gradient to 1.
o.grad = 1;
// If we look up the derivative of `tanh(x)`, we know it's `1 - tanh(x)^2`.
n.grad = o.grad * (1 - o.data**2);
// Remember addition just passes the gradient along (the derivative of addition is 1).
x1w1.grad = n.grad * 1;
x2w2.grad = n.grad * 1;
b.grad = n.grad * 1;
// Remember that the derivative of `mul` is the other input.
x1.grad = x1w1.grad * w1.data;
x2.grad = x2w2.grad * w2.data;
w1.grad = x1w1.grad * x1.data;
w2.grad = x2w2.grad * x2.data;

export default await drawDot(o);
</script>

Intuitively, it makes sense that the gradient of w2 is 0. If we wiggle w2, the
output won't change because x2 is 0.

## Automation

Let's automate this. For each operation, we'll need to define the gradient with
respect to its inputs and multiply it by the output gradient.

We never want to call _backward for any node before we've done all its
dependencies, because we'll need the result of the gradient for the deeper
nodes.

<script>
export function getTopologicalOrder( node ) {
    const result = [];
    const visited = new Set();

    function visit( node ) {
        if ( visited.has( node ) ) return;
        visited.add( node );
        for ( const child of node._prev ) visit( child );
        result.push( node );
    }

    visit( node );

    return result;
}
</script>

<script>
export class Value {
    static operations = new Map();
    constructor(data, _children = []) {
        this.data = data;
        this._prev = _children;
    }
    static addOperation(name, forward, backward) {
        this.operations.set(name, { forward, backward });
        this.prototype[name] = function(...args) {
            args = [ this, ...args ];
            const backwards = backward(...args);
            const result = new Value( forward(...args), args );
            result._op = name;
            return result;
        }
    }
    backward() {
        const reversed = getTopologicalOrder(this).reverse();

        for (const node of reversed) {
            node.grad = 0;
        }

        this.grad = 1;

        for (const node of reversed) {
            if (node._op) {
                const { backward } = Value.operations.get(node._op);
                const args = node._prev;
                const backwards = backward(...args);
                for (let i = 0; i < args.length; i++) {
                    args[i].grad = backwards[i](node);
                }
            }
        }
    }
}

Value.addOperation('add', (a, b) => a.data + b.data, (a, b) => [
    (out) => out.grad,
    (out) => out.grad
]);
Value.addOperation('mul', (a, b) => a.data * b.data, (a, b) => [
    (out) => b.data * out.grad,
    (out) => a.data * out.grad
]);
Value.addOperation('tanh', (a) => Math.tanh(a.data), (a) => [
    (out) => (1 - out.data**2) * out.grad
]);

// Inputs x1, x2.
const x1 = new Value(2);
const x2 = new Value(0);
// Weights w1, w2.
const w1 = new Value(-3);
const w2 = new Value(1);
// Bias b.
const b = new Value(6.8813735870195432);
// Output.
const x1w1 = x1.mul(w1);
const x2w2 = x2.mul(w2);
const n = x1w1.add(x2w2).add(b);
const o = n.tanh();

x1.label = 'x1';
x2.label = 'x2';
w1.label = 'w1';
w2.label = 'w2';
b.label = 'bias';

o.backward();

export default await drawDot(o);
</script>

## Fixing the accumulation bug

Let's try adding the same value together.

<script>
const a = new Value(3);
const b = a.add(a);
b.backward();
export default await drawDot(b);
</script>

Note that the graph for `a` overlaps. But the gradient is 1, while it should be 2.

This is because we're not accumulating the gradients. Instead, the backward pass
is overwritting the last calculated gradient. We need to fix the `backward` function to
accumulate the gradients.

<script>
Value.prototype.backward = function() {
    const reversed = getTopologicalOrder(this).reverse();

    for (const node of reversed) {
        node.grad = 0;
    }

    this.grad = 1;

    for (const node of reversed) {
        if (node._op) {
            const { backward } = Value.operations.get(node._op);
            const args = node._prev;
            const backwards = backward(...args);
            for (let i = 0; i < args.length; i++) {
                args[i].grad += backwards[i](node);
            }
        }
    }
}
const a = new Value(3);
const b = a.add(a);
b.backward();
export default await drawDot(b);
</script>

## Requires grad

We can add a `requiresGrad` flag to keep track of whether a value needs to
calculate a gradient or not.

## Composite tensor operations

As previously explained, we can cluster operations together to form a composite
operation. We can also operate on matrices (or generally tensors) instead
of scaler values. We will take advantage of this because it is faster matrix
multiplication in one batch than to make all the atomic operations separately.

In the last chapter, we only needed `matMul` and `softmaxCrossEntropy`. So let's
implement those.

### Matrix multiplication



### Softmax cross-entropy
