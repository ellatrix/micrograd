---
layout: default
title: '1. Building a neural network from scratch in JavaScript'
permalink: '/'
---

<aside>
    This is an interactive notebook. You can edit the snippets and use
    <code>print( var, label )</code>.
</aside>

I really enjoyed Andrej Karpathy’s “Zero to Hero” series because in it, he
builds everything from scratch. Or almost at least. This post follows his second
video lecture, but in JS instead. While he uses PyTorch’s low level APIs, we’ll
be creating absolutely everything from scratch, without any fancy machine
learning frameworks!

Let’s begin with a simple neural network that learns which character is most
likely to follow given a character. A simple character-level bigram model. We
will train on names, and when done we can generate more (“makemore”) name-like
sequences.

First, we need to fetch a list of names and process those names into a useable
dataset.

<script>
const response = await fetch('https://raw.githubusercontent.com/karpathy/makemore/master/names.txt');
export const text = await response.text();
export const names = text.split('\n');
</script>

Now that we have an array of names, let’s create index-to-character and
character-to-index mappings. Neural networks cannot operate on characters,
we’ll need to map the characters to numbers. Additionally we’ll want to
include a character that can be used to signify the start or end of a name.
Let’s use a newline, set at index 0.

<script>
export const indexToCharMap = [ '.', ...new Set( names.join('') ) ].sort();
export const stringToCharMap = {};

for ( let i = indexToCharMap.length; i--; ) {
    stringToCharMap[ indexToCharMap[ i ] ] = i;
}
</script>

Next, let’s create the training dataset. We need pairs (bigrams) of a character
x preceding a character y. My name `ella` has 5 examples: `.e`, `el`, `ll`,
`la`, and `a.`! To do this we prepend and append the names with `.`

<script>
export const xs = []; // Inputs.
export const ys = []; // Targets, or labels.

for ( const name of names ) {
    const exploded = '.' + name + '.';
    let i = 1;
    while ( exploded[ i ] ) {
        xs.push( stringToCharMap[ exploded[ i - 1 ] ] );
        ys.push( stringToCharMap[ exploded[ i ] ] );
        i++;
    }
}
</script>

That’s it! Now we have our training data. For example for `ella`, `x[0]` gives
us `0` (starting character) and `y[0]` gives us `5` (E). `x[1]` gives us `5` (E)
and `y[1]` gives us `12` (L) and so on.

We now want to build a neural network with a single layer, in which the weights
are a 27×27 matrix. You can see it as a table containing the probabilities of a
character in each column being followed by the character in the row. We could
enlarge the number of columns, but this is easier to visualise later.

The weights are trainable and should be initialised randomly to break symmetry,
otherwise all gradients will be the same. JavaScript does not have a native
matrix data type, so let’s use flat arrays with a `shape` property. For example
a `shape` of `[ 20, 27 ]` is a matrix with 20 rows and 27 columns. Let’s create
a utility function to create a matrix.

<script>
export class FloatMatrix extends Float32Array {
    constructor( data, shape = data?.shape || [] ) {
        const length = shape.reduce( ( a, b ) => a * b, 1 );

        super( data || length );

        if ( this.length !== length ) {
            throw new Error( 'Shape does not match data length.' );
        }

        this.shape = shape;
    }
}

print( new FloatMatrix( null, [ 2, 3 ] ) );
</script>

With `FloatMatrix`, we can now initialise our weights matrix more easily. Let’s
add random values between 1 and -1.

<script>
const totalChars = indexToCharMap.length;
export const W = new FloatMatrix( null, [ totalChars, totalChars ] );
for ( let i = W.length; i--; ) W[ i ] = Math.random() * 2 - 1;
</script>

Given these weights, we now want to calculate a probability distribution for
each example in our training set. We need a way to pluck out the weights row for
each input (x) in the dataset. For example `x[0]` is `0` and needs to pluck out
the 0th row of our weights, which can then give us the probability distribution
for the next character. Our neural net will later on change the weights so the
probability for `y[0]` (E) to be a starting character gets higher.

An easy way to do this is to convert `xs` to a one-hot matrix (`xs.length` x
`totalChars`) and then matrix multiply that with the weights (`totalChars` x
`totalChars`) resulting in the desired `xs.length` x `totalChars` matrix where
each row is the weights for the character in `xs`. Since the one-hot matrix is
all zeros except for one column, it effectively plucks out the row with that
column index from the weights matrix.

First, let’s one-hot encode `xs`.

<script>
export function oneHot( a, length ) {
    const B = new FloatMatrix( null, [ a.length, length ] );
    for ( let i = a.length; i--; ) B[ i * length + a[ i ] ] = 1;
    return B;
}

export const XOneHot = oneHot( xs, indexToCharMap.length );
</script>

Now let’s multiply it by the weights. For that we need to implement matrix
multiplication.

<script>
export function matMul(A, B) {
    const [ m, n ] = A.shape;
    const [ p, q ] = B.shape;
    const C = new FloatMatrix( null, [ m, q ] );

    if ( n !== p ) {
        throw new Error( 'Matrix dimensions do not match.' );
    }

    for ( let m_ = m; m_--; ) {
        for ( let q_ = q; q_--; ) {
            let sum = 0;
            for ( let n_ = n; n_--; ) {
                sum += A[m_ * n + n_] * B[n_ * q + q_];
            }
            C[m_ * q + q_] = sum;
        }
    }

    return C;
}

export const Wx = matMul( XOneHot, W );
</script>

This is the only layer we will have in our neural network for now. Just a
linear layer without a bias, a really simple and dumb neural network.

Now we have the weights for each input, but we somehow want these numbers to
represent the probabilities for the next character. For this we can use the
softmax, which scales numbers into probabilities. So for each row of `Wx` we
want to call the softmax function so that each row is a probability distribution
summing to 1. You can see `Wx` a bit as “log-counts” (also called “logits”) that
can be exponentiated to get fake “counts” (the occurrences of a character for a
previous character). To get probabilities, they can be normalised by dividing
over the sum. This is basically what softmax does.

First we need to create a softmax function. We want to calculate the softmax
for every row. We’re subtracting the maximum before exponentiating for
numerical stability.

<script>
export function softmaxByRow( A ) {
    const [m, n] = A.shape;
    const B = new FloatMatrix( null, A.shape );
    for ( let m_ = m; m_--; ) {
        let max = -Infinity;
        for ( let n_ = n; n_--; ) {
            const value = A[m_ * n + n_];
            if (value > max) max = value;
        }
        let sum = 0;
        for ( let n_ = n; n_--; ) {
            const i = m_ * n + n_;
            // Subtract the max to avoid overflow
            sum += B[i] = Math.exp(A[i] - max);
        }
        for ( let n_ = n; n_--; ) {
            B[m_ * n + n_] /= sum;
        }
    }
    return B;
}

export const probs = softmaxByRow( Wx );
</script>

Now that we have probabilities for each row (summing to 1), we can use this
later to sample from the model.

## Evaluating the model

We now need a way evaluate the model mathematically. We can do this by creating
a function to calculate a loss. As long as it’s all differentiable, we can then
calculate the gradients with respect to the weights matrix and tune it, giving
us a better loss and thus better probabilities. The idea is that we can keep
doing this and iteratively tune the weights matrix to give one with as low a
loss as possible.

The loss function we will use here is called the negative log likelihood, which
measures how well the data fits our model.

First, we need to pluck out the probabilities given the characters in `y`, also
called the labels or targets. If you do this with untrained weights, you’ll
notice that these probabilities are not very good. Then we calculate the log
probabilities or likelihoods by taking the log of each probability. To get the
log likelihood of everything together, we take the mean.

Here’s how to do it in JS in a single loop.

<script>
export function negativeLogLikelihood( probs, ys ) {
    const [m, n] = probs.shape;
    let sum = 0;
    for ( let m_ = m; m_--; ) {
        // Sum the logProbs (log likelihoods) of the correct label.
        sum += Math.log( probs[ m_ * n + ys[ m_ ] ] );
    }
    const mean = sum / m;
    // Mean negative log likelihood.
    return - mean;
}

// Let's keep track of the losses.
export const loss = negativeLogLikelihood( probs, ys );
export const losses = [ loss ];
</script>

# Backward pass to calculate gradients

Now we want to calculate the gradients for the weights matrix. We’ll split this
into two: first calculate the gradients for `Wx`, and then from those the
gradients for `W`.

Now this is not explained in Karpathy’s
[second video](https://href.li/?https://www.youtube.com/watch?v=PaCmpygFfXo),
he’s just using PyTorch’s backward. We can’t do that cause this is hardcore from
scratch! Fortunately, when using the softmax activation combined with
cross-entropy loss, the gradient simplifies to the difference between our
predicted probabilities and the actual labels, which he explains in
[part 4](https://href.li/?https://youtu.be/q8SA3rM6ckI?si=vXBKdMh7sSO44VJT&t=5187).
This is only for a single example, so when spreading the gradient we also need
to divide by the number of rows.

<script>
export function softmaxCrossEntropyGradient( probs, ys ) {
    const [m, n] = probs.shape;
    const gradient = new FloatMatrix( probs );
    for ( let m_ = m; m_--; ) {
        // Subtract 1 for the gradient of the correct label.
        gradient[ m_ * n + ys[ m_ ] ] -= 1;
        for ( let n_ = n; n_--; ) {
            // Divide by the number of rows.
            gradient[ m_ * n + n_ ] /= m;
        }
    }
    return gradient;
}

export const WxGradient = softmaxCrossEntropyGradient( probs, ys );
</script>

Now the easier one: the derivative of a matrix multiplication `A * B` with
respect to the second parameter is `dB = A^T * dAB`.

<script>
export function transpose( A ) {
    const [ m, n ] = A.shape;
    const B = new FloatMatrix( null, [ n, m ] );

    for ( let m_ = m; m_--; ) {
        for ( let n_ = n; n_--; ) {
            B[n_ * m + m_] = A[m_ * n + n_];
        }
    }

    return B;
}

export const WGradient = matMul( transpose( XOneHot ), WxGradient );
</script>

And finally, we can update the weights using these gradients! Let’s first
set a learning rate of 10.

<script>
export const learningRate = 10;
</script>

<script>
for ( let i = W.length; i--; ) W[ i ] -= learningRate * WGradient[ i ];
print( W );
</script>

If we now calculate the loss again, it should be lower!

<script>
const newProbs = softmaxByRow( matMul( XOneHot, W ) );
export const newLoss = negativeLogLikelihood( newProbs, ys );
print( loss, 'oldLoss' );
</script>

## Iterate

Now we need to iterate over this many times.

<script>
export async function iteration() {
    const Wx = await matMul( XOneHot, W );
    const probs = softmaxByRow( Wx );

    losses.push( negativeLogLikelihood( probs, ys ) );

    // Backpropagation.
    const WxGradient = softmaxCrossEntropyGradient( probs, ys );
    const WGradient = await matMul( transpose( XOneHot ), WxGradient );

    for ( let i = W.length; i--; ) W[ i ] -= learningRate * WGradient[ i ];
}
</script>

Just so we can visualise this better, let’s create two graphs using
[Plotly](https://plotly.com/javascript/):

* The losses for each iteration.
* A table with all bigram combinations, darker means a higher occurrence.

<script>
export { default as Plotly } from 'https://cdn.jsdelivr.net/npm/plotly.js-dist@2.26.2/+esm';
export const graphs = document.createElement('div');
graphs.append( document.createElement('div') );
graphs.append( document.createElement('div') );
graphs.style.display = 'flex';
</script>

For each iteration, let’s update the graphs.

<script data-iterations="10" id="iteration">
await iteration();
await Plotly.react(
    graphs.firstChild,
    [ { x: losses.map( ( _, i ) => i ), y: losses } ],
    {
        width: 500, height: 500,
        yaxis: { title: 'Loss', type: 'log' },
        xaxis: { title: 'Iterations' }
    },
    { displayModeBar: false }
);
const r = indexToCharMap.map( ( _, i ) => i )
await Plotly.react(
    graphs.lastChild,
    [ {
        x: indexToCharMap, y: indexToCharMap,
        z: r.map((_, m_) => r.map((_, n_) => Math.exp(W[m_ * r.length + n_]))),
        type: 'heatmap', showscale: false,
        colorscale: [ [ 0, 'white' ], [ 1, 'black' ] ],
    } ],
    {
        width: 500, height: 500,
        yaxis: { tickvals: [], autorange: 'reversed' },
        xaxis: { tickvals: [], },
        margin: { t: 10, b: 10, l: 10, r: 10 },
        annotations: r.map((_, m_) => r.map((_, n_) => ({
            x: indexToCharMap[n_],
            y: indexToCharMap[m_],
            text: `${indexToCharMap[m_]}${indexToCharMap[n_]}`,
            showarrow: false,
            font: { color: 'white' }
        }))).flat(),
    },
    { displayModeBar: false }
);
export default graphs;
</script>

This works, but it’s very slow!

## Bonus! GPU!

Running matrix multiplication on the CPU is slow. Even running it unoptimised on
the GPU will be much faster. It took me a while to recreate matrix
multiplication with WebGPU. After crashing my computer a few times (I swear it
did!), I ended up with some code that works. Here’s the
[code](https://href.li/?https://github.com/ellatrix/micrograd/blob/main/matmul-gpu.js),
and the
[shader](https://href.li/?https://github.com/ellatrix/micrograd/blob/main/matmul.wgsl).
No external libraries, so we’re not cheating. This only works in Chrome and Edge
though.

<script>
const { GPU } = await import( new URL( './matmul-gpu.js', location ) );
export const matMul = ( await GPU() )?.matMul || matMul;
</script>

Let's now go [back to iterating](#iteration), set it to <mark>200</mark>, and
run it again until the loss is below <mark>2.5</mark>.

## Sampling

To sample from the model, we need to input a single character that is
one-hot encoded. Matrix multiply that with the weights, then calculate the
probabilities as before. Then we can take a random number between 0 and 1
and pick from the probabilities. Higher probabilities have a larger “surface
area” in the sample function.

<script data-iterations="10">
function sample(probs) {
    const sample = Math.random();
    let total = 0;
    for ( let i = probs.length; i--; ) {
        total += probs[ i ];
        if ( sample < total ) return i;
    }
}

const indices = [ 0 ];

do {
    const context = indices.slice( -1 );
    const Wc = await matMul( oneHot( context, indexToCharMap.length ), W );
    const probs = softmaxByRow( Wc );
    indices.push( sample( probs ) );
} while ( indices[ indices.length - 1 ] );

export const name = indices.slice( 1, -1 ).map( ( i ) => indexToCharMap[ i ] ).join( '' );
</script>

The output is not great, but it's also not completely random. If you're
curious, run this notebook again without training and you'll see that
sampling with untrained weights is completely random.

In the next notebook we'll see how to enlarge the network, get a better
loss, and better samples.

