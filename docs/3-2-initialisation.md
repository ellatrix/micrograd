---
layout: default
title: '3.2. makemore: Initialisation'
permalink: '/makemore-initialisation'
---

<script>
const { random, softmaxByRow, matMul } = await import( new URL( './1-bigram-utils.js', location ) );
const {
    Value,
    FloatMatrix,
    IntMatrix,
    buildDataSet,
    miniBatch,
    shuffle,
    createLossesGraph
} = await import( new URL( './3-0-makemore-MLP-utils.js', location ) );
export { default as Plotly } from 'https://cdn.jsdelivr.net/npm/plotly.js-dist@2.26.2/+esm';
</script>

<script>
const response = await fetch('https://raw.githubusercontent.com/karpathy/makemore/master/names.txt');
const text = await response.text();
const names = text.split('\n');
const indexToCharMap = [ '.', ...new Set( names.join('') ) ].sort();
const stringToCharMap = {};

for ( let i = indexToCharMap.length; i--; ) {
    stringToCharMap[ indexToCharMap[ i ] ] = i;
}

const hyperParameters = {
    embeddingDimensions: 10,
    blockSize: 3,
    neurons: 200,
    batchSize: 32,
    learningRate: 0.1,
};

shuffle( names );
const n1 = Math.floor( names.length * 0.8 );
const n2 = Math.floor( names.length * 0.9 );
const [ Xtr, Ytr ] = buildDataSet( names.slice( 0, n1 ), stringToCharMap, hyperParameters.blockSize );
const [ Xdev, Ydev ] = buildDataSet( names.slice( n1, n2 ), stringToCharMap, hyperParameters.blockSize );
const [ Xte, Yte ] = buildDataSet( names.slice( n2 ), stringToCharMap, hyperParameters.blockSize );
const vocabSize = indexToCharMap.length;

function createNetwork() {
    const { embeddingDimensions, blockSize, neurons } = hyperParameters;
    const C = new Value( new FloatMatrix( random, [ vocabSize, embeddingDimensions ] ) );
    const W1 = new Value( new FloatMatrix( random, [ embeddingDimensions * blockSize, neurons ] ) );
    const b1 = new Value( new FloatMatrix( random, [ neurons ] ) );
    const W2 = new Value( new FloatMatrix( random, [ neurons, vocabSize ] ) );
    const b2 = new Value( new FloatMatrix( random, [ vocabSize ] ) );
    function logitFn( X ) {
        const embedding = C.gather( X ).reshape( [ X.shape[ 0 ], embeddingDimensions * blockSize ] );
        const hidden = embedding.matMulBias( W1, b1 ).tanh();
        return hidden.matMulBias( W2, b2 );
    }
    logitFn.params = [ C, W1, b1, W2, b2 ];
    return logitFn;
}
const batchLosses = [];
const losses = [];
const network = createNetwork();
</script>

<script>
const graph = document.createElement( 'div' );
print(graph);
for ( let i = 0; i < 1000; i++ ) {
    const [ Xbatch, Ybatch ] = miniBatch( Xtr, Ytr, hyperParameters.batchSize );
    const loss = network( Xbatch ).softmaxCrossEntropy( Ybatch );
    await loss.forward();
    batchLosses.push( loss.data );
    await loss.backward();
    const learningRate = batchLosses.length < 2000 ? 0.1 : 0.01;
    for ( const param of network.params ) {
        for ( let i = param.data.length; i--; ) {
            param.data[ i ] -= learningRate * param.grad[ i ];
        }
    }

    if ( batchLosses.length % 100 === 0 ) {
        const loss = network( Xdev ).softmaxCrossEntropy( Ydev );
        await loss.forward();
        losses.push( loss.data );
    }

    await createLossesGraph( graph, batchLosses, losses );
}
</script>

The network is very unproperly configured at initialization. What initial loss
would we expect? The probability of any one character should be roughly 1/vocabSize. It should be a uniform distribution. So the loss we would expect is -log(1/vocabSize).

<script>
print( -Math.log( 1 / 27 ) );
</script>

Yet the inital loss is much higher.

Let's say we have a smaller network where 4 logits come out.

<script>
const logits = new FloatMatrix( [ 0, 0, 0, 0 ], [ 1, 4 ] );
print( softmaxByRow( logits ) );
</script>

The softmax of these logits gives us a probability distribution, and we can see
that it is exactly uniform if all logits are the same.

Let's initialize a new network and check the values of the logits. Perhaps run
it a few to get some variation.

<script>
const logits = createNetwork()( Xdev );
await logits.forward();
print( logits.data );
</script>

As you can see the logits can take on large values and be quite spread out, 
which drives up the loss.

So how can we decrease it? The last step of the network is a matMul, so we
should make sure the weights are closer to zero. We can initialize the bias with
zero.

<script>
function createNetwork() {
    const { embeddingDimensions, blockSize, neurons } = hyperParameters;
    const C = new Value( new FloatMatrix( random, [ vocabSize, embeddingDimensions ] ) );
    const W1 = new Value( new FloatMatrix( random, [ embeddingDimensions * blockSize, neurons ] ) );
    const b1 = new Value( new FloatMatrix( random, [ neurons ] ) );
    const W2 = new Value( new FloatMatrix( () => random() * 0.01, [ neurons, vocabSize ] ) );
    const b2 = new Value( new FloatMatrix( null, [ vocabSize ] ) );
    function logitFn( X ) {
        const embedding = C.gather( X ).reshape( [ X.shape[ 0 ], embeddingDimensions * blockSize ] );
        const hidden = embedding.matMulBias( W1, b1 ).tanh();
        return hidden.matMulBias( W2, b2 );
    }
    logitFn.params = [ C, W1, b1, W2, b2 ];
    return logitFn;
}
</script>

Let's run it again.

<script>
const logits = createNetwork()( Xdev );
await logits.forward();
print( logits.data );
</script>

Let's check the loss.

<script>
const loss = createNetwork()( Xdev ).softmaxCrossEntropy( Ydev );
await loss.forward();
print( loss.data );
</script>

This is very close to what we're expecting!

Can we actually set the weights to zero as well? Then we'd get exactly what
we're looking for. But we need some entropy in the weights to break the symmetry
and allow for learning.

<script>
const batchLosses = [];
const losses = [];
const network = createNetwork();
</script>

<script>
const graph = document.createElement( 'div' );
print(graph);
for ( let i = 0; i < 1000; i++ ) {
    const [ Xbatch, Ybatch ] = miniBatch( Xtr, Ytr, hyperParameters.batchSize );
    const loss = network( Xbatch ).softmaxCrossEntropy( Ybatch );
    await loss.forward();
    batchLosses.push( loss.data );
    await loss.backward();
    const learningRate = batchLosses.length < 2000 ? 0.1 : 0.01;
    for ( const param of network.params ) {
        for ( let i = param.data.length; i--; ) {
            param.data[ i ] -= learningRate * param.grad[ i ];
        }
    }

    if ( batchLosses.length % 100 === 0 ) {
        const loss = network( Xdev ).softmaxCrossEntropy( Ydev );
        await loss.forward();
        losses.push( loss.data );
    }

    await createLossesGraph( graph, batchLosses, losses );
}
</script>

Great, now we have much less of a hockey stick graph, and waste less training
time.

But there's still a deeper issue at initialization. The logits are now ok, but
the problem now is with the output of the hidden layer. Let's have a look at the
histogram.

<script>
const graph = document.createElement( 'div' );
const graph2 = document.createElement( 'div' );
print( graph );
print( graph2 );
const [ X ] = miniBatch( Xtr, Ytr, hyperParameters.batchSize );
const { embeddingDimensions, blockSize, neurons } = hyperParameters;
const C = new Value( new FloatMatrix( random, [ vocabSize, embeddingDimensions ] ) );
const W1 = new Value( new FloatMatrix( random, [ embeddingDimensions * blockSize, neurons ] ) );
const b1 = new Value( new FloatMatrix( random, [ neurons ] ) );
const embedding = C.gather( X ).reshape( [ X.shape[ 0 ], embeddingDimensions * blockSize ] );
const hidden = embedding.matMulBias( W1, b1 ).tanh();
await hidden.forward();
Plotly.newPlot( graph, [ { x: Array.from( hidden.data ), type: 'histogram' } ] );
Plotly.newPlot( graph2, [{
    z: [...Array(hidden.data.shape[0])].map((_, i) => 
        Array.from(hidden.data).slice(i * hidden.data.shape[1], (i + 1) * hidden.data.shape[1])
        .map(value => value > 0.9 ? 1 : 0)
    ),
    type: 'heatmap',
    colorscale: 'Greys',
    showscale: false
    }], {
    height: 300,
    xaxis: { visible: false },
    yaxis: { visible: false }
},{ displayModeBar: false });
</script>

As you can see, a very large number come out close to 1 or -1. The tanh is very
active (squashing large values), and the output is not uniform.

Why is this a problem? You may recall the the derivative of tanh is 1 - tanh^2.
When the values are 1 or -1, the derivative is 0, we are killing the gradient.

The solution is the same as before: in front of the the tanh activation, we have
a matMul, so if we scale the weights down, the preactivations will be closer to zero,
and the tanh will squash less because there's no extreme values.

<script>
function createNetwork() {
    const { embeddingDimensions, blockSize, neurons } = hyperParameters;
    const C = new Value( new FloatMatrix( random, [ vocabSize, embeddingDimensions ] ) );
    const W1 = new Value( new FloatMatrix( () => random() * 0.2, [ embeddingDimensions * blockSize, neurons ] ) );
    const b1 = new Value( new FloatMatrix( null, [ neurons ] ) );
    const W2 = new Value( new FloatMatrix( () => random() * 0.01, [ neurons, vocabSize ] ) );
    const b2 = new Value( new FloatMatrix( null, [ vocabSize ] ) );
    function logitFn( X ) {
        const embedding = C.gather( X ).reshape( [ X.shape[ 0 ], embeddingDimensions * blockSize ] );
        const hidden = embedding.matMulBias( W1, b1 ).tanh();
        return hidden.matMulBias( W2, b2 );
    }
    logitFn.params = [ C, W1, b1, W2, b2 ];
    return logitFn;
}
</script>

<script>
const batchLosses = [];
const losses = [];
const network = createNetwork();
</script>

<script>
const graph = document.createElement( 'div' );
print(graph);
for ( let i = 0; i < 1000; i++ ) {
    const [ Xbatch, Ybatch ] = miniBatch( Xtr, Ytr, hyperParameters.batchSize );
    const loss = network( Xbatch ).softmaxCrossEntropy( Ybatch );
    await loss.forward();
    batchLosses.push( loss.data );
    await loss.backward();
    const learningRate = batchLosses.length < 2000 ? 0.1 : 0.01;
    for ( const param of network.params ) {
        for ( let i = param.data.length; i--; ) {
            param.data[ i ] -= learningRate * param.grad[ i ];
        }
    }

    if ( batchLosses.length % 100 === 0 ) {
        const loss = network( Xdev ).softmaxCrossEntropy( Ydev );
        await loss.forward();
        losses.push( loss.data );
    }

    await createLossesGraph( graph, batchLosses, losses );
}
</script>

The difference is not very big, but the deeper the network is, the less
forgiving it is to these errors.

How do we know with what value to scale the weights with?

Let's say we random inputs and random weights, both of which would be
initialized at a standard deviation of ~1.

<script>
function standardDeviation(values) {
    const mean = values.reduce((a, b) => a + b) / values.length;
    const variance = values
        .map(x => Math.pow(x - mean, 2))
        .reduce((a, b) => a + b) / values.length;
    return Math.sqrt(variance);
}

const X = new FloatMatrix( random, [ 1000, 10 ] );
const W = new FloatMatrix( random, [ 10, 200 ] );
const Y = matMul( X, W );
print( standardDeviation( Array.from( X ) ) );
print( standardDeviation( Array.from( Y ) ) );
</script>

So how do we scale the weights to preserve the distribution? If we scale them
up, there will be more and more extreme values. If we scale them down, the
standart deviation will shrink. What do we multiply the weights by to exactly
preserve it?

It turns out the correct mathematical answer is to multiply the weights by
1/sqrt( weight number of rows ).

<script>
const W = new FloatMatrix( () => random() / 10**0.5, [ 10, 200 ] );
const Y = matMul( X, W );
print( standardDeviation( Array.from( Y ) ) );
</script>

As you can see, the standard deviation is now ~1 too.

But on top of this we also have the tanh activation, which squashes the values
further.

<script>
const W = new FloatMatrix( () => random() / 10**0.5, [ 10, 200 ] );
const Y = matMul( X, W );
for ( let i = Y.length; i--; ) {
    Y[ i ] = Math.tanh( Y[ i ] );
}
print( standardDeviation( Array.from( Y ) ) );
</script>

So we'll need a slight gain to compensate for this. It turns out that, for tanh,
a gain of 5/3 is needed. Other activation functions may need a different gain.
E.g. relu, which throws away 50% of the values, needs a gain of 2. See Kaiming
He, 2020. This would be called the Kaiming initialization, which we can use
instead of the random scale.

<script>
print( (5/3) / (hyperParameters.embeddingDimensions * hyperParameters.blockSize**0.5) );
</script>

Let's look at a more modern solution, called batch normalization, which removes
the need for such careful initialization.
