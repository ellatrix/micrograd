---
layout: default
title: '3.2. makemore: Initialisation'
permalink: '/makemore-initialisation'
---

<script>
const { random } = await import( new URL( './1-bigram-utils.js', location ) );
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
    embeddingDimensions: 3,
    blockSize: 4,
    neurons: 100,
    batchSize: 32,
    learningRate: 0.1,
};

shuffle( names );
const n1 = Math.floor( names.length * 0.8 );
const n2 = Math.floor( names.length * 0.9 );
const [ Xtr, Ytr ] = buildDataSet( names.slice( 0, n1 ), stringToCharMap, hyperParameters.blockSize );
const [ Xdev, Ydev ] = buildDataSet( names.slice( n1, n2 ), stringToCharMap, hyperParameters.blockSize );
const [ Xte, Yte ] = buildDataSet( names.slice( n2 ), stringToCharMap, hyperParameters.blockSize );
const totalChars = indexToCharMap.length;

function createNetwork() {
    const { embeddingDimensions, blockSize, neurons } = hyperParameters;
    const C = new Value( random( [ totalChars, embeddingDimensions ] ) );
    const W1 = new Value( random( [ embeddingDimensions * blockSize, neurons ] ) );
    const b1 = new Value( random( [ neurons ] ) );
    const W2 = new Value( random( [ neurons, totalChars ] ) );
    const b2 = new Value( random( [ totalChars ] ) );
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
for ( let i = 0; i < 100; i++ ) {
    const [ Xbatch, Ybatch ] = miniBatch( Xtr, Ytr, hyperParameters.batchSize );
    const loss = network( Xbatch ).softmaxCrossEntropy( Ybatch );
    await loss.forward();
    batchLosses.push( loss.data );
    await loss.backward();
    for ( const param of network.params ) {
        for ( let i = param.data.length; i--; ) {
            param.data[ i ] -= hyperParameters.learningRate * param.grad[ i ];
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

We want logits to be uniformly 0. Weights should be sampled from N(0, 1/sqrt(n)). Need entropy for symmetry breaking.

### Tanh too saturated

Lot's of -1 and 1 preactivations. tanh backward is 1 - tanh^2 so it stops the backpropagation.
Vanishing gradients.

Kaiming He, 2020.

We need a slight gain because the tanh is a squashing function. 5/3.
