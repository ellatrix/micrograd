---
layout: default
title: '5. makemore: WaveNet'
permalink: '/makemore-wavenet'
---

Deeper network similar to WaveNet. Paper: https://arxiv.org/pdf/1609.03499. It's an autoregressive model and tries to predict the next character in a sequence.

Let's implement it.

We need to modify matMulBias to ignore the inner dimensions:

<script>
print( (await matMulBias( new FloatMatrix( random, [ 4, 5, 6, 80 ] ), new FloatMatrix( random, [ 80, 200 ] ) ) ).shape ) // [ 4, 5, 6, 200 ];
</script>

<script>
import { random } from './1-bigram-utils.js';
import { buildDataSet, shuffle, Value, FloatMatrix, miniBatch, createLossesGraph } from './3-0-makemore-MLP-utils.js';
const response = await fetch('https://raw.githubusercontent.com/karpathy/makemore/master/names.txt');
const text = await response.text();
const names = text.split('\n');
const indexToCharMap = [ '.', ...new Set( names.join('') ) ].sort();
const vocabSize = indexToCharMap.length;
const stringToCharMap = {};

for ( let i = indexToCharMap.length; i--; ) {
    stringToCharMap[ indexToCharMap[ i ] ] = i;
}

shuffle( names );

// Hyperparameters
const nEmbed = 10;
const blockSize = 8;
const nHidden = 200;

const n1 = Math.floor( names.length * 0.8 );
const n2 = Math.floor( names.length * 0.9 );
const [ Xtr, Ytr ] = buildDataSet( names.slice( 0, n1 ), stringToCharMap, blockSize );
const [ Xdev, Ydev ] = buildDataSet( names.slice( n1, n2 ), stringToCharMap, blockSize );
const [ Xte, Yte ] = buildDataSet( names.slice( n2 ), stringToCharMap, blockSize );

</script>

<script data-src="utils.js">
import { random } from './1-bigram-utils.js';
import { Value, FloatMatrix } from './3-0-makemore-MLP-utils.js';
</script>

<script>
import { Linear, BatchNorm1d, Tanh, Embedding, Flatten, Sequential } from './3-4-layer-organisation-utils.js';
export { default as Plotly } from 'https://cdn.jsdelivr.net/npm/plotly.js-dist@2.26.2/+esm';

const model = new Sequential([
    new Embedding( vocabSize, nEmbed ),
    new Flatten(),
    new Linear( nEmbed * blockSize, nHidden ), new BatchNorm1d( nHidden ), new Tanh(),
    new Linear( nHidden, vocabSize ),
]);

// Scale down weights to 0.01 to be less confident.
for (let w = model.layers.at(-1).weight.data, i = w.length; i--;) w[i] *= 0.1;

print( model.params(), 'Parameters' );
print( model.params().reduce( ( acc, param ) => acc + param.data.length, 0 ), 'Number of parameters' );

const batchLosses = [];
const losses = [];
const batchSize = 64;
</script>

<script>
const graph = document.createElement( 'div' );
print(graph);
for ( let i = 0; i < 1000; i++ ) {
    const [ Xbatch, Ybatch ] = miniBatch( Xtr, Ytr, batchSize );
    const logits = model.apply( Xbatch );
    const loss = logits.softmaxCrossEntropy( Ybatch );
    await loss.forward();
    batchLosses.push( loss.data );

    await loss.backward();
    const learningRate = batchLosses.length < 2000 ? 0.1 : 0.01;
    for ( const param of model.params() ) {
        for ( let i = param.data.length; i--; ) {
            param.data[ i ] -= learningRate * param.grad[ i ];
        }
    }

    if ( batchLosses.length % 100 === 0 ) {
        // Take the mean of the last 100 losses.
        const meanLoss = batchLosses.slice( -100 ).reduce( ( acc, curr ) => acc + curr, 0 ) / 100;
        losses.push( meanLoss );
    }

    await createLossesGraph( graph, batchLosses, losses );
}
</script>



