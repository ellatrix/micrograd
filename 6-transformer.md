---
layout: default
title: '6. Transformer'
permalink: '/transformer'
---

Paper: https://arxiv.org/abs/1706.03762

Let's try building an training a transformer-based language model. We're not
going to train on a chunk of the internet, we need a smaller dataset. We'll also
keep it character based.

Let's fetch the data.

<script>
const response = await fetch('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt');
const text = await response.text();
</script>

Now let's create a way to encode and decode the text to integers.

<script>
const indexToCharMap = [ ...new Set( text ) ].sort();
const stringToCharMap = {};
for ( let i = indexToCharMap.length; i--; ) {
    stringToCharMap[ indexToCharMap[ i ] ] = i;
}
const vocabSize = indexToCharMap.length;

function encode( text ) {
    return [ ...text ].map( ( char ) => stringToCharMap[ char ] );
}

function decode( indices ) {
    return indices.map( ( index ) => indexToCharMap[ index ] ).join('');
}
</script>

Now we encode the text.

<script>
import { random, sample, softmaxByRow } from './1-bigram-utils.js';
import { buildDataSet, Value, IntMatrix, miniBatch, createLossesGraph } from './3-0-makemore-MLP-utils.js';
const n = Math.floor( text.length * 0.9 );
const trainData = new IntMatrix( encode( text.slice( 0, n ) ), [ n ] );
const valData = new IntMatrix( encode( text.slice( n ) ), [ text.length - n ] );
</script>

<script>
const blockSize = 8;
const batchSize = 4;

function getBatch( split ) {
    const data = split === 'train' ? trainData : valData;
    const ix = Array.from( { length: batchSize }, () => Math.floor( Math.random() * ( data.length - blockSize ) ) );
    return [
        new IntMatrix( ix.flatMap( ( i ) => Array.from( data ).slice( i, i + blockSize ) ), [ batchSize, blockSize ] ),
        new IntMatrix( ix.flatMap( ( i ) => Array.from( data ).slice( i + 1, i + blockSize + 1 ) ), [ batchSize, blockSize ] )
    ];
}

const [ x, y ] = getBatch( 'train' );
</script>

<script>
import { Linear, BatchNorm1d, Tanh, Embedding, Flatten, Sequential } from './3-4-layer-organisation-utils.js';

const model = new Sequential([
    new Embedding( vocabSize, vocabSize ),
]);

const logits = model.apply( x );
const loss = logits
    .reshape( ( [ B, T, C ] ) => [ B * T, C ] )
    .softmaxCrossEntropy( new IntMatrix( y, [ y.length ] ) );
await loss.forward();
print( loss.data );
</script>

<script>
async function generate( seed, length ) {
    let out = encode( seed );
    
    while ( out.length < length ) {
        const logits = model
            .apply( new IntMatrix( out, [ 1, out.length ] ) )
            .reshape( ( [ B, T, C ] ) => [ B * T, C ] );
        await logits.forward();
        const probs = softmaxByRow( logits.data );
        const [ B, C ] = probs.shape;
        const samples = new IntMatrix( null, [ B, 1 ] );
        for ( let i = B; i--; ) {
            samples[ i ] = sample( Array.from( probs ).slice( i * C, ( i + 1 ) * C ) );
        }
        out.push( ...samples );
    }

    console.log( out );

    return decode( out );
}

print( await generate( '\n', 100 ) );
</script>

<script>
import Plotly from 'https://cdn.jsdelivr.net/npm/plotly.js-dist@2.26.2/+esm';

const batchLosses = [];
const losses = [];
</script>

<script>
const graph = document.createElement( 'div' );
print(graph);
for ( let i = 0; i < 10; i++ ) {
    const [ x, y ] = getBatch( 'train' );
    const logits = model.apply( x );
    const loss = logits
        .reshape( ( [ B, T, C ] ) => [ B * T, C ] )
        .softmaxCrossEntropy( new IntMatrix( y, [ y.length ] ) );
    await loss.forward();
    batchLosses.push( loss.data );

    await loss.backward();
    for ( const param of model.params() ) {
        for ( let i = param.data.length; i--; ) {
            param.data[ i ] -= 0.01 * param.grad[ i ];
        }
    }
    await createLossesGraph( graph, batchLosses, losses );
    console.log( i, loss.data, model.params() );
}
</script>
