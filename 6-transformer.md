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
import { random, sample, softmax, softmaxByRow, transpose } from './1-bigram-utils.js';
import { buildDataSet, Value, miniBatch, createLossesGraph, matMul, FloatMatrix, IntMatrix, createFloatMatrix } from './3-0-makemore-MLP-utils.js';

const n = Math.floor( text.length * 0.9 );
const trainData = new IntMatrix( encode( text.slice( 0, n ) ) ).reshape( [ n ] );
const valData = new IntMatrix( encode( text.slice( n ) ) ).reshape( [ text.length - n ] );
</script>

<script>
const blockSize = 8;
const batchSize = 4;

function getBatch( split ) {
    const data = split === 'train' ? trainData : valData;
    const ix = Array.from( { length: batchSize }, () => Math.floor( Math.random() * ( data.length - blockSize ) ) );
    return [
        new IntMatrix( ix.flatMap( ( i ) => Array.from( data ).slice( i, i + blockSize ) ) ).reshape( [ batchSize, blockSize ] ),
        new IntMatrix( ix.flatMap( ( i ) => Array.from( data ).slice( i + 1, i + blockSize + 1 ) ) ).reshape( [ batchSize, blockSize ] )
    ];
}

const [ x, y ] = getBatch( 'train' );
</script>

<script>
import { Embedding, Linear, Sequential } from './3-4-layer-organisation-utils.js'; 
import { LinearBroadcast } from './5-wavenet-utils.js';

// function tril( A, replacement = 0 ) {
//     // Set the upper triangle to replacement.
//     for ( let i = A.length; i--; ) {
//         for ( let j = i + 1; j < A.length; j++ ) {
//             A[ i ][ j ] = replacement;
//         }
//     }
//     return A;
// }

Value.addOperation( 'attentionHead', async (
    k, // (B, T, C)
    q, // (B, T, C)
    v, // (B, T, C)
) => {
    const [ B, T, C ] = k.shape;
    const scale = C ** -0.5;
    const wei = createFloatMatrix( [ B, T, T ] );
    const out = createFloatMatrix( [ B, T, C ] );
    for ( let b_ = B; b_--; ) {
        const start = b_ * T * C;
        const end = start + T * C;
        const qBatch = q.subarray( start, end ).reshape( [ T, C ] );
        const kBatch = k.subarray( start, end ).reshape( [ T, C ] );
        // (B, T, C) @ ( (B, T, C) -> (B, C, T) ) -> (B, T, T)
        wei.set( await matMul( qBatch, transpose( kBatch ) ), b_ * T * T );
        // Clamp to -Infinity the upper right triangle.
        const offset = b_ * T * T;
        for ( let t_ = T; t_--; ) {
            const t_offset = offset + t_ * T;
            // We could avoid scaling where we set to -Infinity.
            for ( let t2_ = T; t2_--; ) {
                wei[ t_offset + t2_ ] *= scale;
            }
            for ( let t2_ = t_ + 1; t2_ < T; t2_++ ) {
                wei[ t_offset + t2_ ] = -Infinity;
            }
            softmax( wei.subarray( t_offset, t_offset + T ) );
        }
        // (B, T, T) @ (B, T, C) -> (B, T, C)
        out.set(
            await matMul(
                wei.subarray( b_ * T * T, (b_ + 1) * T * T ).reshape( [ T, T ] ),
                v.subarray( b_ * T * C, (b_ + 1) * T * C ).reshape( [ T, C ] )
            ),
            b_ * T * C
        );
    }
    return [out];
});

const nEmbed = 32;
const headSize = 16;

export class Head {
    constructor( nEmbed, headSize ) {
        this.K = new LinearBroadcast( nEmbed, headSize );
        this.Q = new LinearBroadcast( nEmbed, headSize );
        this.V = new LinearBroadcast( nEmbed, headSize );
    }
    apply( X ) {
        const k = this.K.apply( X );
        const q = this.Q.apply( X );
        const v = this.V.apply( X );
        return k.attentionHead( q, v );
    }
    params() {
        return [ ...this.K.params(), ...this.Q.params(), ...this.V.params() ];
    }
}

const model = new Sequential([
    new Embedding( vocabSize, nEmbed ),
    new Head( nEmbed, headSize ),
]);

const out = model.apply( x );
await out.forward();
print( out.data );
</script>

<script>
const model = new Sequential([
    new Embedding( vocabSize, nEmbed ),
    new Linear( nEmbed, vocabSize ),
]);

const logits = model.apply( x );
const loss = logits
    .reshape( ( [ B, T, C ] ) => [ B * T, C ] )
    .softmaxCrossEntropy( new IntMatrix( y ).reshape( [ y.length ] ) );
await loss.forward();
print( loss.data );
</script>

<script>
async function generate( seed, length ) {
    let out = encode( seed );
    
    while ( out.length < length ) {
        const logits = model
            .apply( new IntMatrix( out ).reshape( [ 1, out.length ] ) )
            .reshape( ( [ B, T, C ] ) => [ B * T, C ] );
        await logits.forward();
        const probs = softmaxByRow( logits.data );
        const [ B, C ] = probs.shape;
        const samples = createFloatMatrix( [ B, 1 ] );
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
for ( let i = 0; i < 2; i++ ) {
    const [ x, y ] = getBatch( 'train' );
    const logits = model.apply( x );
    const loss = logits
        .reshape( ( [ B, T, C ] ) => [ B * T, C ] )
        .softmaxCrossEntropy( new IntMatrix( y ).reshape( [ y.length ] ) );
    await loss.forward();
    batchLosses.push( loss.data );

    await loss.backward();
    for ( const param of model.params() ) {
        for ( let i = param.data.length; i--; ) {
            param.data[ i ] -= 0.01 * param.grad[ i ];
        }
    }
    await createLossesGraph( graph, batchLosses, losses );
}
</script>
