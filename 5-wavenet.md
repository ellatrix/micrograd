---
layout: default
title: '5. makemore: WaveNet'
permalink: '/makemore-wavenet'
---

Deeper network similar to WaveNet. Paper: https://arxiv.org/pdf/1609.03499. It's an autoregressive model and tries to predict the next character in a sequence.

Let's implement it.

We need to modify matMulBias to ignore the inner dimensions:

<script data-src="utils.js">
import { random, transpose } from './1-bigram-utils.js';
import { matMul, FloatMatrix, createFloatMatrix, Value } from './3-0-makemore-MLP-utils.js';

Value.addOperation( 'matMulBiasBroadcast', async ( A, B, bias ) => {
    const K = A.shape.at(-1);
    const restDims = A.shape.slice(0, -1);
    const [k2, N] = B.shape;

    if (K !== k2) {
        throw new Error(`Shape mismatch: A.shape=[${A.shape}], B.shape=[${B.shape}]`);
    }

    const restSize = restDims.reduce((a, b) => a * b, 1);
    const flatA = new FloatMatrix(A).reshape( [restSize, K] );
    const result = new FloatMatrix(await matMul(flatA, B)).reshape( [...restDims, N] );

    if ( bias ) {
        if ( N !== bias.length ) {
            throw new Error('Bias vector dimension does not match the resulting matrix rows.');
        }

        // Add the biases to every row.
        for ( let m_ = restSize; m_--; ) {
            for ( let n_ = N; n_--; ) {
                result[ m_ * N + n_ ] += bias[ n_ ];
            }
        }
    }

    const out = [
        result,
        async ( grad ) => {
            const flatGrad = new FloatMatrix(grad).reshape( [restSize, N] );
            const flatGradA = await matMul(flatGrad, transpose(B));
            return new FloatMatrix(flatGradA).reshape( [...restDims, K] );
        },
        async ( grad ) => {
            const flatGrad = new FloatMatrix(grad).reshape( [restSize, N] );
            const flatGradB = await matMul(transpose(flatA), flatGrad);
            return new FloatMatrix(flatGradB).reshape( [K, N] );
        }
    ];

    if ( bias ) {
        out.push( ( grad ) => {
            const B = createFloatMatrix( [ N ] );
            for ( let m_ = restSize; m_--; ) {
                for ( let n_ = N; n_--; ) {
                    B[ n_ ] += grad[ m_ * N + n_ ];
                }
            }
            return B;
        } );
    }

    return out;
} );

// print( (await matMulBroadcast( new FloatMatrix( random, [ 4, 5, 80 ] ), new FloatMatrix( random, [ 80, 200 ] ) ) ).shape ) // [ 4, 5, 200 ];
</script>

<script>
import { random } from './1-bigram-utils.js';
import { buildDataSet, shuffle, Value, createFloatMatrix, miniBatch, createLossesGraph } from './3-0-makemore-MLP-utils.js';
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
const nHidden = 68;

const n1 = Math.floor( names.length * 0.8 );
const n2 = Math.floor( names.length * 0.9 );
const [ Xtr, Ytr ] = buildDataSet( names.slice( 0, n1 ), stringToCharMap, blockSize );
const [ Xdev, Ydev ] = buildDataSet( names.slice( n1, n2 ), stringToCharMap, blockSize );
const [ Xte, Yte ] = buildDataSet( names.slice( n2 ), stringToCharMap, blockSize );

</script>

<script data-src="utils.js">
export class FlattenConsecutive {
    constructor( n ) {
        this.n = n;
    }
    apply( X ) {
        return X.reshape( ( [ b, t, c ] ) => {
            return t / this.n === 1 ? [ b, c * this.n ] : [ b, t / this.n, c * this.n ];
        });
    }
    params() {
        return [];
    }
}
export class LinearBroadcast {
    constructor( fan_in, fan_out, bias = true ) {
        this.weight = new Value( createFloatMatrix( [ fan_in, fan_out ], () => random() / fan_in ** 0.5 ) );
        if ( bias ) {
            this.bias = new Value( createFloatMatrix( [ fan_out ], () => 0 ) );
        }
    }
    apply( X ) {
        return X.matMulBiasBroadcast( this.weight, this.bias );
    }
    params() {
        return this.bias ? [ this.weight, this.bias ] : [ this.weight ];
    }
}
</script>

<script>
import { Linear, BatchNorm1d, Tanh, Embedding, Flatten, Sequential } from './3-4-layer-organisation-utils.js';
export { default as Plotly } from 'https://cdn.jsdelivr.net/npm/plotly.js-dist@2.26.2/+esm';

const model = new Sequential([
    new Embedding( vocabSize, nEmbed ),
    new FlattenConsecutive( 2 ), new LinearBroadcast( nEmbed * 2, nHidden ), new BatchNorm1d( nHidden ), new Tanh(),
    new FlattenConsecutive( 2 ), new LinearBroadcast( nHidden * 2, nHidden ), new BatchNorm1d( nHidden ), new Tanh(),
    new FlattenConsecutive( 2 ), new LinearBroadcast( nHidden * 2, nHidden ), new BatchNorm1d( nHidden ), new Tanh(),
    new LinearBroadcast( nHidden, vocabSize ),
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
for ( let i = 0; i < 200; i++ ) {
    const [ Xbatch, Ybatch ] = miniBatch( Xtr, Ytr, batchSize );
    const logits = model.apply( Xbatch );
    const loss = logits.softmaxCrossEntropy( Ybatch );
    await loss.forward();
    console.log(loss.data);
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



