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
    // Reshape a shallow subarray, not the original!
    const flatA = A.subarray().reshape([restSize, K]);

    return [
        (await matMul(flatA, B, false, false, bias)).reshape([...restDims, N]),
        async ( grad ) => {
            // Reshape a shallow subarray, not the original!
            const flatGrad = grad.subarray().reshape([restSize, N]);
            const flatA = A.subarray().reshape([restSize, K]);
            const out = await Promise.all([
                matMul(flatGrad, B, false, true),
                matMul(flatA, flatGrad, true, false)
            ]).then(([flatGradA, flatGradB]) => [
                flatGradA.reshape([...restDims, K]),
                flatGradB.reshape([K, N])
            ]);
            if ( bias ) {
                const biasGrad = createFloatMatrix( [ N ] );
                for ( let m_ = restSize; m_--; ) {
                    for ( let n_ = N; n_--; ) {
                        biasGrad[ n_ ] += grad[ m_ * N + n_ ];
                    }
                }
                out.push( biasGrad );
            }
            return out;
        },
    ];
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
    #params = [];
    constructor( fan_in, fan_out, bias = true ) {
        this.#params = [
            new Value( createFloatMatrix( [ fan_in, fan_out ], () => random() / fan_in ** 0.5 ) )
        ];
        if ( bias ) { 
            this.#params.push( new Value( createFloatMatrix( [ fan_out ] ) ) );
        }
    }
    apply( X ) {
        return X.matMulBiasBroadcast( ...this.#params );
    }
    params() {
        return this.#params;
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



