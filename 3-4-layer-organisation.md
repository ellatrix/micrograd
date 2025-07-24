---
layout: default
title: '3.4. makemore: Layer Organisation'
permalink: '/makemore-layer-organisation'
---

Let's build some utilities to make it easier to build deeper neural networks.

Now we can track the running mean and standard deviation in the batch norm
layer.

<script data-src="utils.js">
import { random } from './1-bigram-utils.js';
import { Value, FloatMatrix, createFloatMatrix } from './3-0-makemore-MLP-utils.js';
import './3-3-batch-norm-utils.js';

export class Linear {
    constructor( fan_in, fan_out, bias = true ) {
        this.weight = new Value( createFloatMatrix( [ fan_in, fan_out ], () => random() / fan_in ** 0.5 ) );
        if ( bias ) {
            this.bias = new Value( createFloatMatrix( [ fan_out ], () => 0 ) );
        }
    }
    apply( X ) {
        return X.matMulBias( this.weight, this.bias );
    }
    params() {
        return this.bias ? [ this.weight, this.bias ] : [ this.weight ];
    }
}
export class BatchNorm1d {
    constructor( dim ) {
        this.gain = new Value( createFloatMatrix( [ dim ], () => 1 ) );
        this.bias = new Value( createFloatMatrix( [ dim ], () => 0 ) );
    }
    apply( X ) {
        return X.batchNorm( this.gain, this.bias );
    }
    params() {
        return [ this.gain, this.bias ];
    }
}
export class Tanh {
    apply( X ) {
        return X.tanh();
    }
    params() {
        return [];
    }
}
export class Embedding {
    constructor( vocabSize, embeddingDimensions ) {
        this.weight = new Value( createFloatMatrix( [ vocabSize, embeddingDimensions ], random ) );
    }
    apply( X ) {
        return this.weight.gather( X );
    }
    params() {
        return [ this.weight ];
    }
}
export class Flatten {
    constructor() {}
    apply( X ) {
        return X.reshape( ( [ first, ...rest ] ) => [ first, rest.reduce( ( acc, curr ) => acc * curr, 1 ) ] );
    }
    params() {
        return [];
    }
}
export class Sequential {
    constructor( layers ) {
        this.layers = layers;
    }
    apply( X ) {
        return this.layers.reduce( ( acc, layer ) => layer.apply( acc ), X );
    }
    params() {
        return this.layers.flatMap( layer => layer.params() );
    }
}
</script>

<script>
import { random } from './1-bigram-utils.js';
import { buildDataSet, shuffle, Value, miniBatch, createLossesGraph } from './3-0-makemore-MLP-utils.js';
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
const blockSize = 4;
const nHidden = 200;

const n1 = Math.floor( names.length * 0.8 );
const n2 = Math.floor( names.length * 0.9 );
const [ Xtr, Ytr ] = buildDataSet( names.slice( 0, n1 ), stringToCharMap, blockSize );
const [ Xdev, Ydev ] = buildDataSet( names.slice( n1, n2 ), stringToCharMap, blockSize );
const [ Xte, Yte ] = buildDataSet( names.slice( n2 ), stringToCharMap, blockSize );

</script>

Now we can more easily stack layers.

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

To do: build graph with multiple linear layers, see part 3 video.