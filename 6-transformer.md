---
layout: default
title: '6. Transformer'
permalink: '/transformer'
---

<aside>
    This covers the <a href="https://www.youtube.com/watch?v=kCc8FmEb1nY">Let's build GPT</a> video, except we're doing everything from scratch in vanilla JS!
</aside>

Paper: https://arxiv.org/abs/1706.03762

In this chapter, we're going re-build everything and not use anything from
previous chapters. We're going to all the compute power we can, so we'll build a
new autograd engine around webGPU buffers instead of TypedArrays.

Let's try building an training a transformer-based language model. We're not
going to train on a chunk of the internet, we need a smaller dataset. We'll also
keep it character based. So let's use the tiny Shakespeare dataset, like the
video.

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
import { sample, softmaxByRow, negativeLogLikelihood, softmaxCrossEntropyGradient, random } from './1-bigram-utils.js';
import { createLossesGraph, matMul, batchMatMul, batchSoftmaxRowTril, batchSoftmaxRowTrilBackward, relu, scatterAdd, FloatMatrix, IntMatrix, createFloatMatrix, biasGradSum } from './3-0-makemore-MLP-utils.js';

const n = Math.floor( text.length * 0.9 );
const trainData = encode( text.slice( 0, n ) );
const valData = encode( text.slice( n ) );

import { getTopologicalOrder } from './2-autograd-utils.js';
window.forwardTimes = [];
window.backwardTimes = [];
export class Value {
    static operations = new Map();
    _dependents = [];
    constructor(data, _children = [], _op) {
        this.data = data;
        this._op = _op;
        this._prev = _children;
        for ( const child of this._prev ) {
            if ( child instanceof Value ) child._dependents.push( this );
        }
    }

    static addOperation(operation, shapeFn, forward) {
        this.operations.set(operation, forward);
        this.prototype[operation] = function (...args) {
            return new Value(
                createFloatMatrix(
                    shapeFn(
                        ...[this, ...args]
                        .map(arg => arg instanceof Value ? arg.data : arg)
                    )
                ),
                [this, ...args],
                operation
            );
        };
    }

    async _forward() {
        if (this._forwardReady) return this._forwardReady;

        this._forwardReady = (async () => {
            if (!this._op) {
                if (this.data === null) {
                    throw new Error("Leaf node has no data during forward pass.");
                }
                return this.data;
            }

            const args = this._prev;

            // Wait for all child nodes
            await Promise.all(args.map(arg => arg instanceof Value ? arg._forward() : null));

            const inputData = args.map(arg => arg instanceof Value ? arg.data : arg);

            const opFn = Value.operations.get(this._op);
            if (!opFn) throw new Error(`Missing operation handler for op: ${this._op}`);

            const start = performance.now();
            const [data, calculateGrad] = await opFn(...inputData);
            const end = performance.now();
            window.forwardTimes.push({ label: this._op, start, end });

            this.data = data;

            this._backward = async () => {
                const start = performance.now();
                const grads = await calculateGrad(this.grad);
                for (let i = 0; i < grads.length; i++) {
                    const child = args[i];
                    if (child instanceof Value) {
                        child.grad = child.grad ? add(child.grad, grads[i]) : grads[i];
                    }
                }
                const end = performance.now();
                window.backwardTimes.push({ label: this._op, start, end });
            };

            return data;
        })();

        return this._forwardReady;
    }

    async backward() {
        const reversed = getTopologicalOrder(this).reverse();

        for (const node of reversed) {
            node.grad = null;
        }

        this.grad = createFloatMatrix(this.data.shape ?? [1]).fill(1);

        for (const node of reversed) {
            await node._backward?.();
        }
    }

    forward() {
        const order = getTopologicalOrder(this);

        for (const node of order) {
            delete node._forwardReady;
        }

        return this._forward();
    }
}
</script>

<script>
// Hyperparameters.
const blockSize = 256;
const batchSize = 64;

function getBatch( split ) {
    const data = split === 'train' ? trainData : valData;
    const ix = Array.from( { length: batchSize }, () => Math.floor( Math.random() * ( data.length - blockSize ) ) );
    return [
        new IntMatrix( ix.flatMap( ( i ) => data.slice( i, i + blockSize ) ) ).reshape( [ batchSize, blockSize ] ),
        new IntMatrix( ix.flatMap( ( i ) => data.slice( i + 1, i + blockSize + 1 ) ) ).reshape( [ batchSize, blockSize ] )
    ];
}

const [ x, y ] = getBatch( 'train' );
</script>

<script>
import { Sequential } from './3-4-layer-organisation-utils.js';

Value.addOperation(
    'softmaxCrossEntropy',
    ( A ) => [ 1 ],
    ( A, indices ) => {
        const data = softmaxByRow( A );
        return [
            negativeLogLikelihood( data, indices ),
            () => [ softmaxCrossEntropyGradient( data, indices ) ]
        ];
    }
);

Value.addOperation(
    'reshape',
    ( A ) => A.shape,
    ( A, shape ) => [
        new FloatMatrix( A ).reshape( shape ),
        ( grad ) => [ new FloatMatrix( grad ).reshape( A.shape ) ],
    ]
);

export function gather2d(A, indices) {
    const shape = indices.shape ?? [ indices.length ];
    const Dim = A.shape[1];
    const R = createFloatMatrix( [...shape, Dim] );
    for (let i = indices.length; i--;) {
        const index = indices[i];
        for (let j = Dim; j--;) {
            R[i * Dim + j] = A[index * Dim + j];
        }
    }
    return R;
}

Value.addOperation(
    'gather2d',
    ( A, indices ) => {
        return [ ...(indices.shape ?? [ indices.length ]), A.shape[1] ];
    },
    ( A, indices ) => [
        gather2d( A, indices ),
        async ( grad ) => {
            return [await scatterAdd( grad, indices, A.shape )];
        }
    ]
);

export class Embedding {
    constructor( vocabSize, embeddingDimensions ) {
        this.weight = new Value( createFloatMatrix( [ vocabSize, embeddingDimensions ], random ) );
    }
    apply( X ) {
        return this.weight.gather2d( X );
    }
    params() {
        return [ this.weight ];
    }
}

Value.addOperation(
    'matMulBiasBroadcast',
    ( A, B, bias ) => [ ...A.shape.slice(0, -1), B.shape[1] ],
    async ( A, B, bias ) => {
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
                return await Promise.all([
                    matMul(flatGrad, B, false, true),
                    matMul(flatA, flatGrad, true, false),
                    bias ? biasGradSum(grad, restSize, N) : null
                ]).then(([flatGradA, flatGradB, biasGrad]) => {
                    const grads = [
                        flatGradA.reshape([...restDims, K]),
                        flatGradB.reshape([K, N]),
                    ];
                    if ( bias ) {
                        grads.push( biasGrad );
                    }
                    return grads;
                });
            },
        ];
    }
);

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

Value.addOperation(
    'batchMatMul',
    (A, B, aT, bT) => [
        ...A.shape.slice(0, -2),
        aT ? A.shape.at(-1) : A.shape.at(-2),
        bT ? B.shape.at(-2) : B.shape.at(-1)
    ],
    async ( A, B, aT, bT ) => [
        await batchMatMul(A, B, aT, bT),
        async ( grad ) => {
            return await Promise.all([
                aT ? batchMatMul( B, grad, bT, true) : batchMatMul(grad, B, false, !bT),
                bT ? batchMatMul(grad, A, true, aT) : batchMatMul(A, grad, !aT, false)
            ]);
        }
    ]
);

Value.addOperation(
    'batchSoftmaxRowTril',
    ( In ) => In.shape,
    async ( In ) => {
        const Out = await batchSoftmaxRowTril(In);
        return [
            Out,
            async ( dOut ) => {
                return [await batchSoftmaxRowTrilBackward(dOut, Out)];
            }
        ]
    }
);

function scale( A, scalar ) {
    for ( let i = A.length; i--; ) A[ i ] *= scalar;
    return A;
}

Value.addOperation(
    'scale',
    ( A ) => A.shape,
    async ( A, scalar ) => [
        scale(A, scalar),
        async ( grad ) => [ scale(grad, scalar) ]
    ]
);

export class Head {
    constructor( nEmbed, headSize ) {
        this.K = new LinearBroadcast( nEmbed, headSize, false );
        this.Q = new LinearBroadcast( nEmbed, headSize, false );
        this.V = new LinearBroadcast( nEmbed, headSize, false );
    }
    apply( X ) {
        const k = this.K.apply( X );
        const q = this.Q.apply( X );
        const v = this.V.apply( X );
        return q
            // (B, T, C) @ ( (B, T, C)ᵀ -> (B, C, T) ) -> (B, T, T)
            .batchMatMul( k, false, true )
            .scale( 16 ** -0.5 )
            .batchSoftmaxRowTril()
            // (B, T, T) @ (B, T, C) -> (B, T, C)
            .batchMatMul( v )
    }
    params() {
        return [ ...this.K.params(), ...this.Q.params(), ...this.V.params() ];
    }
}

Value.addOperation(
    'concatLastDim',
    ( ...args ) => {
        const [ first ] = args;
        if ( ! args.every( arg => arg.shape.toString() === first.shape.toString() ) ) {
            throw new Error( 'Shape mismatch: ' + args.map( arg => arg.shape ).join( ', ' ) );
        }
        return [ ...first.shape.slice(0, -1), args.length * first.shape.at(-1) ];
    },
    async (...args) => {
        const n = args.length;
        const [ B, T, C ] = args[0].shape;
        const out = createFloatMatrix([ B, T, n * C ]);

        for (let i = 0; i < n; i++) {
            const src = args[i];
            for (let j = 0; j < B * T; j++) {
                const srcStart = j * C;
                const dstStart = j * n * C + i * C;
                out.set(src.subarray(srcStart, srcStart + C), dstStart);
            }
        }

        return [
            out,
            async (dout) => {
                return args.map((_, i) => {
                    const grad = createFloatMatrix([ B, T, C ]);
                    for (let j = 0; j < B * T; j++) {
                        const srcStart = j * n * C + i * C;
                        const dstStart = j * C;
                        grad.set(dout.subarray(srcStart, srcStart + C), dstStart);
                    }
                    return grad;
                });
            }
        ];
    }
);

function add( A, B ) {
    if ( A.shape.toString() !== B.shape.toString() ) {
        throw new Error('Shape mismatch: a.shape=' + A.shape + ', b.shape=' + B.shape);
    }

    const out = new FloatMatrix(A);
    for (let i_ = out.length; i_--;) out[i_] += B[i_];
    return out;
}

Value.addOperation(
    'add',
    ( a, b ) => {
        if ( a.shape.toString() !== b.shape.toString() ) {
            throw new Error( 'Shape mismatch: a.shape=' + a.shape + ', b.shape=' + b.shape );
        }
        return a.shape;
    },
    async (
        a, // (B, T, C)
        b, // (B, T, C)
    ) => {
        return [ add(a, b), (dout) => [dout, dout] ];
    }
);

class MultiHeadAttention {
    constructor( nEmbed, nHeads, headSize ) {
        this.heads = Array.from( { length: nHeads }, () => new Head( nEmbed, headSize ) );
        this.proj = new LinearBroadcast( nEmbed, nEmbed );
    }
    apply( x ) {
        const heads = this.heads.map( head => head.apply( x ) );
        const out = heads[0].concatLastDim( ...heads.slice(1) );
        return this.proj.apply( out );
    }
    params() {
        return [ ...this.heads.flatMap( head => head.params() ), ...this.proj.params() ];
    }
}

Value.addOperation(
    'expandAndTile',
    ( x, Bsize ) => [ Bsize, ...x.shape ],
    async (
        x,     // shape: (D1, D2, ..., Dn)
        Bsize  // number: B
    ) => {
        const shape = x.shape;
        const D = x.length;
        const out = createFloatMatrix([Bsize, ...shape]);

        for (let b_ = 0; b_ < Bsize; b_++) {
            out.set(x, b_ * D);
        }

        return [
            out,
            async (dout) => {
                const dx = createFloatMatrix(shape);
                for (let b_ = 0; b_ < Bsize; b_++) {
                    const offset = b_ * D;
                    for (let i = 0; i < D; i++) {
                        dx[i] += dout[offset + i];
                    }
                }
                return [dx];
            }
        ];
    }
);

Value.addOperation(
    'relu',
    ( A ) => A.shape,
    async (A) => {
        const out = await relu(A);
        return [
            out,
            (grad) => {
                const dA = new FloatMatrix(grad);
                for (let i = dA.length; i--;) {
                    if ( out[i] === 0 ) {
                        dA[i] = 0;
                    }
                }
                return [dA];
            },
        ];
    }
);

class ReLU {
    apply( X ) {
        return X.relu();
    }
    params() {
        return [];
    }
}

class FeedForward {
    constructor( nEmbed ) {
        this.net = new Sequential([
            new LinearBroadcast( nEmbed, 4 * nEmbed ),
            new ReLU(),
            new LinearBroadcast( 4 * nEmbed, nEmbed ), // Projection.
        ]);
    }
    apply( x ) {
        return this.net.apply( x );
    }
    params() {
        return this.net.params();
    }
}

Value.addOperation(
    'layerNorm',
    (A) => A.shape,
    (A, gain, bias) => {
        const n = A.shape.at(-1);
        const restDims = A.shape.slice(0, -1);
        const m = restDims.reduce((a, b) => a * b, 1);
        const lnraw = new FloatMatrix(A);
        const lnmean = createFloatMatrix([m]);
        const lnvar = createFloatMatrix([m]);
        const lnvarinv = createFloatMatrix([m]);
        const lnout = createFloatMatrix(A.shape);

        // Compute mean per "row"
        for (let i = 0; i < m; i++) {
            let sum = 0;
            for (let j = 0; j < n; j++) {
                sum += A[i * n + j];
            }
            lnmean[i] = sum / n;
        }

        // Compute variance per "row"
        for (let i = 0; i < m; i++) {
            let varSum = 0;
            for (let j = 0; j < n; j++) {
                const diff = A[i * n + j] - lnmean[i];
                varSum += diff * diff;
            }
            lnvar[i] = varSum / n;
            lnvarinv[i] = 1 / Math.sqrt(lnvar[i] + 1e-5);
        }

        // Normalize and apply gain and bias
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                const idx = i * n + j;
                lnraw[idx] = (A[idx] - lnmean[i]) * lnvarinv[i];
                lnout[idx] = gain[j] * lnraw[idx] + bias[j];
            }
        }

        return [
            lnout,
            (grad) => {
                const dA = new FloatMatrix(A);
                const dGain = createFloatMatrix(gain.shape);
                const dBias = createFloatMatrix(bias.shape);
                const gradSum = createFloatMatrix([m]);
                const gradXnormSum = createFloatMatrix([m]);

                // Sum over last dim per row
                for (let i = 0; i < m; i++) {
                    for (let j = 0; j < n; j++) {
                        const idx = i * n + j;
                        gradSum[i] += grad[idx];
                        gradXnormSum[i] += grad[idx] * lnraw[idx];
                        dGain[j] += grad[idx] * lnraw[idx];
                        dBias[j] += grad[idx];
                    }
                }

                // Backprop layer norm
                for (let i = 0; i < m; i++) {
                    for (let j = 0; j < n; j++) {
                        const idx = i * n + j;
                        dA[idx] = gain[j] * lnvarinv[i] / n * (
                            n * grad[idx] - 
                            gradSum[i] - 
                            lnraw[idx] * gradXnormSum[i]
                        );
                    }
                }

                return [dA, dGain, dBias];
            },
        ];
    }
);

class LayerNorm {
    constructor( nEmbed ) {
        this.gain = new Value( createFloatMatrix( [ nEmbed ], () => 1 ) );
        this.bias = new Value( createFloatMatrix( [ nEmbed ], () => 0 ) );
    }
    apply( x ) {
        return x.layerNorm( this.gain, this.bias );
    }
    params() {
        return [ this.gain, this.bias ];
    }
}

class AttentionBlock {
    constructor( nEmbed, nHeads ) {
        const headSize = nEmbed / nHeads;
        this.head = new MultiHeadAttention( nEmbed, nHeads, headSize );
        this.feedForward = new FeedForward( nEmbed );
        this.layerNorm1 = new LayerNorm( nEmbed );
        this.layerNorm2 = new LayerNorm( nEmbed );
    }
    apply( x ) {
        // Residual connections.
        // (Note: this doubled the initial loss, but fixed by layerNorm.)
        x = x.add( this.head.apply( this.layerNorm1.apply( x ) ) ); // (B, T, C)
        x = x.add( this.feedForward.apply( this.layerNorm2.apply( x ) ) ); // (B, T, C)
        return x;
    }
    params() {
        return [
            ...this.head.params(),
            ...this.feedForward.params(),
            ...this.layerNorm1.params(),
            ...this.layerNorm2.params(),
        ];
    }
}

class AttentionModel {
    constructor( vocabSize, nEmbed, nHeads, nLayers ) {
        this.tokenEmbedding = new Embedding( vocabSize, nEmbed );
        this.positionEmbedding = new Embedding( blockSize, nEmbed );
        this.blocks = new Sequential(
            Array.from( { length: nLayers }, () => new AttentionBlock( nEmbed, nHeads ) )
        );
        this.layerNorm = new LayerNorm( nEmbed );
        this.llmHead = new LinearBroadcast( nEmbed, vocabSize );
    }
    apply( x ) {
        const tokenEmbedding = this.tokenEmbedding.apply( x ); // (B, T, C)
        const positionEmbedding = this.positionEmbedding.apply( Array.from( { length: blockSize }, ( _, i ) => i ) ); // (T, C)
        x = tokenEmbedding.add( positionEmbedding.expandAndTile( x.shape[0] ) ); // (B, T, C)
        x = this.blocks.apply( x ); // (B, T, C)
        x = this.layerNorm.apply( x ); // (B, T, C)
        const logits = this.llmHead.apply( x ); // (B, T, vocabSize)
        return logits;
    }
    params() {
        return [
            ...this.tokenEmbedding.params(),
            ...this.positionEmbedding.params(),
            ...this.blocks.params(),
            ...this.layerNorm.params(),
            ...this.llmHead.params(),
        ];
    }
}

// Hyperparameters.
const nEmbed = 384;
const nHeads = 6;
const nLayers = 6;

const model = new AttentionModel( vocabSize, nEmbed, nHeads, nLayers );

print(model.params().reduce((a, b) => a + b.data.length, 0), 'number of params');

const logits = model.apply( x );
await logits.forward();

console.log(logits);

print( logits.data );
</script>

<script>

const loss = logits
    .reshape( ( [ B, T, C ] ) => [ B * T, C ] )
    .softmaxCrossEntropy( new IntMatrix( y ).reshape( [ y.length ] ) );
await loss.forward();
print( loss.data );
</script>

<script>
async function generate( seed, length ) {
    let out = encode( seed.padStart( blockSize, '\n' ).slice( -blockSize ) );
    
    while ( out.length < length ) {
        const logits = model
            .apply( new IntMatrix( out.slice( -blockSize ) ).reshape( [ 1, blockSize ] ) )
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
const waterfallForward = document.createElement( 'div' );
const waterfallBackward = document.createElement( 'div' );
print(graph);
print( waterfallForward );
print( waterfallBackward );
function createWaterfallChart(element, data) {
    // Normalize times to start from 0
    const minStart = Math.min(...data.map(d => d.start));
    const normalizedData = data.map(d => ({
        ...d,
        start: d.start - minStart,
        end: d.end - minStart
    }));

    // Sort by start time
    normalizedData.sort((a, b) => a.start - b.start);

    // Extended color palette
    const colorPalette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ];

    const uniqueLabels = [...new Set(normalizedData.map(d => d.label))];
    const labelColors = Object.fromEntries(
        uniqueLabels.map((label, i) => [label, colorPalette[i % colorPalette.length]])
    );

    const traces = [];
    const rows = [];

    for (const task of normalizedData) {
        // Assign to row without time conflict
        let row = 0;
        while (rows[row]?.some(d => !(task.end <= d.start || task.start >= d.end))) {
            row++;
        }
        if (!rows[row]) rows[row] = [];
        rows[row].push(task);

        traces.push({
            type: "bar",
            orientation: "h",
            x: [task.end - task.start],
            y: [row],
            base: task.start,
            name: task.label,
            marker: { color: labelColors[task.label] },
            hoverinfo: 'name+x',
            showlegend: !traces.some(t => t.name === task.label)
        });
    }

    const maxTime = Math.max(...normalizedData.map(d => d.end));

    return Plotly.newPlot(element, traces, {
        title: "Time-based Waterfall Chart",
        barmode: "stack",
        xaxis: {
            title: "Time",
            range: [0, maxTime]
        },
        yaxis: {
            showticklabels: false,
            title: "",
            autorange: "reversed"
        },
        margin: { t: 40 }
    });
}

for ( let i = 0; i < 1; i++ ) {
    const start = performance.now();
    const [ x, y ] = getBatch( 'train' );
    console.log( performance.now() - start, 'ms getBatch' );
    const startModel = performance.now();
    const logits = model.apply( x );
    const loss = logits
        .reshape( ( [ B, T, C ] ) => [ B * T, C ] )
        .softmaxCrossEntropy( new IntMatrix( y ).reshape( [ y.length ] ) );
    console.log( performance.now() - startModel, 'ms model' );

    window.bufferTimes = [];
    window.forwardTimes = [];
    const startForward = performance.now();
    await loss.forward();
    console.log( performance.now() - startForward, 'ms forward' );
    console.log( window.bufferTimes.reduce((a, b) => a + b, 0), 'ms buffer' );
    batchLosses.push( loss.data );
    console.log( loss.data );
    createWaterfallChart( waterfallForward, window.forwardTimes );

    window.bufferTimes = [];
    window.backwardTimes = [];
    const startBackward = performance.now();
    await loss.backward();
    console.log( performance.now() - startBackward, 'ms backward' );
    console.log( window.bufferTimes.reduce((a, b) => a + b, 0), 'ms buffer' );
    createWaterfallChart( waterfallBackward, window.backwardTimes );

    const startUpdate = performance.now();
    for ( const param of model.params() ) {
        for ( let i = param.data.length; i--; ) {
            param.data[ i ] -= 0.001 * param.grad[ i ];
        }
    }
    console.log( performance.now() - startUpdate, 'ms update' );
    await createLossesGraph( graph, batchLosses, losses );
    console.log( performance.now() - start, 'ms total' );
}
</script>
