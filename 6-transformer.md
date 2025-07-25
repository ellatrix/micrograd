---
layout: default
title: '6. Transformer'
permalink: '/transformer'
---

<aside>
    This covers the <a href="https://www.youtube.com/watch?v=kCc8FmEb1nY">Let's build GPT</a> video.
</aside>

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
// Hyperparameters.
const blockSize = 8;
const batchSize = 32;

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

Value.addOperation( 'attentionHead', async (
    k, // (B, T, C)
    q, // (B, T, C)
    v, // (B, T, C)
) => {
    const [ B, T, C ] = k.shape;
    const scale = C ** -0.5;
    const weiCache = [];
    const out = createFloatMatrix( [ B, T, C ] );
    const batchPromises = [];
    for ( let b_ = B; b_--; ) {
        const startTC = b_ * T * C;
        const endTC = startTC + T * C;
        const qBatch = q.subarray( startTC, endTC ).reshape( [ T, C ] );
        const kBatch = k.subarray( startTC, endTC ).reshape( [ T, C ] );
        const vBatch = v.subarray( startTC, endTC ).reshape( [ T, C ] );

        batchPromises.push(
            // (B, T, C) @ ( (B, T, C) -> (B, C, T) ) -> (B, T, T)
            matMul( qBatch, transpose( kBatch ) )
            .then( weiBatch => {
                // Clamp to -Infinity the upper right triangle.
                // const offset = b_ * T * T;
                for ( let t_ = T; t_--; ) {
                    const t_offset = t_ * T;
                    for ( let t2_ = T; t2_--; ) {
                        if ( t2_ > t_ ) {
                            weiBatch[t_offset + t2_] = -Infinity;
                        } else {
                            weiBatch[t_offset + t2_] *= scale;
                        }
                    }
                    softmax( weiBatch.subarray( t_offset, t_offset + T ) );
                }
                weiCache[b_] = weiBatch;
                return weiBatch;
            })
            // (B, T, T) @ (B, T, C) -> (B, T, C)
            .then( weiBatch => matMul( weiBatch, vBatch ) )
            .then( outBatch => {
                out.set( outBatch, b_ * T * C );
            })
        );
    }
    await Promise.all(batchPromises);
    return [
        out,
        async ( dout ) => {
            const dK = createFloatMatrix( [ B, T, C ] );
            const dQ = createFloatMatrix( [ B, T, C ] );
            const dV = createFloatMatrix( [ B, T, C ] );
            const batchPromises = [];

            for ( let b_ = B; b_--; ) {
                const startTC = b_ * T * C;
                const endTC = startTC + T * C;
                const qBatch = q.subarray( startTC, endTC ).reshape( [ T, C ] );
                const kBatch = k.subarray( startTC, endTC ).reshape( [ T, C ] );
                const vBatch = v.subarray( startTC, endTC ).reshape( [ T, C ] );
                const dOutBatch = dout.subarray(startTC, endTC).reshape([ T, C ]);
                const weiBatch = weiCache[b_];

                batchPromises.push(
                    matMul(dOutBatch, transpose(vBatch)) // (T, T)
                    .then(dWei => {
                        // Backprop through softmax
                        const gradAttn = createFloatMatrix([ T, T ]);
                        for (let t_ = T; t_--;) {
                            const attnRow = weiBatch.subarray(t_ * T, (t_ + 1) * T);
                            const dWeiRow = dWei.subarray(t_ * T, (t_ + 1) * T);
                            for (let t2_ = T; t2_--;) {
                                let sum = 0;
                                for (let t3_ = T; t3_--;) {
                                    const delta = t2_ === t3_ ? 1 : 0;
                                    sum += attnRow[t3_] * (delta - attnRow[t2_]) * dWeiRow[t3_];
                                }
                                gradAttn[t_ * T + t2_] = sum;
                            }
                        }
                        return gradAttn;
                    })
                    .then(gradAttn => Promise.all([
                        matMul(gradAttn, kBatch), // (T, C)
                        matMul(transpose(gradAttn), qBatch), // (T, C)
                        matMul(transpose(weiBatch), dOutBatch) // (T, C)
                    ])).then(([_dq, _dk, _dv]) => {
                        // Same length.
                        for (let i = _dq.length; i--;) {
                            _dq[i] *= scale;
                            _dk[i] *= scale;
                        }
                        dQ.set(_dq, startTC);
                        dK.set(_dk, startTC);
                        dV.set(_dv, startTC);
                    })
                );
            }
            await Promise.all(batchPromises);
            return [dK, dQ, dV];
        }
    ];
});

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
        return k.attentionHead( q, v );
    }
    params() {
        return [ ...this.K.params(), ...this.Q.params(), ...this.V.params() ];
    }
}

Value.addOperation('concatLastDim', async (...args) => {
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
});

Value.addOperation('add', async (
    a, // (B, T, C)
    b, // (B, T, C)
) => {
    if ( a.shape.toString() !== b.shape.toString() ) {
        throw new Error('Shape mismatch: a.shape=' + a.shape + ', b.shape=' + b.shape);
    }

    const out = new FloatMatrix(a);
    for (let i_ = out.length; i_--;) out[i_] += b[i_];
    return [ out, (dout) => [dout, dout] ];
});

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

Value.addOperation('expandAndTile', async (
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
});

Value.addOperation('relu', (A) => {
    const out = new FloatMatrix(A);

    for (let i = out.length; i--;) {
        if ( out[i] < 0 ) {
            out[i] = 0;
        }
    }

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
});

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

Value.addOperation('layerNorm', (A, gain, bias) => {
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
});

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
const nEmbed = 32;
const nHeads = 4;
const nLayers = 3;

const model = new AttentionModel( vocabSize, nEmbed, nHeads, nLayers );

print(model.params().reduce((a, b) => a + b.data.length, 0), 'number of params');

const logits = model.apply( x );
await logits.forward();

import { getTopologicalOrder } from './2-autograd-utils.js';
console.log( getTopologicalOrder( logits ) );

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
print(graph);
for ( let i = 0; i < 10; i++ ) {
    const start = performance.now();
    const [ x, y ] = getBatch( 'train' );
    const logits = model.apply( x );
    const loss = logits
        .reshape( ( [ B, T, C ] ) => [ B * T, C ] )
        .softmaxCrossEntropy( new IntMatrix( y ).reshape( [ y.length ] ) );
    await loss.forward();
    batchLosses.push( loss.data );
    console.log( loss.data );

    await loss.backward();
    for ( const param of model.params() ) {
        for ( let i = param.data.length; i--; ) {
            param.data[ i ] -= 0.001 * param.grad[ i ];
        }
    }
    await createLossesGraph( graph, batchLosses, losses );
    const end = performance.now();
    console.log( end - start, 'ms' );
}
</script>
