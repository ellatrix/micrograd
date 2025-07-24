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
import { Embedding, Linear, Sequential, Tanh } from './3-4-layer-organisation-utils.js'; 
import { LinearBroadcast } from './5-wavenet-utils.js';

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
            for ( let t2_ = T; t2_--; ) {
                if ( t2_ > t_ ) {
                    wei[t_offset + t2_] = -Infinity;
                } else {
                    wei[t_offset + t2_] *= scale;
                }
            }
            softmax( wei.subarray( t_offset, t_offset + T ) );
        }
        const weiBatch = wei.subarray( b_ * T * T, (b_ + 1) * T * T ).reshape( [ T, T ] );
        const vBatch = v.subarray( b_ * T * C, (b_ + 1) * T * C ).reshape( [ T, C ] );
        // (B, T, T) @ (B, T, C) -> (B, T, C)
        out.set( await matMul( weiBatch, vBatch ), b_ * T * C );
    }
    return [
        out,
        async ( dout ) => {
            const dK = createFloatMatrix( [ B, T, C ] );
            const dQ = createFloatMatrix( [ B, T, C ] );
            const dV = createFloatMatrix( [ B, T, C ] );

            for ( let b_ = B; b_--; ) {
                const startTC = b_ * T * C;
                const startTT = b_ * T * T; 
                const qBatch = q.subarray( startTC, startTC + T * C ).reshape( [ T, C ] );
                const kBatch = k.subarray( startTC, startTC + T * C ).reshape( [ T, C ] );
                const vBatch = v.subarray( startTC, startTC + T * C ).reshape( [ T, C ] );
                const weiBatch = wei.subarray( startTT, startTT + T * T ).reshape( [ T, T ] );
                const dOutBatch = dout.subarray(startTC, startTC + T * C).reshape([ T, C ]);
                const dWei = await matMul(dOutBatch, transpose(vBatch)); // (T, T)
                dV.set( await matMul(transpose(weiBatch), dOutBatch), startTC ); // (T, C)

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

                const _dq = await matMul(gradAttn, kBatch); // (T, C)
                const _dk = await matMul(transpose(gradAttn), qBatch); // (T, C)

                // Same length.
                for (let i = _dq.length; i--;) {
                    _dq[i] *= scale;
                    _dk[i] *= scale;
                }

                dQ.set(_dq, startTC);
                dK.set(_dk, startTC);
            }

            return [dK, dQ, dV];
        }
    ];
});

const nEmbed = 32;
const nHeads = 4;
const headSize = nEmbed / nHeads;

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

class MultiHeadAttention {
    constructor( nEmbed, nHeads, headSize ) {
        this.heads = Array.from( { length: nHeads }, () => new Head( nEmbed, headSize ) );
    }
    apply( x ) {
        const heads = this.heads.map( head => head.apply( x ) );
        return heads[0].concatLastDim( ...heads.slice(1) );
    }
    params() {
        return this.heads.flatMap( head => head.params() );
    }
}

Value.addOperation('add', async (
    a, // (B, T, C)
    b, // (T, C)
) => {
    const out = createFloatMatrix(a.shape);
    const [ B, T, C ] = a.shape;

    // Forward pass: out = a + b (broadcast b over B)
    for (let b_ = B; b_--;) {
        const offset = b_ * T * C;
        for (let t = 0; t < T; t++) {
            const rowOffset = offset + t * C;
            const bRowOffset = t * C;
            for (let c = 0; c < C; c++) {
                out[rowOffset + c] = a[rowOffset + c] + b[bRowOffset + c];
            }
        }
    }

    return [
        out,
        async (dout) => {
            const dA = dout; // Gradient passes through directly to a
            const dB = createFloatMatrix(b.shape); // (T, C)

            // Sum dout over batch dimension for dB
            for (let b_ = 0; b_ < B; b_++) {
                const offset = b_ * T * C;
                for (let t = 0; t < T; t++) {
                    const rowOffset = offset + t * C;
                    const bRowOffset = t * C;
                    for (let c = 0; c < C; c++) {
                        dB[bRowOffset + c] += dout[rowOffset + c];
                    }
                }
            }

            return [dA, dB];
        }
    ];
});

class FeedForward {
    constructor( nEmbed ) {
        this.net = new Sequential([
            new LinearBroadcast( nEmbed, nEmbed ),
            new Tanh(),
        ]);
    }
    apply( x ) {
        return this.net.apply( x );
    }
    params() {
        return this.net.params();
    }
}

class AttentionModel {
    constructor( vocabSize, nEmbed, headSize ) {
        this.tokenEmbedding = new Embedding( vocabSize, nEmbed );
        this.positionEmbedding = new Embedding( blockSize, nEmbed );
        this.head = new MultiHeadAttention( nEmbed, nHeads, headSize );
        this.feedForward = new FeedForward( nEmbed );
        this.llmHead = new LinearBroadcast( nEmbed, vocabSize );
    }
    apply( x ) {
        const tokenEmbedding = this.tokenEmbedding.apply( x ); // (B, T, C)
        const positionEmbedding = this.positionEmbedding.apply( Array.from( { length: blockSize }, ( _, i ) => i ) ); // (T, C)
        console.log({positionEmbedding});
        x = tokenEmbedding.add( positionEmbedding ); // (B, T, C)
        x = this.head.apply( x ); // (B, T, C)
        x = this.feedForward.apply( x ); // (B, T, C)
        const logits = this.llmHead.apply( x ); // (B, T, vocabSize)
        console.log( logits );
        return logits;
    }
}

const model = new AttentionModel( vocabSize, nEmbed, headSize );

const logits = model.apply( x );
await logits.forward();
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
    let out = encode( seed );
    
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
for ( let i = 0; i < 2; i++ ) {
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
}
</script>
