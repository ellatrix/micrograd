---
layout: default
title: '3.3. makemore: Batch Norm'
permalink: '/makemore-batch-norm'
---

<aside>
    This covers the <a href="https://www.youtube.com/watch?v=TCH_1BHY58I">Building makemore Part 3: Activations & Gradients, BatchNorm (40:40-1:03:07)</a> video.
</aside>

In the previous section we saw that it was a good idea to have preactivation
values roughly unit gaussian (mean 0, standard deviation 1) at initialisation.
The insight of [Batch Normalization](https://arxiv.org/pdf/1502.03167), Sergey
Ioffe et al, 2015, was to simply make the preactivation values unit gaussian.
This is possible because it's a perfectly differentiable operation.

Here is the formula for batch normalisation.

Mini batch mean:

<div class="math">
$$
\mu_B \leftarrow \frac{1}{m} \sum_{i=1}^m x_i
$$
</div>

Mini batch variance:

<div class="math">
$$
\sigma_B^2 \leftarrow \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2
$$
</div>

Normalise:

<div class="math">
$$
\hat{x}_i \leftarrow \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$
</div>

Scale and shift:

<div class="math">
$$
x_i \leftarrow \gamma \hat{x}_i + \beta
$$
</div>

Where $$\gamma$$ and $$\beta$$ are learnable parameters. $$\epsilon$$ is a small constant
to avoid division by zero.

Why is gamma and beta needed? Well, we want it to be unit gaussian at
initialisation, but we also want to allow the neural net to change it.

Let's implement it.

<script>
import { random, softmaxByRow, matMul } from './1-bigram-utils.js';
import {
    Value,
    FloatMatrix,
    createFloatMatrix,
    buildDataSet,
    miniBatch,
    shuffle,
    createLossesGraph
} from './3-0-makemore-MLP-utils.js';
export { default as Plotly } from 'https://cdn.jsdelivr.net/npm/plotly.js-dist@2.26.2/+esm';
</script>

<script data-src="utils.js">
import { Value, FloatMatrix } from './3-0-makemore-MLP-utils.js';

Value.addOperation('batchNorm', (A, gain, bias) => {
    const n = A.shape.at(-1);
    const restDims = A.shape.slice(0, -1);
    const m = restDims.reduce((a, b) => a * b, 1);
    const bnraw = new FloatMatrix(A);
    const bnmean = createFloatMatrix( [n] );
    const bnvar = createFloatMatrix( [n] );
    const bnvarinv = createFloatMatrix( [n] );

    for (let n_ = n; n_--;) {
        let sum = 0;
        for (let m_ = m; m_--;) {
            sum += A[m_ * n + n_];
        }
        bnmean[n_] = sum / m;
    }

    for (let n_ = n; n_--;) {
        let variance = 0;
        for (let m_ = m; m_--;) {
            variance += (A[m_ * n + n_] - bnmean[n_]) ** 2;
        }
        // Apply Bessel's correction here
        bnvar[n_] = variance / (m - 0);
        bnvarinv[n_] = 1 / Math.sqrt(bnvar[n_] + 1e-5);
    }

    for (let m_ = m; m_--;) {
        for (let n_ = n; n_--;) {
            const i = m_ * n + n_;
            bnraw[i] = (A[i] - bnmean[n_]) * bnvarinv[n_];
        }
    }

    const bnout = createFloatMatrix( A.shape );

    for (let m_ = m; m_--;) {
        for (let n_ = n; n_--;) {
            const i = m_ * n + n_;
            bnout[i] = gain[n_] * bnraw[i] + bias[n_];
        }
    }

    return [
        bnout,
        (grad) => {
            // bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))
            const dA = new FloatMatrix(A);
            const outGradSum = createFloatMatrix( [n] );
            const outGradXbnrawSum = createFloatMatrix( [n] );
    
            // Calculate sums along the batch dimension (m)
            for (let n_ = n; n_--;) {
                for (let m_ = m; m_--;) {
                    const i = m_ * n + n_;
                    outGradSum[n_] += grad[i];
                    outGradXbnrawSum[n_] += grad[i] * bnraw[i];
                }
            }
    
            // Calculate the gradient
            for (let m_ = m; m_--;) {
                for (let n_ = n; n_--;) {
                    const i = m_ * n + n_;
                    dA[i] = gain[n_] * bnvarinv[n_] / m * (
                        m * grad[i] - 
                        outGradSum[n_] - 
                        m / (m - 0) * bnraw[i] * outGradXbnrawSum[n_]
                    );
                }
            }
    
            return dA;
        },
        (grad) => {
            const dGain = createFloatMatrix( gain.shape );
    
            // Sum along the 0th dimension (batch dimension).
            for (let n_ = n; n_--;) {
                for (let m_ = m; m_--;) {
                    const i = m_ * n + n_;
                    dGain[n_] += grad[i] * bnraw[i];
                }
            }
    
            return dGain;
        },
        (grad) => {
            const dBias = createFloatMatrix( bias.shape );
    
            // Sum along the 0th dimension (batch dimension).
            for (let n_ = n; n_--;) {
                for (let m_ = m; m_--;) {
                    dBias[n_] += grad[m_ * n + n_];
                }
            }
    
            return dBias;
        }
    ];
});
</script>

<script>
function createNetwork() {
    const { embeddingDimensions, blockSize, neurons } = hyperParameters;
    const C = new Value( createFloatMatrix( [ vocabSize, embeddingDimensions ], random ) );
    const W1 = new Value( createFloatMatrix( [ embeddingDimensions * blockSize, neurons ], () => random() * 0.2 ) );
    // const b1 = new Value( createFloatMatrix( [ neurons ] ) );
    const W2 = new Value( createFloatMatrix( [ neurons, vocabSize ], () => random() * 0.01 ) );
    const b2 = new Value( createFloatMatrix( [ vocabSize ] ) );
    const bngain = new Value( createFloatMatrix( [ neurons ], () => 1 ) );
    const bnbias = new Value( createFloatMatrix( [ neurons ] ) );
    function logitFn( X ) {
        const embedding = C.gather( X ).reshape( [ X.shape[ 0 ], embeddingDimensions * blockSize ] );
        // Note: we should remove the bias here becaus with batchNorm it's no
        // longer doing anything.
        const preactivation = embedding.matMulBias( W1 );
        const hidden = preactivation.batchNorm( bngain, bnbias );
        const activation = hidden.tanh();
        return activation.matMulBias( W2, b2 );
    }
    logitFn.params = [ C, W1, /*b1,*/ W2, b2, bngain, bnbias ];
    return logitFn;
}

const response = await fetch('https://raw.githubusercontent.com/karpathy/makemore/master/names.txt');
const text = await response.text();
const names = text.split('\n');
const indexToCharMap = [ '.', ...new Set( names.join('') ) ].sort();
const stringToCharMap = {};

for ( let i = indexToCharMap.length; i--; ) {
    stringToCharMap[ indexToCharMap[ i ] ] = i;
}

const hyperParameters = {
    embeddingDimensions: 10,
    blockSize: 3,
    neurons: 200,
    batchSize: 32,
    learningRate: 0.1,
};

shuffle( names );
const n1 = Math.floor( names.length * 0.8 );
const n2 = Math.floor( names.length * 0.9 );
const [ Xtr, Ytr ] = buildDataSet( names.slice( 0, n1 ), stringToCharMap, hyperParameters.blockSize );
const [ Xdev, Ydev ] = buildDataSet( names.slice( n1, n2 ), stringToCharMap, hyperParameters.blockSize );
const [ Xte, Yte ] = buildDataSet( names.slice( n2 ), stringToCharMap, hyperParameters.blockSize );
const vocabSize = indexToCharMap.length;
</script>

<script>
const batchLosses = [];
const losses = [];
const network = createNetwork();
</script>

<script>
const graph = document.createElement( 'div' );
print(graph);
for ( let i = 0; i < 1000; i++ ) {
    const [ Xbatch, Ybatch ] = miniBatch( Xtr, Ytr, hyperParameters.batchSize );
    const loss = network( Xbatch ).softmaxCrossEntropy( Ybatch );
    await loss.forward();
    batchLosses.push( loss.data );
    await loss.backward();
    const learningRate = batchLosses.length < 2000 ? 0.1 : 0.01;
    for ( const param of network.params ) {
        for ( let i = param.data.length; i--; ) {
            param.data[ i ] -= learningRate * param.grad[ i ];
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

We don't expect much improvement because it's a very simple network. But once
the network becomes deeper with different operation, it will become very
difficult to manually tune it so that all the activation are roughly gaussian.
It also has a regularising effect.

This comes at a terrible cost. The examples in the batch are coupled.

Now we need to find a way to sample from the network. The neural net expects
batches as an input now. We need to keep track of the batch mean and standard
deviaton over time.

Either we take the mean and standard deviation of the entire dataset, or we keep
a running mean and standard deviation. If we do the former, we need to refactor
the batchNorm operation to return the mean and standard deviation (or take it as
an input and calculate it outside) so that we can update the running mean and
standard deviation on every batch iteration.

To do that, let's first organise our layers better.

Note: biases in preactivation are now no longer doing anything, so we should
removed them to not waste compute. The batch norm bias is now in charge of
biasing.

Normally we place batch norm after layers that have multiplication.

Group normalisation and others have become more common because they avoid the
coupling of examples, which is less bug prone. It's better to avoid batch norm
if we can.
