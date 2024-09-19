---
layout: default
title: '3.3. makemore: Batch Norm'
permalink: '/makemore-batch-norm'
---

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
const { random, softmaxByRow, matMul } = await import( new URL( './1-bigram-utils.js', location ) );
const {
    Value,
    FloatMatrix,
    IntMatrix,
    buildDataSet,
    miniBatch,
    shuffle,
    createLossesGraph
} = await import( new URL( './3-0-makemore-MLP-utils.js', location ) );
export { default as Plotly } from 'https://cdn.jsdelivr.net/npm/plotly.js-dist@2.26.2/+esm';
</script>

<script>

let bnmean;
let bnvar;
let bnvarinv;
let bnraw;

Value.addOperation('batchNorm', (A, gain, bias) => {
    A = A.data;
    const [m, n] = A.shape;
    bnraw = new FloatMatrix(A);
    bnmean = new FloatMatrix(null, [m]);
    bnvar = new FloatMatrix(null, [m]);
    bnvarinv = new FloatMatrix(null, [m]);

    for (let m_ = m; m_--;) {
        let sum = 0;
        for (let n_ = n; n_--;) {
            sum += A[m_ * n + n_];
        }
        const mean = sum / n;

        let variance = 0;
        for (let n_ = n; n_--;) {
            variance += (A[m_ * n + n_] - mean) ** 2;
        }
        variance /= n; // -1 for Bessel's correction?

        const varinv = (variance + 1e-5) ** -0.5;

        for (let n_ = n; n_--;) {
            bnraw[m_ * n + n_] = (A[m_ * n + n_] - mean) * varinv;
        }

        bnmean[m_] = mean;
        bnvar[m_] = variance;
        bnvarinv[m_] = varinv;
    }

    gain = gain.data;
    bias = bias.data;

    const bnout = new FloatMatrix(bnraw);

    for (let m_ = m; m_--;) {
        for (let n_ = n; n_--;) {
            const i = m_ * n + n_;
            bnout[i] = gain[m_] * bnraw[i] + bias[m_];
        }
    }

    return bnout;
}, (A, gain, bias) => [
    (out) => {
        // bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum (0) - n/ (n-1)*bnraw*(dhpreact*bnraw).sum(0))
        const A_data = A.data;
        const gain_data = gain.data;
        const outGrad = out.grad;
        const [m, n] = A_data.shape;
        const dA = new FloatMatrix(A_data);
        const outGradSum = new FloatMatrix(null, [m]);
        const outGradXbnrawSum = new FloatMatrix(null, [m]);

        for (let m_ = m; m_--;) {
            for (let n_ = n; n_--;) {
                outGradSum[m_] += outGrad[m_ * n + n_];
                outGradXbnrawSum[m_] += outGrad[m_ * n + n_] * bnraw[m_ * n + n_];
            }
        }

        for (let m_ = m; m_--;) {
            for (let n_ = n; n_--;) {
                const i = m_ * n + n_;
                dA[i] = gain[m_] * bnvarinv[m_] / n * (n * outGrad[i] - outGradSum[m_] - n / (n - 1) * bnraw[i] * outGradXbnrawSum[m_]);
            }
        }

        return dA;
    },
    (out) => {
        const A_data = A.data;
        const dGain = new FloatMatrix(gain.data);
        const outGrad = out.grad;
        const [ m, n ] = outGrad.shape;

        // Sum along the 0th dimension.
        for (let m_ = m; m_--;) {
            for (let n_ = n; n_--;) {
                dGain[m_] += outGrad[m_ * n + n_] * A_data[m_ * n + n_];
            }
        }

        return dGain;
    },
    (out) => {
        const dBias = new FloatMatrix(bias.data);
        const outGrad = out.grad;
        const [ m, n ] = outGrad.shape;

        // Sum along the 0th dimension.
        for (let m_ = m; m_--;) {
            for (let n_ = n; n_--;) {
                dBias[m_] += out[m_ * n + n_];
            }
        }

        return dBias;
    }
]);

function createNetwork() {
    const { embeddingDimensions, blockSize, neurons } = hyperParameters;
    const C = new Value( new FloatMatrix( random, [ vocabSize, embeddingDimensions ] ) );
    const W1 = new Value( new FloatMatrix( () => random() * 0.2, [ embeddingDimensions * blockSize, neurons ] ) );
    const b1 = new Value( new FloatMatrix( null, [ neurons ] ) );
    const W2 = new Value( new FloatMatrix( () => random() * 0.01, [ neurons, vocabSize ] ) );
    const b2 = new Value( new FloatMatrix( null, [ vocabSize ] ) );
    const bngain = new Value( new FloatMatrix( () => 1, [ neurons ] ) );
    const bnbias = new Value( new FloatMatrix( null, [ neurons ] ) );
    function logitFn( X ) {
        const embedding = C.gather( X ).reshape( [ X.shape[ 0 ], embeddingDimensions * blockSize ] );
        const preactivation = embedding.matMulBias( W1, b1 );
        const hidden = preactivation.batchNorm( bngain, bnbias );
        const activation = hidden.tanh();
        return activation.matMulBias( W2, b2 );
    }
    logitFn.params = [ C, W1, b1, W2, b2, bngain, bnbias ];
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
    break;
}

