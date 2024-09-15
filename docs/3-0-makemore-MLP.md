---
layout: default
title: '3. makemore: MLP'
permalink: '/makemore-MLP'
---

We will reuse the following functions from previous chapters.

<script>
const { GPU } = await import( new URL( './matmul-gpu.js', location ) );
const { matMul } = await GPU();
const {
    random,
    oneHot,
    transpose,
    softmaxByRow,
    negativeLogLikelihood,
    softmaxCrossEntropyGradient,
    sample
} = await import( new URL( './1-bigram-utils.js', location ) );
</script>

<script data-src="utils.js">
const matrixMixin = (Base) => class extends Base {
    constructor(data, shape = data?.shape || []) {
        const length = shape.reduce((a, b) => a * b, 1);

        // Call the parent class constructor
        super(data || length);

        if (this.length !== length) {
            throw new Error('Shape does not match data length.');
        }

        this.shape = shape;
    }
};
export class FloatMatrix extends matrixMixin(Float32Array) {}
export class IntMatrix extends matrixMixin(Int32Array) {}
</script>

In the first chapter, we created a bigram model, but it didn't produce very
name-like sequences. The problem is that it was only looking at pairs of
characters, and didn't consider characters further back. The problem with the
bigram model is that the table will grow exponentially for each character of
added context. For example, to look at trigrams (3 characters), we would need
a table that is 27x27x27 = 19683 entries. With 4 characters (4-grams) the table
would grow to 27^4 = 531441 entries.

Let's implement [A Neural Probabilistic Language
Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), Bengio et al. 2003.

Let's again fetch the names as we did in the first chapter.

<script>
const response = await fetch('https://raw.githubusercontent.com/karpathy/makemore/master/names.txt');
const text = await response.text();
const names = text.split('\n');
</script>

And we again make the index-to-character and character-to-index mappings.

<script>
const indexToCharMap = [ '.', ...new Set( names.join('') ) ].sort();
const stringToCharMap = {};

for ( let i = indexToCharMap.length; i--; ) {
    stringToCharMap[ indexToCharMap[ i ] ] = i;
}
</script>

Now we build the dataset. But unlike last time, we want to dynamically build the
dataset based on the context length. We'll call this the block size. It's a
hyper parameter we can tune to experiment with later to try to get a better
result.

<script data-src="utils.js">
export function buildDataSet( names, stringToCharMap, blockSize ) {
    let X = [];
    let Y = [];

    for ( const name of names ) {
        const context = '.'.repeat( blockSize ) + name + '.';
        let i = blockSize;
        while ( context[ i ] ) {
            const x = context.slice( i - blockSize, i );
            const y = context[ i ];
            X.push( ...[ ...x ].map( ( char ) => stringToCharMap[ char ] ) );
            Y.push( stringToCharMap[ y ] );
            i++;
        }
    }

    return [
        new IntMatrix( X, [ X.length / blockSize, blockSize ] ),
        new IntMatrix( Y, [ Y.length ] )
    ];
}
</script>

<script>
const hyperParameters = { blockSize: 3 };
const [ X, Y ] = buildDataSet( names, stringToCharMap, hyperParameters.blockSize );
</script>

Instead of x (the inputs) being the same shape as y (the targets or labels), x
is now y.length x blockSize matrix.

We now want to create an embedding matrix. Each character can be embedded in
2D space. We'll randomly initialise this, it will be trained. Not that the
embedding dimensions can be larger than 2, it just makes it easier to visualise
the 2D space later. Again this is a hyper parameter we can tune.

<script>
hyperParameters.embeddingDimensions = 2;
const totalChars = indexToCharMap.length;
const CData = random( [ totalChars, hyperParameters.embeddingDimensions ] );
</script>

How to we grab the embedding for a character? One way to grab the embedding for
a character is to use the character's index.

<script>
const indexOfB = stringToCharMap[ 'b' ];
const embeddingForB = [
    CData[ indexOfB * hyperParameters.embeddingDimensions + 0 ],
    CData[ indexOfB * hyperParameters.embeddingDimensions + 1 ],
];
</script>

As we saw last time, this can also be accomplished by one-hot encoding the
character and then multiplying it by the embedding matrix.

<script>
const oneHotForB = oneHot( [ indexOfB ], totalChars );
const embeddingForB = await matMul( oneHotForB, CData );
</script>

However, the first method is more efficient. Let's write a utility function.

<script data-src="utils.js">
export function gather(A, indices) {
    const shape = indices.shape ?? [ indices.length ];
    if (A.shape.length !== 2) {
        const R = new FloatMatrix( null, shape );
        for (let i = indices.length; i--;) {
            R[i] = A[indices[i]];
        }
        return R;
    }
    const Dim = A.shape[1];
    const R = new FloatMatrix( null, [...shape, Dim] );
    for (let i = indices.length; i--;) {
        const index = indices[i];
        for (let j = Dim; j--;) {
            R[i * Dim + j] = A[index * Dim + j];
        }
    }
    return R;
}
</script>

<script>
const embeddingForB = gather( CData, new Int32Array( [ indexOfB ] ) );
</script>

Now we can easily grab the embeddings for each context character in the input.

<script >
const CX = gather( CData, X );
</script>

Now we'll initialize the weights and biases for the MLP.

<script>
hyperParameters.neurons = 100;
const { embeddingDimensions, blockSize, neurons } = hyperParameters;
const W1Data = random( [ embeddingDimensions * blockSize, neurons ] );
const b1Data = random( [ neurons ] );
</script>

But how can we multiply these matrices together? We must re-shape (essentially
flatten) the CX matrix so that the embeddings for each character in the block
size forms a single row.

<script>
const { embeddingDimensions, blockSize } = hyperParameters;
const CXReshaped = new FloatMatrix( CX, [ X.shape[ 0 ], embeddingDimensions * blockSize ] );
</script>

Now we can multiply the matrices, add the biases, and apply the element-wise
tanh activation function. This forms the hidden layer.

<script data-src="utils.js">
export async function matMulBias( A, B, bias ) {
    const data = await matMul(A, B);
    if ( ! bias ) return data;
    const [ m, n ] = data.shape;
    if (n !== bias.length ) {
        throw new Error('Bias vector dimension does not match the resulting matrix rows.');
    }
    // Add the biases to every row.
    for ( let m_ = m; m_--; ) {
        for ( let n_ = n; n_--; ) {
            data[ m_ * n + n_ ] += bias[ n_ ];
        }
    }
    return data;
}
</script>

<script>
const h = await matMulBias( CXReshaped, W1Data, b1Data );
// Activation function.
for ( let i = h.length; i--; ) h[ i ] = Math.tanh( h[ i ] );
</script>

Output layer.

<script>
const { neurons } = hyperParameters;
const W2Data = random( [ neurons, totalChars ] );
const b2Data = random( [ totalChars ] );
const logits = await matMul( h, W2Data );
const [ m, n ] = logits.shape;
// Add the biases to every row.
for ( let m_ = m; m_--; ) {
    for ( let n_ = n; n_--; ) {
        logits[ m_ * n + n_ ] += b2Data[ n_ ];
    }
}
</script>

Softmax. Talk about what softmax cross entropy is, how it's beneficial to
cluster for efficiency. As we saw in chapter 2, it's a much more simple backward
pass.

<script>
const probs = softmaxByRow( logits );
</script>

Every row of `probs` sums to ~1.

<script>
const row1 = new FloatMatrix( null, [ 1, totalChars ] );
for ( let i = totalChars; i--; ) {
    row1[ 0 * totalChars + i ] = probs[ 0 * totalChars + i ];
}
const sumOfRow1 = row1.reduce( ( a, b ) => a + b, 0 );
</script>

Calculate the loss, which we'd like to minimize.

<script>
const mean = negativeLogLikelihood( probs, Y );
</script>

Great, we now have the forward pass. Let's use the approach we saw in chapter 2
to automatically calculate the gradients.

There's a few important differences from chapter 2.

1. Instead of scalar values, we now have matrices.
2. The matMul operation on the GPU is asynchronous.

Other than that, the code is largely the same. We also saw in chapter 1 how to
calculate the gradients for the matMul operation and softmax cross entropy. The
difference here is that we add the bias in a single operation for performance.
We also saw in chapter 2 how to calculate the gradients for the tanh activation
function.

Explain the gather operation.

<script data-src="utils.js">
const { getTopologicalOrder } = await import( new URL( './2-autograd-utils.js', location ) );
const { GPU } = await import( new URL( './matmul-gpu.js', location ) );
const { matMul } = await GPU();
const {
    transpose,
    softmaxByRow,
    negativeLogLikelihood,
    softmaxCrossEntropyGradient,
} = await import( new URL( './1-bigram-utils.js', location ) );
export class Value {
    static operations = new Map();
    constructor(data, _children = [], _op) {
        this.data = data;
        this._op = _op;
        this._prev = _children;
    }
    static addOperation(name, forward, backward) {
        this.operations.set(name, { forward, backward });
        this.prototype[name] = function(...args) {
            return new Value( null, [ this, ...args ], name );
        }
    }
    async forward() {
        const order = getTopologicalOrder(this);

        for (const node of order) {
            if (node._op) {
                const { forward } = Value.operations.get(node._op);
                const args = node._prev;
                node.data = await forward(...args);
            }
        }
    }
    async backward() {
        const reversed = getTopologicalOrder(this).reverse();

        for (const node of reversed) {
            node.grad = null;
        }

        this.grad = new FloatMatrix( null, this.data.shape ).fill( 1 );

        for (const node of reversed) {
            if (node._op) {
                const { backward } = Value.operations.get(node._op);
                const args = node._prev;
                const backwards = backward(...args);
                for (let i = 0; i < args.length; i++) {
                    if (args[i] instanceof Value) {
                        const grad = await backwards[i](node);
                        // Accumulate the gradients!
                        args[i].grad = args[i].grad ? add( args[i].grad, grad ) : grad;
                    }
                }
            }
        }
    }
}

function add( A, B ) {
    if ( A.shape.toString() !== B.shape.toString() ) {
        throw new Error( 'Matrix dimensions do not match.' );
    }

    const C = new FloatMatrix( A );
    for ( let i = C.length; i--; ) C[ i ] += B[ i ];
    return C;
}

Value.addOperation( 'matMulBias', async ( A, B, bias ) => {
    return await matMulBias( A.data, B.data, bias.data );
}, ( A, B, bias ) => [
    async ( out ) => {
        return await matMul( out.grad, transpose( B.data ) )
    },
    async ( out ) => await matMul( transpose( A.data ), out.grad ),
    ( out ) => {
        const A = out.grad;
        const [ m, n ] = A.shape;
        const B = new FloatMatrix( null, [ n ] );
        // Gradients for the biases are the sum of the gradients for
        // each row.
        for ( let m_ = m; m_--; ) {
            for ( let n_ = n; n_--; ) {
                B[ n_ ] += A[ m_ * n + n_ ];
            }
        }
        return B;
    }
] );

Value.addOperation( 'tanh', ( A ) => {
    const data = new FloatMatrix( A.data );
    for ( let i = data.length; i--; ) data[ i ] = Math.tanh( data[ i ] );
    return data;
}, ( A ) => [
    ( out ) => {
        const B = new FloatMatrix( out.grad );
        const tanhA = out.data;
        for ( let i = B.length; i--; ) B[ i ] *= ( 1 - Math.pow( tanhA[ i ], 2 ) );
        return B;
    }
] );

Value.addOperation( 'gather', ( A, indices ) => {
    return gather( A.data, indices );
}, ( A, indices ) => [
    ( out ) => {
        const B = out.grad;
        const C = new FloatMatrix( null, A.data.shape );
        if ( A.data.shape.length !== 2 ) {
            for ( let i = B.length; i--; ) C[ indices[i] ] += B[i];
        } else {
            const Dim = A.data.shape[1];
            for ( let i = B.length; i--; ) {
                const index = indices[i];
                for ( let j = Dim; j--; ) {
                    C[ index * Dim + j ] += B[ i * Dim + j ];
                }
            }
        }

        return C;
    }
] );

Value.addOperation( 'softmaxCrossEntropy', ( A, indices ) => {
    const data = softmaxByRow( A.data );
    return negativeLogLikelihood( data, indices );
}, ( A, indices ) => [
    ( out ) => {
        const B = softmaxByRow( A.data );
        return softmaxCrossEntropyGradient( B, indices );
    }
] );

Value.addOperation( 'reshape', ( A, shape ) => {
    return new FloatMatrix( A.data, shape );
}, ( A, shape ) => [
    ( out ) => {
        return new FloatMatrix( out.grad, A.shape );
    }
] );
</script>

Now we can rebuild the mathematical operations we did before, and we should get
the same loss.

<script>
function logitFn( X ) {
    const { embeddingDimensions, blockSize } = hyperParameters;
    const { C, W1, b1, W2, b2 } = params;
    const embedding = C.gather( X ).reshape( [ X.shape[ 0 ], embeddingDimensions * blockSize ] );
    const hidden = embedding.matMulBias( W1, b1 ).tanh();
    return hidden.matMulBias( W2, b2 );
}
const C = new Value( CData );
const W1 = new Value( W1Data );
const b1 = new Value( b1Data );
const W2 = new Value( W2Data );
const b2 = new Value( b2Data );
const params = { C, W1, b1, W2, b2 };
const loss = logitFn( X ).softmaxCrossEntropy( Y );
await loss.forward();
</script>

Let's calculate the gradients.

<script>
hyperParameters.learningRate = 0.1;
const losses = [];
</script>

<script>
export { default as Plotly } from 'https://cdn.jsdelivr.net/npm/plotly.js-dist@2.26.2/+esm';
const graphs = [ document.createElement( 'div' ), document.createElement( 'div' ) ];
</script>

<script>
async function createLossesGraph( element, losses ) {
    await Plotly.react(
        element,
        [ { x: losses.map( ( _, i ) => i ), y: losses } ],
        {
            width: 500, height: 500,
            yaxis: { title: 'Loss', type: 'log' },
            xaxis: { title: 'Iterations' }
        },
        { displayModeBar: false }
    );
}
export async function createEmbeddingGraph( element, C ) {
    await Plotly.react(element, [
        {
            // get even indices from C.
            x: Array.from( C.data ).filter( ( _, i ) => i % 2 ),
            // get uneven indices from C.
            y: Array.from( C.data ).filter( ( _, i ) => ! ( i % 2 ) ),
            text: indexToCharMap,
            mode: 'markers+text',
            type: 'scatter',
            name: 'Embedding',
            marker: {
                size: 14,
                color: '#fff',
                line: { color: 'rgb(0,0,0)', width: 1 }
            }
        }
    ], {
        width: 500, height: 500,
        title: 'Embedding'
    });
}
</script>

<script>
const iterations = 5;
print(graphs);
for ( let i = 0; i < iterations; i++ ) {
    await loss.forward();
    losses.push( loss.data );
    await loss.backward();
    for ( const param of Object.values( params ) ) {
        for ( let i = param.data.length; i--; ) {
            param.data[ i ] -= hyperParameters.learningRate * param.grad[ i ];
        }
    }
    await createLossesGraph( graphs[0], losses );
    await createEmbeddingGraph( graphs[1], C );
}
</script>

This runs very slowly!

## Mini-batching

Instead of training on the entire dataset on every iteration, we can use a
subset of the dataset, called a mini-batch. This allows us to train on more data
in the same amount of time, speeding up training, and it can also help prevent
overfitting.

<script>
const batchLosses = [];
function resetParameters() {
    const { embeddingDimensions, blockSize, neurons } = hyperParameters;
    const { C, W1, b1, W2, b2 } = params;
    C.data = random( [ totalChars, embeddingDimensions ] );
    W1.data = random( [ embeddingDimensions * blockSize, neurons ] );
    b1.data = random( [ neurons ] );
    W2.data = random( [ neurons, totalChars ] );
    b2.data = random( [ totalChars ] );
    losses.length = 0;
    batchLosses.length = 0;
}
resetParameters();
hyperParameters.batchSize = 32;
</script>

It's much better to have an appropriate gradient and take more steps than it is
to have an exact gradient and take fewer steps.

Now it's important to note that the losses are for the mini-batch, not the
entire dataset. We coulde calculate the loss on the entire dataset, but not on
every iteration as this would slow us down. Instead we can calculate the loss
on the entire dataset once at the end.

<script data-src="utils.js">
export async function createLossesGraph( element, batchLosses, losses ) {
    Plotly.react(element, [
        {
            y: batchLosses,
            name: 'Batch losses',
        },
        {
            y: losses,
            x: Array.from( losses ).map( ( _, i ) => ( i + 1 ) * batchLosses.length / losses.length ),
            name: 'Training losses',
        },
    ], {
        title: 'Losses',
        width: 500,
        height: 500,
        yaxis: { title: 'Loss', type: 'log' },
        xaxis: { title: 'Iterations' }
    });
}
</script>

<script data-src="utils.js">
export function miniBatch( X, Y, batchSize ) {
    const indices = Int32Array.from( { length: batchSize }, () => Math.random() * X.shape[ 0 ] );
    return [ gather( X, indices ), gather( Y, indices ) ];
}
</script>

<script>
const iterations = 100;
print(graphs);
for ( let i = 0; i < iterations; i++ ) {
    const [ Xbatch, Ybatch ] = miniBatch( X, Y, hyperParameters.batchSize );
    const loss = logitFn( Xbatch ).softmaxCrossEntropy( Ybatch );
    await loss.forward();
    batchLosses.push( loss.data );
    await loss.backward();
    for ( const param of Object.values( params ) ) {
        for ( let i = param.data.length; i--; ) {
            param.data[ i ] -= hyperParameters.learningRate * param.grad[ i ];
        }
    }

    if ( batchLosses.length % 100 === 0 ) {
        const loss = logitFn( X ).softmaxCrossEntropy( Y );
        await loss.forward();
        losses.push( loss.data );
    }

    await createLossesGraph( graphs[0], batchLosses, losses );
    await createEmbeddingGraph( graphs[1], C );
}
</script>

If you run this 200-300 times, you'll see that the embedding clusters together
the similar characters, such as the vowels, and the `.` will distance itself
from all other characters.

## Learning rate decay

Once it starts to plateau, we can reduce the learning rate an order of
magnitude.

<script>
hyperParameters.learningRate = 0.01;
</script>

Go back to the iterations and run again.

## Splitting the dataset

As the capacity of the models increases (with more neurons, layers, etc), it
becomes more prone to overfitting. One way to combat this is to split the data
into training, validation, and test sets. The loss can get close to zero, but
all the the model is doing is memorizing the training data. We need to evaluate
against a validation set to see how well the model is performing.

Practically this means that, when sampling, we'll only get names that exist in
the dataset. We won't get any new sequences. The loss on names that are withheld
from the training set can be really high.

The standard split is 80% for training, 10% for validation (dev), and 10% for testing.

<script data-src="utils.js">
export function shuffle( array ) {
  let i = array.length;
  while (i--) {
    const randomIndex = Math.floor(Math.random() * i);
    [array[i], array[randomIndex]] = [array[randomIndex], array[i]];
  }
}
</script>

<script>
shuffle( names );
const n1 = Math.floor( names.length * 0.8 );
const n2 = Math.floor( names.length * 0.9 );
const { blockSize } = hyperParameters;
const [ Xtr, Ytr ] = buildDataSet( names.slice( 0, n1 ), stringToCharMap, blockSize );
const [ Xdev, Ydev ] = buildDataSet( names.slice( n1, n2 ), stringToCharMap, blockSize );
const [ Xte, Yte ] = buildDataSet( names.slice( n2 ), stringToCharMap, blockSize );
</script>

<script>
resetParameters();
hyperParameters.learningRate = 0.1;
</script>

<script>
const iterations = 100;
print(graphs);
for ( let i = 0; i < iterations; i++ ) {
    const [ Xbatch, Ybatch ] = miniBatch( Xtr, Ytr, hyperParameters.batchSize );
    const loss = logitFn( Xbatch ).softmaxCrossEntropy( Ybatch );
    await loss.forward();
    batchLosses.push( loss.data );
    await loss.backward();
    for ( const param of Object.values( params ) ) {
        for ( let i = param.data.length; i--; ) {
            param.data[ i ] -= hyperParameters.learningRate * param.grad[ i ];
        }
    }

    if ( batchLosses.length % 100 === 0 ) {
        const loss = logitFn( Xdev ).softmaxCrossEntropy( Ydev );
        await loss.forward();
        losses.push( loss.data );
    }

    await createLossesGraph( graphs[0], batchLosses, losses );
    await createEmbeddingGraph( graphs[1], C );
}
</script>

## Increasing the embedding size

The bottleneck seems to be the embedding size.

Right now every character is put on a 2d plane. Let's try a 3d embedding.

<script>
hyperParameters.embeddingDimensions = 3;
resetParameters();
</script>

<script>
async function create3DEmbeddingGraph( element, C ) {
    await Plotly.react(element, [
        {
            x: Array.from( C.data ).filter( ( _, i ) => i % 3 === 0 ),
            y: Array.from( C.data ).filter( ( _, i ) => i % 3 === 1 ),
            z: Array.from( C.data ).filter( ( _, i ) => i % 3 === 2 ),
            text: indexToCharMap,
            mode: 'markers+text',
            type: 'scatter3d',
            name: 'Embedding',
            marker: { size: 5, color: '#000' }
        }
    ], {
        width: 500,
        height: 500,
        title: 'Embedding'
    });
}
</script>

<script>
const iterations = 100;
print(graphs);
for ( let i = 0; i < iterations; i++ ) {
    const [ Xbatch, Ybatch ] = miniBatch( Xtr, Ytr, hyperParameters.batchSize );
    const loss = logitFn( Xbatch ).softmaxCrossEntropy( Ybatch );
    await loss.forward();
    batchLosses.push( loss.data );
    await loss.backward();
    for ( const param of Object.values( params ) ) {
        for ( let i = param.data.length; i--; ) {
            param.data[ i ] -= hyperParameters.learningRate * param.grad[ i ];
        }
    }

    if ( batchLosses.length % 100 === 0 ) {
        const loss = logitFn( Xdev ).softmaxCrossEntropy( Ydev );
        await loss.forward();
        losses.push( loss.data );
    }

    await createLossesGraph( graphs[0], batchLosses, losses );
    await create3DEmbeddingGraph( graphs[1], C );
}
</script>

## Excercise: try different hyperparameters

* even higher embedding dimensions
* more or less neurons
* batch size
* higher context length
* Play with learning rate decay.

<script>
print(hyperParameters);
</script>

## Sample names

<script>
export const names = [];
const { blockSize } = hyperParameters;

for (let i = 0; i < 5; i++) {
    let out = Array( blockSize ).fill( 0 );

    do {
        const context = new FloatMatrix( out.slice( -blockSize ), [ 1, blockSize ] );
        const logits = logitFn( context );
        await logits.forward();
        const probs = softmaxByRow( logits.data );
        const ix = sample( probs );
        out.push( ix );
    } while ( out[ out.length - 1 ] !== 0 );

    names.push( out.slice( blockSize, -1 ).map( ( i ) => indexToCharMap[ i ] ).join( '' ) );
}
</script>
