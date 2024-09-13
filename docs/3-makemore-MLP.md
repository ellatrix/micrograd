---
layout: default
title: '3. makemore: MLP'
permalink: '/makemore-MLP'
---

In the first chapter, we created a bigram model, but it didn't produce very
name-like sequences. The problem is that it was only looking at pairs of
characters, and didn't consider characters further back. The problem with the
bigram model is that the table will grow exponentially for each character of
added context. For example, to look at trigrams (3 characters), we would need
a table that is 27x27x27 = 19683 entries. With 4 characters (4-grams) the table
would grow to 27^4 = 531441 entries.

Let's again fetch the names as we did in the first chapter.

<script>
const { GPU } = await import( new URL( './matmul-gpu.js', location ) );
export const matMul = ( await GPU() )?.matMul || matMul;
export class FloatMatrix extends Float32Array {
    constructor( data, shape = data?.shape || [] ) {
        const length = shape.reduce( ( a, b ) => a * b, 1 );

        super( data || length );

        if ( this.length !== length ) {
            throw new Error( 'Shape does not match data length.' );
        }

        this.shape = shape;
    }
}
export function oneHot( a, length ) {
    const B = new FloatMatrix( null, [ a.length, length ] );
    for ( let i = a.length; i--; ) B[ i * length + a[ i ] ] = 1;
    return B;
}
export function softmaxByRow( A ) {
    const [m, n] = A.shape;
    const B = new FloatMatrix( null, A.shape );
    for ( let m_ = m; m_--; ) {
        let max = -Infinity;
        for ( let n_ = n; n_--; ) {
            const value = A[m_ * n + n_];
            if (value > max) max = value;
        }
        let sum = 0;
        for ( let n_ = n; n_--; ) {
            const i = m_ * n + n_;
            // Subtract the max to avoid overflow
            sum += B[i] = Math.exp(A[i] - max);
        }
        for ( let n_ = n; n_--; ) {
            B[m_ * n + n_] /= sum;
        }
    }
    return B;
}
export function getTopologicalOrder( node ) {
    const result = [];
    const visited = new Set();

    function visit( node ) {
        if ( visited.has( node ) || ! node._prev ) return;
        visited.add( node );
        for ( const child of node._prev ) visit( child );
        result.push( node );
    }

    visit( node );

    return result;
}
export function transpose( A ) {
    const [ m, n ] = A.shape;
    const B = new FloatMatrix( null, [ n, m ] );

    for ( let m_ = m; m_--; ) {
        for ( let n_ = n; n_--; ) {
            B[n_ * m + m_] = A[m_ * n + n_];
        }
    }

    return B;
}
</script>

<script>
const response = await fetch('https://raw.githubusercontent.com/karpathy/makemore/master/names.txt');
export const text = await response.text();
export const names = text.split('\n');
</script>

And we again make the index-to-character and character-to-index mappings.

<script>
export const indexToCharMap = [ '.', ...new Set( names.join('') ) ].sort();
export const stringToCharMap = {};

for ( let i = indexToCharMap.length; i--; ) {
    stringToCharMap[ indexToCharMap[ i ] ] = i;
}
</script>

Now we build the dataset. But unlike last time, we want to dynamically build the
dataset based on the context length. We'll call this the block size.

<script>
export function buildDataSet( names, blockSize ) {
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

    X = new Int32Array( X );
    Y = new Int32Array( Y );
    X.shape = [ X.length / blockSize, blockSize ];
    Y.shape = [ Y.length ];

    return [ X, Y ];
}
export const blockSize = 3;
const dataset = buildDataSet( names, blockSize );
export const X = dataset[ 0 ];
export const Y = dataset[ 1 ];
</script>

x = inputs
y = targets or labels

Instead of x being the same shape as y, x is now n x blockSize matrix.

<script>
export function random( shape ) {
    const m = new FloatMatrix( null, shape );
    for ( let i = m.length; i--; ) m[ i ] = Math.random() * 2 - 1;
    return m;
}
export const totalChars = indexToCharMap.length;
export const embeddingDimensions = 2;
export const CData = random( [ totalChars, embeddingDimensions ] );
</script>

How to we grab the embedding for a character?

One way to grab the embedding for a character is to use the character's index.

<script>
export const indexOfB = stringToCharMap[ 'b' ];
export const embeddingForB = [
    CData[ indexOfB * embeddingDimensions + 0 ],
    CData[ indexOfB * embeddingDimensions + 1 ],
];
</script>
    
As we saw last time, this can also be accomplished by one-hot encoding the
character and then multiplying it by the embedding matrix.

<script>
export const oneHotForB = oneHot( [ indexOfB ], totalChars );
export const embeddingForB = await matMul( oneHotForB, CData );
</script>

However, the first method is more efficient. Let's write a utility function.

<script>
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
export const embeddingForB = gather( CData, new Int32Array( [ indexOfB ] ) );
</script>

Now we can easily grab the embeddings for each context character in the input.

<script >
export const CX = gather( CData, X );
</script>

Now we'll initialize the weights and biases for the MLP.

<script>
export const neurons = 100;
export const W1Data = random( [ embeddingDimensions * blockSize, neurons ] );
export const b1Data = random( [ neurons ] );
</script>

But how can we multiply these matrices together? We must re-shape the CX matrix.

<script>
export const CXReshaped = new FloatMatrix( CX, [ X.shape[ 0 ], embeddingDimensions * blockSize ] );
</script>

Now we can multiply the matrices.

<script>
export const h = await matMul( CXReshaped, W1Data );
const [ m, n ] = h.shape;
// Add the biases to every row.
for ( let m_ = m; m_--; ) {
    for ( let n_ = n; n_--; ) {
        h[ m_ * n + n_ ] += b1Data[ n_ ];
    }
}
// Activation function.
for ( let i = h.length; i--; ) h[ i ] = Math.tanh( h[ i ] );
</script>

Output layer.

<script>
export const W2Data = random( [ neurons, totalChars ] );
export const b2Data = random( [ totalChars ] );
export const logits = await matMul( h, W2Data );
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
export const probs = softmaxByRow( logits );
</script>

Every row of `probs` sums to 1.

<script>
const row1 = new FloatMatrix( null, [ 1, totalChars ] );
for ( let i = totalChars; i--; ) {
    row1[ 0 * totalChars + i ] = probs[ 0 * totalChars + i ];
}
export const sumOfRow1 = row1.reduce( ( a, b ) => a + b, 0 );
</script>

Calculate the loss, which we'd like to minimize.

<script>
let sum = 0;
const [ m, n ] = probs.shape;
for ( let m_ = m; m_--; ) {
    // Sum the logProbs (log likelihoods) of the correct label.
    sum += Math.log( probs[ m_ * n + Y[ m_ ] ] );
}

// Divide by the number of rows (amount of labels).
export const mean = -sum / m;
</script>

Great, we now have the forward pass. Let's use the approach we saw in chapter 2
to automatically calculate the gradients.

<script>
function add( A, B ) {
    if ( A.shape.toString() !== B.shape.toString() ) {
        throw new Error( 'Matrix dimensions do not match.' );
    }

    const C = empty( A.shape );
    for ( let i = A.length; i--; ) C[ i ] = A[ i ] + B[ i ];
    return C;
}
export class Value {
    static operations = new Map();
    constructor(data, _children = []) {
        this.data = data;
        this._prev = _children;
    }
    static addOperation(name, forward, backward) {
        this.operations.set(name, { forward, backward });
        this.prototype[name] = function(...args) {
            args = [ this, ...args ];
            const forwardResult = forward(...args);
            const createResult = (data) => {
                const result = new Value(data, args);
                result._op = name;
                return result;
            };
            return forwardResult instanceof Promise ?
                forwardResult.then(createResult) :
                createResult(forwardResult);
        }
    }
    async backward() {
        const reversed = getTopologicalOrder(this).reverse();

        for (const node of reversed) {
            node.grad = 0;
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
                        args[i].grad = args[i].grad ? add( args[i].grad, grad ) : grad;
                    }
                }
            }
        }
    }
}

Value.addOperation( 'matMulBias', async ( A, B, bias ) => {
    const data = await matMul(A.data, B.data);
    if ( ! bias ) return data;
    const b = bias.data;
    const [ m, n ] = data.shape;
    if (n !== b.length ) {
        throw new Error('Bias vector dimension does not match the resulting matrix rows.');
    }
    // Add the biases to every row.
    for ( let m_ = m; m_--; ) {
        for ( let n_ = n; n_--; ) {
            data[ m_ * n + n_ ] += b[ n_ ];
        }
    }
    return data;
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
    const [ m, n ] = data.shape;
    let sum = 0;
    for ( let m_ = m; m_--; ) {
        sum += Math.log( data[ m_ * n + indices[ m_ ] ] );
    }
    return -sum / m;
}, ( A, indices ) => [
    ( out ) => {
        const B = softmaxByRow( A.data );
        const [m, n] = B.shape;

        for ( let m_ = m; m_--; ) {
            // Subtract 1 for the gradient of the correct label.
            B[ m_ * n + indices[ m_ ] ] -= 1;
            for ( let n_ = n; n_--; ) {
                // Divide by the number of rows.
                B[ m_ * n + n_ ] /= m;
            }
        }

        return B;
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
export const C = new Value( CData );
export const W1 = new Value( W1Data );
export const b1 = new Value( b1Data );
export const W2 = new Value( W2Data );
export const b2 = new Value( b2Data );
export const params = [ C, W1, b1, W2, b2 ];
console.log(C.gather( X ))
const embedding = C.gather( X ).reshape( [ X.shape[ 0 ], embeddingDimensions * blockSize ] );
const h = ( await embedding.matMulBias( W1, b1 ) ).tanh();
const logits = await h.matMulBias( W2, b2 );
export const loss = logits.softmaxCrossEntropy( Y );
</script>

Let's calculate the gradients.

<script>
export const learningRate = 0.1;
export const losses = [];
</script>

<script>
export { default as Plotly } from 'https://cdn.jsdelivr.net/npm/plotly.js-dist@2.26.2/+esm';
export const graphs = document.createElement('div');
graphs.append( document.createElement('div') );
graphs.append( document.createElement('div') );
graphs.style.display = 'flex';
</script>

<script>
export async function createLossesGraph( element, losses ) {
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
export const learningRate = 0.1;
export const params = [ C, W1, b1, W2, b2 ];
export function resetParams() {
    C.data = new FloatMatrix( CData );
    W1.data = new FloatMatrix( W1Data );
    b1.data = new FloatMatrix( b1Data );
    W2.data = new FloatMatrix( W2Data );
    b2.data = new FloatMatrix( b2Data );
}
export async function lossFn( X, Y ) {
    const embedding = C.gather( X ).reshape( [ X.shape[ 0 ], embeddingDimensions * blockSize ] );
    const h = ( await embedding.matMulBias( W1, b1 ) ).tanh();
    const logits = await h.matMulBias( W2, b2 );
    return logits.softmaxCrossEntropy( Y );
}
resetParams();
</script>

<script data-iterations="10">
const loss = await lossFn( X, Y );
losses.push( loss.data );
await loss.backward();
for ( const param of params ) {
    for ( let i = param.data.length; i--; ) {
        param.data[ i ] -= learningRate * param.grad[ i ];
    }
}
await createLossesGraph( graphs.firstChild, losses );
await createEmbeddingGraph( graphs.lastChild, C );
export default graphs;
</script>

If you run this 200-300 times, you'll see that the embedding clusters together
the similar characters, such as the vowels, and the `.` will distance itself
from all other characters.

## Mini-batching

Instead of training on the entire dataset on every iteration, we can use a
subset of the dataset, called a mini-batch. This allows us to train on more data
in the same amount of time, speeding up training, and it can also help prevent
overfitting.

<script>
export const batchLosses = [];
export const losses = [];
export const batchSize = 32;
resetParams();
</script>

It's much better to have an appropriate gradient and take more steps than it is
to have an exact gradient and take fewer steps.

Now it's important to note that the losses are for the mini-batch, not the
entire dataset. We coulde calculate the loss on the entire dataset, but not on
every iteration as this would slow us down. Instead we can calculate the loss
on the entire dataset once at the end.

<script>
export async function createLossesGraph( element ) {
    Plotly.react(element, [
        {
            y: batchLosses,
            name: 'Batch losses',
            hoverinfo: 'none'
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
        yaxis: {
            title: 'Loss',
            type: 'log'
        },
        xaxis: {
            title: 'Iterations'
        }
    });
}
</script>

<script data-iterations="100">
const indices = Int32Array.from( { length: batchSize }, () => Math.random() * X.shape[ 0 ] );
indices.shape = [ indices.length ];
const Xbatch = gather( X, indices );
const Ybatch = gather( Y, indices );
const loss = await lossFn( Xbatch, Ybatch );
batchLosses.push( loss.data );
await loss.backward();
for ( const param of params ) {
    for ( let i = param.data.length; i--; ) {
        console.log(param)
        param.data[ i ] -= learningRate * param.grad[ i ];
    }
}

if ( batchLosses.length % 100 === 0 ) {
    losses.push( (await lossFn( X, Y )).data );
}

await createLossesGraph( graphs.firstChild, losses );
await createEmbeddingGraph( graphs.lastChild, C );
export default graphs;
</script>

You might notice that the loss is kind of unstable after a few 100 iterations.
This is because we're using a learning rate that's too high.

## What is a good learning rate?

Let's try to find a good learning rate between 1 and 0.001.

<script>
function linspace(start, end, num) {
    const step = (end - start) / (num - 1);
    return Array.from({ length: num }, (_, i) => start + (step * i));
}
export const learningRateExponents = linspace( -3, 0, 1000 );
export const learningRates = learningRateExponents.map( ( exponent ) => Math.pow( 10, exponent ) );
export const graph = document.createElement( 'div' );
</script>

<script>
resetParams();
export const losses = [];
export const loss = await lossFn( X, Y );
</script>

<script data-iterations="1000">
const indices = Int32Array.from( { length: batchSize }, () => Math.random() * X.shape[ 0 ] );
indices.shape = [ indices.length ];
const Xbatch = gather( X, indices );
const Ybatch = gather( Y, indices );
const loss = await lossFn( Xbatch, Ybatch );
losses.push( loss.data );
await loss.backward();
for ( const param of params ) {
    for ( let i = param.data.length; i--; ) {
        param.data[ i ] -= learningRates[ losses.length ] * param.grad[ i ];
    }
}

await Plotly.react(graph, [{
    x: [...learningRateExponents],
    y: [...losses],
}], {
    title: 'Loss vs Learning Rate',
    xaxis: {
        title: 'Learning Rate Exponent',
    },
    yaxis: {
        title: 'Loss',

    },
    width: 500,
    height: 500
});
export default graph;
</script>

## Learning rate decay

Once it starts to plateau, we can reduce the learning rate an order of
magnitude.

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

<script>
function shuffle(array) {
  let currentIndex = array.length;

  // While there remain elements to shuffle...
  while (currentIndex != 0) {

    // Pick a remaining element...
    let randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;

    // And swap it with the current element.
    [array[currentIndex], array[randomIndex]] = [
      array[randomIndex], array[currentIndex]];
  }
}
shuffle( names );
const n1 = Math.floor( names.length * 0.8 );
const n2 = Math.floor( names.length * 0.9 );
const [ Xtr, Ytr ] = buildDataSet( names.slice( 0, n1 ), blockSize );
const [ Xdev, Ydev ] = buildDataSet( names.slice( n1, n2 ), blockSize );
const [ Xte, Yte ] = buildDataSet( names.slice( n2 ), blockSize );
export { Xtr, Ytr, Xdev, Ydev, Xte, Yte };
</script>

<script>
resetParams();
export const losses = [];
export const batchLosses = [];
</script>

<script data-iterations="100">
const indices = Int32Array.from( { length: batchSize }, () => Math.random() * Xtr.shape[ 0 ] );
indices.shape = [ indices.length ];
const Xbatch = gather( Xtr, indices );
const Ybatch = gather( Ytr, indices );
const loss = await lossFn( Xbatch, Ybatch );
batchLosses.push( loss.data );
await loss.backward();
for ( const param of params ) {
    for ( let i = param.data.length; i--; ) {
        param.data[ i ] -= learningRate * param.grad[ i ];
    }
}

if ( batchLosses.length % 100 === 0 ) {
    losses.push( (await lossFn( Xdev, Ydev )).data );
}

await createLossesGraph( graphs.firstChild, losses );
await createEmbeddingGraph( graphs.lastChild, C );
export default graphs;
</script>

## Increasing the size of the neural net

300 neurons. (It won't have much effect.)

## Increasing the batch size

To prevent jitter and improve the gradient.

## Increasing the embedding size

The bottleneck seems to be the embedding size.

Right now every character is put on a 2d plane. Let's try a 3d embedding.

## Excercise: try different hyperparameters

* even higher embedding dimensions
* more or less neurons
* batch size
* higher context length
* Play with learning rate decay.

## Sample names

<script>
export const names = [];

for (let i = 0; i < 5; i++) {
    let out = Array( blockSize ).fill( 0 );

    do {
        const context = new FloatMatrix( out.slice( -blockSize ), [ 1, blockSize ] );
        const logits = await logitFn( context );
        const probs = softmaxByRow( logits.data );
        const ix = sample( probs );
        out.push( ix );
    } while ( out[ out.length - 1 ] !== 0 );

    names.push( out.slice( blockSize, -1 ).map( ( i ) => indexToCharMap[ i ] ).join( '' ) );
}
</script>

## Better initialization (video 3)

We want logits to be uniformly 0. Weights should be sampled from N(0, 1/sqrt(n)). Need entropy for symmetry breaking.

### Tanh too saturated

Lot's of -1 and 1 preactivations. tanh backward is 1 - tanh^2 so it stops the backpropagation.
Vanishing gradients.

Kaiming He, 2020.

We need a slight gain because the tanh is a squashing function. 5/3.

## Batch normalization

https://arxiv.org/pdf/1502.03167

## Organization