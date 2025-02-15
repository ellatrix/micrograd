
const { GPU } = await import( new URL( './matmul-gpu.js', location ) );
const { matMul } = await GPU();

const matrixMixin = (Base) => class extends Base {
    constructor(data, shape = data?.shape || []) {
        const length = shape.reduce((a, b) => a * b, 1);

        if  ( typeof data === 'function' ) {
            data = Array.from( { length }, data );
        }

        super(data || length);

        if (this.length !== length) {
            throw new Error('Shape does not match data length.');
        }

        this.shape = shape;
    }
};
export class FloatMatrix extends matrixMixin(Float32Array) {}
export class IntMatrix extends matrixMixin(Int32Array) {}

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

const { getTopologicalOrder } = await import( new URL( './2-autograd-utils.js', location ) );
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
    static addOperation(operation, forward) {
        this.operations.set(operation, forward);
        this.prototype[operation] = function(...args) {
            return new Value( null, [ this, ...args ], operation );
        }
    }
    async forward() {
        const order = getTopologicalOrder(this);

        for (const node of order) {
            if (node._op) {
                const forward = Value.operations.get(node._op);
                const args = node._prev;
                const [data, ...grads] = await forward(...args.map(arg => {
                    return arg instanceof Value ? arg.data : arg;
                }));
                node.data = data;
                node._backward = async () => {
                    for (const [i, gradCalc] of grads.entries()) {
                        const grad = await gradCalc(node.grad);
                        const child = args[i];
                        child.grad = child.grad ? add(child.grad, grad) : grad;
                    }
                };
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
            await node._backward?.();
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

Value.addOperation( 'matMulBias', async ( A, B, bias ) => [
    await matMulBias( A, B, bias ),
    async ( grad ) => await matMul( grad, transpose( B ) ),
    async ( grad ) => await matMul( transpose( A ), grad ),
    ( grad ) => {
        const [ m, n ] = grad.shape;
        const B = new FloatMatrix( null, [ n ] );
        // Gradients for the biases are the sum of the gradients for
        // each row.
        for ( let m_ = m; m_--; ) {
            for ( let n_ = n; n_--; ) {
                B[ n_ ] += grad[ m_ * n + n_ ];
            }
        }
        return B;
    }
] );

Value.addOperation( 'tanh', ( A ) => {
    const data = new FloatMatrix( A );
    for ( let i = data.length; i--; ) data[ i ] = Math.tanh( data[ i ] );
    return [
        data,
        ( grad ) => {
            const B = new FloatMatrix( grad );
            for ( let i = B.length; i--; ) B[ i ] *= ( 1 - Math.pow( data[ i ], 2 ) );
            return B;
        }
    ];
} );

Value.addOperation( 'gather', ( A, indices ) => [
    gather( A, indices ),
    ( grad ) => {
        const B = grad;
        const C = new FloatMatrix( null, A.shape );
        if ( A.shape.length !== 2 ) {
            for ( let i = B.length; i--; ) C[ indices[i] ] += B[i];
        } else {
            const Dim = A.shape[1];
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
    const data = softmaxByRow( A );
    return [
        negativeLogLikelihood( data, indices ),
        () => softmaxCrossEntropyGradient( data, indices )
    ];
} );

Value.addOperation( 'reshape', ( A, shape ) => [
    new FloatMatrix( A, shape ),
    ( grad ) => new FloatMatrix( grad, A.shape )
] );

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

export function miniBatch( X, Y, batchSize ) {
    const indices = Int32Array.from( { length: batchSize }, () => Math.random() * X.shape[ 0 ] );
    return [ gather( X, indices ), gather( Y, indices ) ];
}

export function shuffle( array ) {
  let i = array.length;
  while (i--) {
    const randomIndex = Math.floor(Math.random() * i);
    [array[i], array[randomIndex]] = [array[randomIndex], array[i]];
  }
}
