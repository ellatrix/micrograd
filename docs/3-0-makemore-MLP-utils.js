
import { GPU } from './matmul-gpu.js';
export const { matMul, scatterAdd, batchMatMul, batchSoftmaxRowTril } = await GPU();

const matrixMixin = (Base) => class extends Base {
    #shape = new Int32Array();
    constructor(data, ...args) {
        super(data, ...args);
        this.shape = data?.shape ?? [ this.length ];
    }
    get shape() {
        return Array.from( this.#shape );
    }
    set shape( shape ) {
        if ( typeof shape === 'function' ) shape = shape( this.shape );
        if (this.length !== shape.reduce((a, b) => a * b, 1))
            throw new Error('Shape does not match data length.');
        this.#shape = new Int32Array( shape );
    }
    reshape( shape ) {
        this.shape = shape;
        return this;
    }
};
export class FloatMatrix extends matrixMixin(Float32Array) {}
export class IntMatrix extends matrixMixin(Int32Array) {}

export function createFloatMatrix( shape, fn ) {
    const length = shape.reduce((a, b) => a * b, 1);
    return new FloatMatrix( fn ? Array.from( { length }, fn ) : length ).reshape( shape );
}

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
        new IntMatrix( X ).reshape( [ X.length / blockSize, blockSize ] ),
        new IntMatrix( Y ).reshape( [ Y.length ] )
    ];
}

export function gather(A, indices) {
    const shape = indices.shape ?? [ indices.length ];
    if (A.shape.length !== 2) {
        const R = createFloatMatrix( shape );
        for (let i = indices.length; i--;) {
            R[i] = A[indices[i]];
        }
        return R;
    }
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

import { getTopologicalOrder } from './2-autograd-utils.js';
import {
    transpose,
    softmaxByRow,
    negativeLogLikelihood,
    softmaxCrossEntropyGradient,
} from './1-bigram-utils.js';
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

    static addOperation(operation, forward) {
        this.operations.set(operation, forward);
        this.prototype[operation] = function (...args) {
            return new Value(null, [this, ...args], operation);
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

    // async backward() {
    //     // 1) topo‑sort & reverse to get “downstream → upstream”
    //     const revTopo = getTopologicalOrder(this).reverse();

    //     for (const node of revTopo) {
    //         node.grad = node === this ? createFloatMatrix(this.data.shape ?? [1]).fill(1) : null;
    //         node._backwardReady = (async () => {
    //             await Promise.all( node._dependents.map( async dep => await dep._backwardReady ) );
    //             if ( node._backward ) await node._backward();
    //         })();
    //     }

    //     await Promise.all( revTopo.map(node => node._backwardReady) );
    // }

    forward() {
        const order = getTopologicalOrder(this);

        for (const node of order) {
            delete node._forwardReady;
        }

        return this._forward();
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
    async ( grad ) => {
        const [ m, n ] = grad.shape;
        const biasGrad = createFloatMatrix( [ n ] );
        // Gradients for the biases are the sum of the gradients for
        // each row.
        for ( let m_ = m; m_--; ) {
            for ( let n_ = n; n_--; ) {
                biasGrad[ n_ ] += grad[ m_ * n + n_ ];
            }
        }
        return [
            await matMul( grad, transpose( B ) ),
            await matMul( transpose( A ), grad ),
            biasGrad
        ];
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
            return [B];
        }
    ];
} );

Value.addOperation( 'gather', ( A, indices ) => [
    gather( A, indices ),
    async ( grad ) => {
        const B = grad;
        let dA;
        if ( A.shape.length !== 2 ) {
            dA = createFloatMatrix( A.shape );
            for ( let i = B.length; i--; ) dA[ indices[i] ] += B[i];
        } else {
            dA = await scatterAdd( grad, indices, A.shape );
        }
        return [dA];
    }
] );

Value.addOperation( 'softmaxCrossEntropy', ( A, indices ) => {
    const data = softmaxByRow( A );
    return [
        negativeLogLikelihood( data, indices ),
        () => [ softmaxCrossEntropyGradient( data, indices ) ]
    ];
} );

Value.addOperation( 'reshape', ( A, shape ) => [
    new FloatMatrix( A ).reshape( shape ),
    ( grad ) => [ new FloatMatrix( grad ).reshape( A.shape ) ]
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
