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
class FloatMatrix extends matrixMixin(Float32Array) {}
class IntMatrix extends matrixMixin(Int32Array) {}

async function matMul(A, B) {
    const [ m, n ] = A.shape;
    const [ p, q ] = B.shape;
    const C = new FloatMatrix( null, [ m, q ] );

    if ( n !== p ) {
        throw new Error( 'Matrix dimensions do not match.' );
    }

    for ( let m_ = m; m_--; ) {
        for ( let q_ = q; q_--; ) {
            let sum = 0;
            for ( let n_ = n; n_--; ) {
                sum += A[m_ * n + n_] * B[n_ * q + q_];
            }
            C[m_ * q + q_] = sum;
        }
    }

    return C;
}

async function matMulBroadcast( A, B ) {
    const K = A.shape.at(-1);
    const restDims = A.shape.slice(0, -1);
    const [k2, N] = B.shape;

    if (K !== k2) {
        throw new Error(`Shape mismatch: A.shape=[${A.shape}], B.shape=[${B.shape}]`);
    }

    const flatA = new FloatMatrix(A, [restDims.reduce((a, b) => a * b, 1), K]);
    return new FloatMatrix(await matMul(flatA, B), [...restDims, N]);
}

function softmaxByRow( A ) {
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

function negativeLogLikelihood( probs, ys ) {
    const [m, n] = probs.shape;
    let sum = 0;
    for ( let m_ = m; m_--; ) {
        // Sum the logProbs (log likelihoods) of the correct label.
        sum += Math.log( probs[ m_ * n + ys[ m_ ] ] );
    }
    const mean = sum / m;
    // Mean negative log likelihood.
    return - mean;
}

function softmaxCrossEntropyGradient( probs, ys ) {
    const [m, n] = probs.shape;
    const gradient = new FloatMatrix( probs );
    for ( let m_ = m; m_--; ) {
        // Subtract 1 for the gradient of the correct label.
        gradient[ m_ * n + ys[ m_ ] ] -= 1;
        for ( let n_ = n; n_--; ) {
            // Divide by the number of rows.
            gradient[ m_ * n + n_ ] /= m;
        }
    }
    return gradient;
}

function transpose( A ) {
    const [ m, n ] = A.shape;
    const B = new FloatMatrix( null, [ n, m ] );

    for ( let m_ = m; m_--; ) {
        for ( let n_ = n; n_--; ) {
            B[n_ * m + m_] = A[m_ * n + n_];
        }
    }

    return B;
}

function gather(A, indices) {
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

async function matMulBias( A, B, bias ) {
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

function getTopologicalOrder( node ) {
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

class Value {
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

Value.addOperation( 'matMulBiasBroadcast', async ( A, B, bias ) => {
    const K = A.shape.at(-1);
    const restDims = A.shape.slice(0, -1);
    const [k2, N] = B.shape;

    if (K !== k2) {
        throw new Error(`Shape mismatch: A.shape=[${A.shape}], B.shape=[${B.shape}]`);
    }

    const restSize = restDims.reduce((a, b) => a * b, 1);
    const flatA = new FloatMatrix(A, [restSize, K]);
    const result = new FloatMatrix(await matMul(flatA, B), [...restDims, N]);

    if ( bias ) {
        if ( N !== bias.length ) {
            throw new Error('Bias vector dimension does not match the resulting matrix rows.');
        }

        // Add the biases to every row.
        for ( let m_ = restSize; m_--; ) {
            for ( let n_ = N; n_--; ) {
                result[ m_ * N + n_ ] += bias[ n_ ];
            }
        }
    }

    const out = [
        result,
        async ( grad ) => {
            const flatGrad = new FloatMatrix(grad, [restSize, N]);
            const flatGradA = await matMul(flatGrad, transpose(B));
            return new FloatMatrix(flatGradA, [...restDims, K]);
        },
        async ( grad ) => {
            const flatGrad = new FloatMatrix(grad, [restSize, N]);
            const flatGradB = await matMul(transpose(flatA), flatGrad);
            return new FloatMatrix(flatGradB, [K, N]);
        }
    ];

    if ( bias ) {
        out.push( ( grad ) => {
            const B = new FloatMatrix( null, [ N ] );
            for ( let m_ = restSize; m_--; ) {
                for ( let n_ = N; n_--; ) {
                    B[ n_ ] += grad[ m_ * N + n_ ];
                }
            }
            return B;
        } );
    }

    return out;
} );

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

Value.addOperation('batchNorm', (A, gain, bias) => {
    const n = A.shape.at(-1);
    const restDims = A.shape.slice(0, -1);
    const m = restDims.reduce((a, b) => a * b, 1);
    const bnraw = new FloatMatrix(A);
    const bnmean = new FloatMatrix(null, [n]);
    const bnvar = new FloatMatrix(null, [n]);
    const bnvarinv = new FloatMatrix(null, [n]);

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

    const bnout = new FloatMatrix(null, A.shape);

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
            const outGradSum = new FloatMatrix(null, [n]);
            const outGradXbnrawSum = new FloatMatrix(null, [n]);
    
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
            const dGain = new FloatMatrix(null, gain.shape);
    
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
            const dBias = new FloatMatrix(null, bias.shape);
    
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