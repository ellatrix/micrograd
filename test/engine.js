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

function matMul(A, B) {
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

Value.addOperation('batchNorm', (A, gain, bias) => {
    A = A.data;
    const [m, n] = A.shape;
    bnraw = new FloatMatrix(A);
    bnmean = new FloatMatrix(null, [n]);
    bnvar = new FloatMatrix(null, [n]);
    bnvarinv = new FloatMatrix(null, [n]);

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
        bnvar[n_] = variance / m;
        bnvarinv[n_] = 1 / Math.sqrt(bnvar[n_] + 1e-5);
    }

    for (let m_ = m; m_--;) {
        for (let n_ = n; n_--;) {
            const i = m_ * n + n_;
            bnraw[i] = (A[i] - bnmean[n_]) * bnvarinv[n_];
        }
    }

    gain = gain.data;
    bias = bias.data;

    const bnout = new FloatMatrix(bnraw);

    for (let m_ = m; m_--;) {
        for (let n_ = n; n_--;) {
            const i = m_ * n + n_;
            bnout[i] = gain[n_] * bnraw[i] + bias[n_];
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
                dA[i] = gain_data[n_] * bnvarinv[n_] / n * (n * outGrad[i] - outGradSum[m_] - n / (n - 1) * bnraw[i] * outGradXbnrawSum[m_]);
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
                dGain[n_] += outGrad[m_ * n + n_] * A_data[m_ * n + n_];
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
                dBias[n_] += outGrad[m_ * n + n_];
            }
        }

        return dBias;
    }
]);
