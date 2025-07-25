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
class FloatMatrix extends matrixMixin(Float32Array) {}
class IntMatrix extends matrixMixin(Int32Array) {}

function createFloatMatrix( shape, fn ) {
    const length = shape.reduce((a, b) => a * b, 1);
    return new FloatMatrix( fn ? Array.from( { length }, fn ) : length ).reshape( shape );
}

async function matMul(A, B) {
    const [ m, n ] = A.shape;
    const [ p, q ] = B.shape;
    const C = createFloatMatrix( [ m, q ] );

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

    const flatA = new FloatMatrix(A).reshape( [restDims.reduce((a, b) => a * b, 1), K] );
    return new FloatMatrix(await matMul(flatA, B)).reshape( [...restDims, N] );
}

function softmax( A ) {
    let max = -Infinity;
    for ( let n_ = A.length; n_--; ) {
        const value = A[n_];
        if (value > max) max = value;
    }
    let sum = 0;
    for ( let n_ = A.length; n_--; ) {
        const i = n_;
        // Subtract the max to avoid overflow
        sum += A[i] = Math.exp(A[i] - max);
    }
    for ( let n_ = A.length; n_--; ) {
        A[n_] /= sum;
    }
}

function softmaxByRow( A ) {
    const [m, n] = A.shape;
    const B = new FloatMatrix( A );
    for ( let m_ = m; m_--; ) softmax( B.subarray( m_ * n, (m_ + 1) * n ) );
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
    const B = createFloatMatrix( [ n, m ] );

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
        const R = new FloatMatrix( shape );
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

            const [data, calculateGrad] = await opFn(...inputData);

            this.data = data;

            this._backward = async () => {
                const grads = await calculateGrad(this.grad);
                for (let i = 0; i < grads.length; i++) {
                    const child = args[i];
                    if (child instanceof Value) {
                        child.grad = child.grad ? add(child.grad, grads[i]) : grads[i];
                    }
                }
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

Value.addOperation( 'matMulBiasBroadcast', async ( A, B, bias ) => {
    const K = A.shape.at(-1);
    const restDims = A.shape.slice(0, -1);
    const [k2, N] = B.shape;

    if (K !== k2) {
        throw new Error(`Shape mismatch: A.shape=[${A.shape}], B.shape=[${B.shape}]`);
    }

    const restSize = restDims.reduce((a, b) => a * b, 1);
    // Reshape a shallow subarray, not the original!
    const flatA = A.subarray().reshape([restSize, K]);
    const result = (await matMul(flatA, B)).reshape([...restDims, N]);

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

    return [
        result,
        async ( grad ) => {
            // Reshape a shallow subarray, not the original!
            const flatGrad = grad.subarray().reshape([restSize, N]);
            const flatA = A.subarray().reshape([restSize, K]);
            const flatGradAPromise = matMul(flatGrad, transpose(B));
            const flatGradBPromise = matMul(transpose(flatA), flatGrad);
            const out = [
                (await flatGradAPromise).reshape([...restDims, K]),
                (await flatGradBPromise).reshape([K, N])
            ];
            if ( bias ) {
                const biasGrad = createFloatMatrix( [ N ] );
                for ( let m_ = restSize; m_--; ) {
                    for ( let n_ = N; n_--; ) {
                        biasGrad[ n_ ] += grad[ m_ * N + n_ ];
                    }
                }
                out.push( biasGrad );
            }
            return out;
        },
    ];
} );

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
    ( grad ) => {
        const B = grad;
        const C = createFloatMatrix( A.shape );
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

        return [C];
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

    const bnout = createFloatMatrix( A.shape );

    for (let m_ = m; m_--;) {
        for (let n_ = n; n_--;) {
            const i = m_ * n + n_;
            bnraw[i] = (A[i] - bnmean[n_]) * bnvarinv[n_];
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
            const dGain = createFloatMatrix( gain.shape );
            const dBias = createFloatMatrix( bias.shape );

            // Calculate sums along the batch dimension (m)
            for (let n_ = n; n_--;) {
                for (let m_ = m; m_--;) {
                    const i = m_ * n + n_;
                    outGradSum[n_] += grad[i];
                    outGradXbnrawSum[n_] += grad[i] * bnraw[i];
                    dGain[n_] += grad[i] * bnraw[i];
                    dBias[n_] += grad[i];
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

            return [dA, dGain, dBias];
        },
    ];
});

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

Value.addOperation('dropout', (A, dropoutProb) => {
    const mask = createFloatMatrix(A.shape);
    const out = createFloatMatrix(A.shape);
    const keepProb = 1 - dropoutProb;
    const scale = 1 / keepProb;

    for (let i = A.length; i--;) {
        if ( Math.random() < keepProb ) {
            mask[i] = 1;
            out[i] = A[i] * scale;
        }
    }

    return [
        out,
        (grad) => {
            const dA = new FloatMatrix(grad);
            for (let i = grad.length; i--;) {
                if ( mask[i] === 0 ) {
                    dA[i] = 0;
                }
            }
            return [dA];
        },
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
