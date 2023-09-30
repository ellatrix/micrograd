function empty( shape ) {
    const array = new Float32Array( shape.reduce( ( a, b ) => a * b, 1 ) );
    array.shape = shape;
    return array;
}

function clone( array ) {
    const clone = new Float32Array( array );
    clone.shape = array.shape;
    return clone;
}

function softmaxByRow( A ) {
    const [m, n] = A.shape;
    const B = empty(A.shape);
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

function add( A, B ) {
    if ( A.shape.toString() !== B.shape.toString() ) {
        throw new Error( 'Matrix dimensions do not match.' );
    }

    const C = empty( A.shape );
    for ( let i = A.length; i--; ) C[ i ] = A[ i ] + B[ i ];
    return C;
}

function maybeAdd( a, b ) {
    return a ? add( a, b ) : b;
}

function transpose( A ) {
    const [ m, n ] = A.shape;
    const B = empty( [ n, m ] );

    for ( let m_ = m; m_--; )
        for ( let n_ = n; n_--; )
            B[n_ * m + m_] = A[m_ * n + n_];

    return B;
}

function oneHot( a, length ) {
    const B = empty( [ a.length, length ] );
    for ( let i = a.length; i--; ) B[ i * length + a[ i ] ] = 1;
    return B;
}

function randomMinMax(min, max) {
    return Math.random() * (max - min) + min;
}

function random( shape ) {
    const A = empty( shape );
    for ( let i = A.length; i--; ) A[ i ] = randomMinMax( -1, 1 );
    return A;
}

function sample(probs) {
    const sum = probs.reduce((a, b) => a + b, 0)
    if (sum <= 0) throw Error('probs must sum to a value greater than zero')
    const normalized = probs.map(prob => prob / sum)
    const sample = Math.random()
    let total = 0
    for (let i = 0; i < normalized.length; i++) {
        total += normalized[i]
        if (sample < total) return i
    }
}

function addBias( A, bias ) {
    const [ m, n ] = A.shape;
    if (n !== bias.length ) {
        throw new Error('Bias vector dimension does not match the resulting matrix rows.');
    }
    const B = clone( A );
    for ( let m_ = m; m_--; ) {
        for ( let n_ = n; n_--; ) {
            B[ m_ * n + n_ ] += bias[ n_ ];
        }
    }
    return B;
}

function sumBiasGrad( A ) {
    const [ m, n ] = A.shape;
    const B = empty( [ n ] );
    for ( let m_ = m; m_--; ) {
        for ( let n_ = n; n_--; ) {
            B[ n_ ] += A[ m_ * n + n_ ];
        }
    }
    return B;
}

function gather(A, indices) {
    if (A.shape.length !== 2) {
        const R = empty(indices.shape);
        for (let i = indices.length; i--;) {
            R[i] = A[indices[i]];
        }
        return R;
    }
    const Dim = A.shape[1];
    const R = empty([...indices.shape, Dim]);
    for (let i = indices.length; i--;) {
        const index = indices[i];
        for (let j = Dim; j--;) {
            R[i * Dim + j] = A[index * Dim + j];
        }
    }
    return R;
}

class Layer {
    constructor( data ) {
        this.data = data;
        this.grad = null;
        this._backward = async () => {};
        this._forward = async () => {};
        this._prev = new Set();
        return this;
    }
    matMul( other, bias ) {
        const matMul = Layer.gpu ? Layer.gpu.matMul : Layer.cpu.matMul;
        other = other instanceof Layer ? other : new Layer( other );
        bias = bias instanceof Layer ? bias : new Layer( bias );
        const out = new Layer();
        out._operation = 'matMul';
        out._prev = new Set( [ this, other, bias ] );
        out._forward = async () => {
            await this._forward();
            await other._forward();
            out.data = await matMul( this.data, other.data );
            if ( bias.data ) {
                await bias._forward();
                out.data = addBias( out.data, bias.data );
            }
        };
        out._backward = async () => {
            // Gradient with respect to this.data.
            this.grad = maybeAdd( this.grad, await matMul( out.grad, transpose( other.data ) ) );
            // Gradient with respect to other.data.
            other.grad = maybeAdd( other.grad, await matMul( transpose( this.data ), out.grad ) );
            // Gradient with respect to bias.data.
            if ( bias.data ) {
                bias.grad = maybeAdd( bias.grad, sumBiasGrad( out.grad ) );
            }
        };
        return out;
    }
    tanh() {
        const out = new Layer();
        out._operation = 'tanh';
        out._prev = new Set( [ this ] );
        out._forward = async () => {
            await this._forward();
            const A = this.data;
            const B = empty( A.shape );
            for ( let i = A.length; i--; ) B[ i ] = Math.tanh( A[ i ] );
            out.data = B;
        }
        out._backward = async () => {
            const B = out.grad;
            const tanhA = out.data;
            const C = empty( tanhA.shape );
            for ( let i = tanhA.length; i--; ) C[ i ] = B[i] * (1 - Math.pow(tanhA[i], 2));
            this.grad = maybeAdd( this.grad, C );
        }
        return out;
    }
    gather( indices ) {
        indices = indices instanceof Layer ? indices : new Layer( indices );
        const out = new Layer();
        out._operation = 'gather';
        out._prev = new Set( [ this, indices ] );
        out._forward = async () => {
            await this._forward();
            await indices._forward();
            out.data = gather( this.data, indices.data );
        }
        out._backward = async () => {
            const B = out.grad;
            const C = empty( this.data.shape );
            for ( let i = B.length; i--; ) C[ indices.data[i] ] += B[i];
            this.grad = maybeAdd( this.grad, C );
        }
        return out;
    }
    reshape( fn ) {
        const out = new Layer();
        out._operation = 'reshape';
        out._prev = new Set( [ this ] );
        out._forward = async () => {
            await this._forward();
            out.data = clone( this.data );
            out.data.shape = fn( this.data.shape );
        }
        out._backward = async () => {
            const B = clone( out.grad );
            B.shape = this.data.shape;
            this.grad = maybeAdd( this.grad, B );
        }
        return out;
    }
    // Somehow shortcut with gather?
    softmaxCrossEntropy( onehotLabels ) {
        onehotLabels = onehotLabels instanceof Layer ? onehotLabels : new Layer( onehotLabels );
        const out = new Layer();
        out._operation = 'softmaxCrossEntropy';
        out._prev = new Set( [ this ] );
        out._forward = async () => {
            await this._forward();
            const logits = this.data;
            // Probabilites.
            const R = softmaxByRow( logits );
            this.sofmaxResult = clone( R );
            const [ m, n ] = R.shape;

            for ( let m_ = m; m_--; ) {
                for ( let n_ = n; n_--; ) {
                    const i = m_ * n + n_;
                    // Calculate the logProbs (log likelihoods).
                    R[i] = Math.log( R[i] );
                    // Multiply by the onehotLabels.
                    R[i] *= onehotLabels.data[i];
                }
            }

            let sum = 0;
            for ( let i = R.length; i--; ) sum += R[i];
            // Account for the 0s, so divide by the number of rows.
            const mean = sum / R.shape[ 0 ];
            // Loss = average negative log likelihood.
            out.data = empty( [] ).fill( - mean );
        };
        out._backward = async () => {
            const B = this.sofmaxResult;
            const [m, n] = B.shape;

            for ( let m_ = m; m_--; ) {
                for ( let n_ = n; n_--; ) {
                    const i = m_ * n + n_;
                    // Subtract the onehotLabels.
                    B[i] -= onehotLabels.data[i];
                    // Divide by the number of rows.
                    B[i] /= m;
                }
            }

            this.grad = maybeAdd( this.grad, B );
        };
        return out;
    }
    async forward() {
        await this._forward();
    }
    async backward() {
        const reversed = [ ...this.getTopo() ].reverse();

        for ( const node of reversed ) {
            node.grad = null;
        }

        this.grad = empty( this.data.shape ).fill( 1 );

        for ( const node of reversed ) {
            await node._backward();
        }
    }
    getTopo() {
        if ( this.topo ) {
            return this.topo;
        }

        this.topo = [];

        const visited = new Set();

        const buildTopo = ( node ) => {
            if ( ! visited.has( node ) ) {
                visited.add( node );

                for ( const child of node._prev ) {
                    buildTopo( child );
                }

                this.topo.push( node );
            }
        }

        buildTopo( this );

        return this.topo;
    }
}

Layer.cpu = { matMul }
