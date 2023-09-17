function empty( shape ) {
    const array = new Float32Array( shape.reduce( ( a, b ) => a * b, 1 ) );
    array.shape = shape;
    return array;
}

function matMul(A, B) {
    const [ m, n ] = A.shape;
    const [ p, q ] = B.shape;

    if ( n !== p ) {
        throw new Error( 'Matrix dimensions do not match.' );
    }

    const C = empty( [ m, q ] );

    for ( let m_ = m; m_--; ) {
        for ( let q_ = q; q_--; ) {
            let sum = 0;
            for ( let n_ = n; n_--; ) sum += A[m_ * n + n_] * B[n_ * q + q_];
            C[m_ * q + q_] = sum;
        }
    }

    return C;
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

function initMatrix(v) {
    const array = empty([v.length, v[0].length]);
    const [m, n] = array.shape;

    for ( let m_ = m; m_--; ) {
        for ( let n_ = n; n_--; ) {
            array[m_ * n + n_] = v[m_][n_];
        }
    }

    return array;
}

class Matrix {
    constructor( data ) {
        this.data = data ? initMatrix( data ) : undefined;
        this.grad = undefined;
        this._backward = () => {};
        this._forward = () => {};
        this._prev = new Set();
        return this;
    }
    matMul( other ) {
        other = other instanceof Matrix ? other : new Matrix( other );
        const out = new Matrix();
        out._operation = 'matMul';
        this._prev = new Set( [ this, other ] );
        out._forward = () => {
            this._forward();
            other._forward();
            out.data = matMul( this.data, other.data );
        };
        out._backward = () => {
            // Gradient with respect to this.data.
            this.grad = maybeAdd( this.grad, matMul( out.grad, transpose( other.data ) ) );
            // Gradient with respect to other.data.
            other.grad = maybeAdd( other.grad, matMul( transpose( this.data ), out.grad ) );
        };
        return out;
    }
    softmaxCrossEntropy( onehotLabels ) {
        const out = new Matrix();
        onehotLabels = initMatrix( onehotLabels );
        out._operation = 'softmaxCrossEntropy';
        out._prev = new Set( [ this ] );
        out._forward = () => {
            this._forward();
            const logits = this.data;
            const R = empty( logits.shape );
            const [ m, n ] = logits.shape;

            for ( let m_ = m; m_--; ) {
                let max = -Infinity;
                for ( let n_ = n; n_--; ) {
                    const value = logits[m_ * n + n_];
                    if (value > max) max = value;
                }
                let sum = 0;
                // Calculate the counts.
                for ( let n_ = n; n_--; ) {
                    const i = m_ * n + n_;
                    // Subtract the max to avoid overflow
                    const exp = Math.exp(logits[i] - max);
                    R[i] = exp;
                    sum += exp;
                }
                // Calculate the logProbs.
                for ( let n_ = n; n_--; ) {
                    const i = m_ * n + n_;
                    R[i] = Math.log( R[i] / sum );
                    // Multiply by the onehotLabels.
                    R[i] *= onehotLabels[i];
                }
            }

            let sum = 0;
            for ( let i = R.length; i--; ) sum += R[i];
            // Account for the 0s, so divide by the number of rows.
            const mean = sum / R.shape[ 0 ];
            out.data = - mean;
        };
        out._backward = () => {
            const A = this.data;
            const B = empty(A.shape);
            const [m, n] = A.shape;

            for ( let m_ = m; m_--; ) {
                let max = -Infinity;
                let sumExp = 0;
                for ( let n_ = n; n_--; ) {
                    const value = A[m_ * n + n_];
                    if (value > max) max = value;
                }
                // Softmax.
                for ( let n_ = n; n_--; ) {
                    // Subtract the max to avoid overflow
                    const i = m_ * n + n_;
                    const expValue = Math.exp(A[i] - max);
                    sumExp += expValue;
                    B[i] = expValue;
                }
                for ( let n_ = n; n_--; ) {
                    const i = m_ * n + n_;
                    B[i] /= sumExp;
                    // Subtract the onehotLabels.
                    B[i] -= onehotLabels[i];
                    // Divide by the number of rows.
                    B[i] /= m;
                }
            }
            this.grad = maybeAdd( this.grad, B );
        };
        return out;
    }
    forward() {
        this._forward();
    }
    backward() {
        const reversed = [ ...this.getTopo() ].reverse();

        for ( const node of reversed ) {
            node.grad = null;
        }

        if ( typeof this.data === 'number' ) {
            this.grad = 1;
        } else {
            this.grad = empty( this.data.shape ).fill( 1 );
        }

        for ( const node of reversed ) {
            node._backward();
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
