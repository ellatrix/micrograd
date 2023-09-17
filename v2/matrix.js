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

function checkDimensions( A, B ) {
    const [ m, n ] = A.shape;
    const [ p, q ] = B.shape;

    if ( m !== p || n !== q ) {
        throw new Error( 'Matrix dimensions do not match.' );
    }
}

function add( A, B ) {
    checkDimensions( A, B );
    const C = empty( A.shape );
    for ( let i = A.length; i--; ) C[ i ] = A[ i ] + B[ i ];
    return C;
}

function subtract( A, B ) {
    checkDimensions( A, B );
    const C = empty( A.shape );
    for ( let i = A.length; i--; ) C[ i ] = A[ i ] - B[ i ];
    return C;
}

function exp( A ) {
    const B = empty( A.shape );
    for ( let i = A.length; i--; ) B[ i ] = Math.exp( A[ i ] );
    return B;
}

function log( A ) {
    const B = empty( A.shape );
    for ( let i = A.length; i--; ) B[ i ] = Math.log( A[ i ] );
    return B;
}

function multiply( A, B ) {
    checkDimensions( A, B );
    const C = empty( A.shape );
    for ( let i = A.length; i--; ) C[ i ] = A[ i ] * B[ i ];
    return C;
}

function divide( A, b ) {
    const B = empty( A.shape );
    for ( let i = A.length; i--; ) B[ i ] = A[ i ] / b;
    return B;
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

function ones( A ) {
    return empty( A.shape ).fill( 1 );
}

function maxFillRow( A ) {
    const B = empty( A.shape );
    const [ m, n ] = A.shape;

    for ( let m_ = m; m_--; ) {
        let _max = -Infinity;
        for ( let n_ = n; n_--; ) {
            const value = A[m_ * n + n_];
            if (value > _max) _max = value;
        }
        for ( let n_ = n; n_--; ) B[m_ * n + n_] = _max;
    }

    return B;
}

function probs(A) {
    const B = empty(A.shape);
    const [m, n] = A.shape;

    for ( let m_ = m; m_--; ) {
        let sum = 0;
        for ( let n_ = n; n_--; ) sum += A[m_ * n + n_];
        for ( let n_ = n; n_--; ) B[m_ * n + n_] = A[m_ * n + n_] / sum;
    }

    return B;
}


function mean( A ) {
    let total = 0;
    for ( let i = A.length; i--; ) total += A[i];
    return total / A.length;
}

function softmaxRow(A) {
    const B = empty(A.shape);
    const [m, n] = A.shape;

    for ( let m_ = m; m_--; ) {
        let max = -Infinity;
        let sumExp = 0;
        for ( let n_ = n; n_--; ) {
            const value = A[m_ * n + n_];
            if (value > max) max = value;
        }
        for ( let n_ = n; n_--; ) {
            // Subtract the max to avoid overflow
            const expValue = Math.exp(A[m_ * n + n_] - max);
            sumExp += expValue;
            B[m_ * n + n_] = expValue;
        }
        for ( let n_ = n; n_--; ) B[m_ * n + n_] /= sumExp;
    }

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
    get( what ) {
        if ( typeof this[ what ] === 'number' ) {
            return this[ what ];
        }
        
        const [ m, n ] = this[ what ].shape;
        const array = [];

        for ( let i = 0; i < m; i++ ) {
            const row = [];
            for ( let j = 0; j < n; j++ ) {
                row.push( this[ what ][ i * n + j ] );
            }
            array.push( row );
        }

        return array;
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
            const normLogits = subtract( logits, maxFillRow( logits ) );
            const counts = exp( normLogits );
            const _probs = probs( counts );
            const logProbs = log( _probs );
            const relevantLogProbs = multiply( logProbs, onehotLabels );
            // Account for the 0s in the onehotLabels.
            out.data = - mean( relevantLogProbs ) * relevantLogProbs.shape[ 1 ];
        };
        out._backward = () => {
            const _softmax = softmaxRow( this.data );
            this.grad = maybeAdd( this.grad, divide( subtract( _softmax, onehotLabels ), _softmax.shape[0] ) );
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
            this.grad = ones( this.data );
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
