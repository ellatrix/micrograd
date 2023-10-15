function empty( shape ) {
    const array = new Float32Array( shape.reduce( ( a, b ) => a * b, 1 ) );
    array.shape = shape;
    return array;
}

function clone( array, shape = array.shape ) {
    const clone = new Float32Array( array );
    clone.shape = shape;
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

function random( shape, multiplier = 1 ) {
    const A = empty( shape );
    for ( let i = A.length; i--; ) A[ i ] = randomMinMax( -1, 1 ) * multiplier;
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

function reshape(A, fn) {
    const B = clone(A);
    B.shape = [ ...fn( A.shape ) ];
    return B;
}

function getTopologicalOrder( node ) {
    const result = [];
    const visited = new Set();

    function visit( node ) {
        if ( visited.has( node ) ) return;
        visited.add( node );
        for ( const child of node._prev ) visit( child );
        result.push( node );
    }

    visit( node );

    return result;
}

class Value {
    constructor( data, ...grads ) {
        if ( typeof data === 'function' ) {
            this._forward = data;
        } else {
            this.data = data;
        }
        this.grad = null;
        this._prev = new Set( new Map( grads ).keys() );
        this._backward = async () => {
            // Beware: Map removes duplicate keys, but we want to accumulate
            // gradients.
            for ( const [ node, fn ] of grads ) {
                if ( node._grad ) {
                    const grad = await fn( this );
                    node.grad = node.grad ? add( node.grad, grad ) : grad;
                }
            }
        };
        this._grad = [ ...this._prev ].some( ( _ ) => _._grad );
        return this;
    }
    matMul( other, bias ) {
        const matMul = Value.gpu ? Value.gpu.matMul : Value.cpu.matMul;
        return new Value(
            async () => {
                const data = await matMul(this.data, other.data);
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
            },
            [ this, async ( out ) => await matMul( out.grad, transpose( other.data ) ) ],
            [ other, async ( out ) => await matMul( transpose( this.data ), out.grad ) ],
            ...( bias ? [ [ bias, async ( out ) => {
                const A = out.grad;
                const [ m, n ] = A.shape;
                const B = empty( [ n ] );
                // Gradients for the biases are the sum of the gradients for
                // each row.
                for ( let m_ = m; m_--; ) {
                    for ( let n_ = n; n_--; ) {
                        B[ n_ ] += A[ m_ * n + n_ ];
                    }
                }
                return B;
            } ] ] : [] )
        );
    }
    tanh() {
        return new Value(
            async () => {
                const data = clone( this.data );
                for ( let i = data.length; i--; ) data[ i ] = Math.tanh( data[ i ] );
                return data;
            },
            [ this, async ( out ) => {
                const B = clone( out.grad );
                const tanhA = out.data;
                for ( let i = B.length; i--; ) B[ i ] *= ( 1 - Math.pow( tanhA[ i ], 2 ) );
                return B;
            } ]
        );
    }
    gather( indices ) {
        return new Value(
            async () => gather( this.data, indices ),
            [ this, async ( out ) => {
                const B = out.grad;
                const C = empty( this.data.shape );
                if ( this.data.shape.length !== 2 ) {
                    for ( let i = B.length; i--; ) C[ indices[i] ] += B[i];
                } else {
                    const Dim = this.data.shape[1];
                    for ( let i = B.length; i--; ) {
                        const index = indices[i];
                        for ( let j = Dim; j--; ) {
                            C[ index * Dim + j ] += B[ i * Dim + j ];
                        }
                    }
                }

                return C;
            } ]
        );
    }
    reshape( fn ) {
        return new Value(
            async () => reshape( this.data, fn ),
            [ this, async ( out ) => reshape( out.grad, () => this.data.shape ) ]
        );
    }
    softmaxCrossEntropy( labels ) {
        let sofmaxResult;
        return new Value(
            async () => {
                // Probabilites.
                sofmaxResult = softmaxByRow( this.data );
                const [m, n] = sofmaxResult.shape;

                let sum = 0;
                for ( let m_ = m; m_--; ) {
                    // Sum the logProbs (log likelihoods) of the correct label.
                    sum += Math.log( sofmaxResult[ m_ * n + labels[ m_ ] ] );
                }

                // Divide by the number of rows (amount of labels).
                const mean = sum / m;
                // Loss = average negative log likelihood.
                return empty( [] ).fill( - mean );
            },
            [ this, async () => {
                const B = clone( sofmaxResult );
                const [m, n] = B.shape;

                for ( let m_ = m; m_--; ) {
                    // Subtract 1 for the gradient of the correct label.
                    B[ m_ * n + labels[ m_ ] ] -= 1;
                    for ( let n_ = n; n_--; ) {
                        // Divide by the number of rows.
                        B[ m_ * n + n_ ] /= m;
                    }
                }

                return B;
            } ]
        );
    }
    async forward() {
        for ( const node of getTopologicalOrder( this ) ) {
            if ( node._forward ) {
                node.data = await node._forward();
            }
        }

        return this.data;
    }
    async backward() {
        const reversed = getTopologicalOrder( this ).reverse();

        for ( const node of reversed ) {
            node.grad = null;
        }

        this.grad = empty( this.data.shape ).fill( 1 );

        for ( const node of reversed ) {
            await node._backward();
        }
    }
}

Value.cpu = { matMul }

class Variable extends Value {
    constructor( data ) {
        super( data );
        this._grad = true;
        return this;
    }
}
