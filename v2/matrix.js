function matMul(A, B) {
    let m = A.length;
    let n = A[0].length;
    let p = B[0].length;
    
    let C = Array(m).fill().map(() => Array(p).fill(0));

    for (let i = 0; i < m; i++) {
        for (let j = 0; j < p; j++) {
            for (let k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

function add( A, B ) {
    return A.map( ( row, i ) => row.map( ( col, j ) => col + B[ i ][ j ] ) );
}

function subtract( A, B ) {
    return A.map( ( row, i ) => row.map( ( col, j ) => col - B[ i ][ j ] ) );
}

function exp( A ) {
    return A.map( ( row ) => row.map( ( col ) => Math.exp( col ) ) );
}

function log( A ) {
    return A.map( ( row ) => row.map( ( col ) => Math.log( col ) ) );
}

function multiply( A, B ) {
    return A.map( ( row, i ) => row.map( ( col, j ) => col * B[ i ][ j ] ) );
}

function divide( A, b ) {
    return A.map( ( row, i ) => row.map( ( col, j ) => col / b ) );
}

function maybeAdd( a, b ) {
    return a ? add( a, b ) : b;
}

function transpose( A ) {
    return A[ 0 ].map( ( _, colIndex ) => A.map( row => row[ colIndex ] ) );
}

function ones( rows, cols ) {
    return Array.from( { length: rows }, () => Array.from( { length: cols }, () => 1 ) );
}

function max( A ) {
    return A.map( ( row ) => {
        const max = Math.max( ...row );
        return Array( row.length ).fill( max );
    } );
} 

function mean( A ) {
    let sum = 0;
    let count = 0;

    A.forEach(row => {
        row.forEach(value => {
            sum += value;
            count++;
        });
    });

    return sum / count;
}

function sum(array) {
    let total = 0;
    let i = array.length;
    while (i--) total += array[i];
    return total;
}

function softmax( array ) {
    const max = Math.max( ...array );
    const exps = array.map( x => Math.exp( x - max ) );
    const sum = exps.reduce( ( a, b ) => a + b, 0 );
    return exps.map( x => x / sum );
}

function initMatrix( v ) {
    const array = new Float32Array( v.flat() );
    array.shape = [ v.length, v[ 0 ].length ];
    return array;
}

class Matrix {
    constructor( data ) {
        this.data = data;
        this.grad = undefined;
        this._backward = () => {};
        this._forward = () => {};
        this._prev = new Set();
        return this;
    }
    matMul( other ) {
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
        out._operation = 'softmaxCrossEntropy';
        out._prev = new Set( [ this ] );
        out._forward = () => {
            this._forward();
            const logits = this.data;
            const normLogits = subtract( logits, max( logits ) );
            const counts = exp( normLogits );
            const probs = counts.map( row => {
                const _sum = sum( row );
                return row.map( ( x ) => x / _sum );
            } )
            const logProbs = log( probs );
            const relevantLogProbs = multiply( logProbs, onehotLabels );
            const removeEmpty = relevantLogProbs.map( row => {
                const _sum = sum( row );
                return row.map( () => _sum );
            } );
            out.data = - mean( removeEmpty );
        };
        out._backward = () => {
            const _softmax = this.data.map( row => {
                return softmax( row );
            } );
            this.grad = maybeAdd( this.grad, divide( subtract( _softmax, onehotLabels ), _softmax.length ) );
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
            this.grad = ones( this.data.length, this.data[ 0 ].length );
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
