class Value {
    constructor( data, _children = [], _operation = '', label = '' ) {
        this.data = data;
        this.grad = 0;
        this._backward = () => {};
        this._forward = () => {};
        this._prev = new Set( _children );
        this._operation = _operation;
        this.label = label;
        this.group = Value.group
        return this;
    }
    add( ...others ) {
        others = others.map( other => other instanceof Value ? other : new Value( other ) );
        const out = new Value( undefined, [ this, ...others ], '+' );
        out._forward = () => {
            this._forward();
            out.data = this.data;
            others.forEach( other => {
                other._forward();
                out.data += other.data;
            } );
        };
        out._backward = () => {
            this.grad += out.grad;
            others.forEach( other => other.grad += out.grad );
        };
        return out;
    }
    sub( other ) {
        return this.add( ( new Value( -1 ) ).mul( other ) );
    }
    mul( other ) {
        other = other instanceof Value ? other : new Value( other );
        const out = new Value( undefined, [ this, other ], '*' );
        out._forward = () => {
            this._forward();
            other._forward();
            out.data = this.data * other.data;
        }
        out._backward = () => {
            this.grad += other.data * out.grad;
            other.grad += this.data * out.grad;
        };
        return out;
    }
    tanh() {
        const out = new Value( undefined, [ this ], 'tanh' );
        out._forward = () => {
            this._forward();
            out.data = Math.tanh( this.data );
        };
        out._backward = () => {
            this.grad += ( 1 - Math.tanh( this.data ) ** 2 ) * out.grad;
        };
        return out;
    }
    relu() {
        const out = new Value( undefined, [ this ], 'ReLU' );
        out._forward = () => {
            this._forward();
            out.data = Math.max( 0, this.data );
        };
        out._backward = () => {
            this.grad += ( out.data > 0 ? 1 : 0 ) * out.grad;
        }
        return out;
    }
    exp() {
        const out = new Value( undefined, [ this ], 'exp' );
        out._forward = () => {
            this._forward();
            out.data = Math.exp( this.data );
        };
        out._backward = () => {
            this.grad += out.data * out.grad;
        };
        return out;
    }
    pow( other ) {
        const out = new Value( undefined, [ this ], `**${ other }` );
        out._forward = () => {
            this._forward();
            out.data = this.data ** other;
        };
        out._backward = () => {
            this.grad += other * this.data ** ( other - 1 ) * out.grad;
        };
        return out;
    }
    div( other ) {
        other = other instanceof Value ? other : new Value( other );
        return this.mul( other.pow( -1 ) );
    }
    log() {
        const out = new Value( undefined, [ this ], 'log' );
        out._forward = () => {
            this._forward();
            out.data = Math.log( this.data );
        };
        out._backward = () => {
            this.grad += ( 1 / this.data ) * out.grad;
        }
        return out;
    }
    forward() {
        this._forward();
    }
    backward() {
        const reversed = this.getTopo().reverse();

        for ( const node of reversed ) {
            node.grad = 0;
        }

        this.grad = 1;

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
    // forward() {
    //     this._forward();
    // }
    // backward() {
    //     this.grad = 1;

    //     const reversed = this.getTopo().reverse();

    //     for ( const node of reversed ) {
    //         node._backward();
    //     }
    // }
}

function maybeAdd( a, b ) {
    return a ? a.add( b ) : b;
}

class Matrix {
    constructor( data, _children = [], _operation = '', label = '' ) {
        this.data = data;
        this.grad = undefined;
        this._backward = () => {};
        this._forward = () => {};
        this._prev = new Set( _children );
        this._operation = _operation;
        this.label = label;
        this.group = Matrix.group;
        return this;
    }
    matMul( other ) {
        other = other instanceof Matrix ? other : new Matrix( other );
        const out = new Matrix( undefined, [ this, other ], 'matMul' );
        out._forward = () => {
            this._forward();
            other._forward();
            out.data = this.data.dot( other.data );
        };
        out._backward = () => {
            // Gradient with respect to this.data.
            this.grad = maybeAdd( this.grad, out.grad.dot( other.data.T ) );
            // Gradient with respect to other.data.
            other.grad = maybeAdd( other.grad, this.data.T.dot( out.grad ) );
        };
        return out;
    }
    mul( other ) {
        other = other instanceof Matrix ? other : new Matrix( other );
        const out = new Matrix( undefined, [ this, other ], '*' );
        out._forward = () => {
            this._forward();
            other._forward();
            out.data = this.data.multiply( other.data );
        };
        out._backward = () => {
            this.grad = maybeAdd( this.grad, other.data.multiply( out.grad ) );
            other.grad = maybeAdd( other.grad, this.data.multiply( out.grad ) );
        };
        return out;
    }
    sum( axis = 0 ) {
        const out = new Matrix( undefined, [ this ], 'sum' + axis );
        out._forward = () => {
            this._forward();
            if ( axis === 0 ) {
                out.data = nj.array( this.data.sum() );
            } else {
                out.data = nj.array( this.data.tolist().map( row => {
                    const sum = row.reduce( ( a, b ) => a + b, 0 );
                    return row.map( () => sum );
                } ) );
            }
        };
        out._backward = () => {
            if ( axis === 0 ) {
                const sumGrad = out.grad.tolist()[ 0 ];
                const sumGradBroadcast = nj.array( this.data.tolist().map( () => sumGrad ) );
                this.grad = maybeAdd( this.grad, sumGradBroadcast );
            } else {
                this.grad = maybeAdd( this.grad, nj.array( out.grad.tolist().map( ( [ sum ] ) => Array.from( { length: this.data.shape[ 1 ] }, () => sum ) ) ) );
            }
        };
        return out;
    }
    divSumRows() {
        const out = new Matrix( undefined, [ this ], 'divSumRows' );
        out._forward = () => {
            this._forward();
            out.data = this.data.divide( nj.array( this.data.tolist().map( row => {
                const sum = row.reduce( ( a, b ) => a + b, 0 );
                return row.map( () => sum );
            } ) ) );
        };
        out._backward = () => {
            this.grad = maybeAdd(
                this.grad,
                nj.array( out.grad.tolist().map( ( [ grad ] ) =>
                    Array.from( { length: this.data.shape[ 1 ] }, () => grad / this.data.shape[ 1 ] )
                ) )
            );
        };
        return out;
    }
    mean() {
        const out = new Matrix( undefined, [ this ], 'mean' );
        out._forward = () => {
            this._forward();
            out.data = nj.array( this.data.mean() );
        };
        out._backward = () => {
            const meanGrad = out.grad.tolist()[ 0 ] / this.data.shape.reduce( ( a, b ) => a * b, 1 );
            const meanGradBroadcast = nj.zeros( this.data.shape ).add( meanGrad );
            this.grad = maybeAdd( this.grad, meanGradBroadcast );
        };
        return out;
    }
    gather( indices ) {
        const out = new Matrix( undefined, [ this ], 'gather' );
        out._forward = () => {
            this._forward();
            out.data = nj.array( this.data.tolist().map( ( row, j ) => {
                return row[ indices[ j ] ];
            } ) );
        };
        out._backward = () => {
            const outGradList = out.grad.tolist();
            const broadcast = nj.array( this.data.tolist().map( ( row, j ) => {
                return row.map( ( _, i ) => {
                    return i === indices[ j ] ? outGradList[ j ] : 0;
                } );
            } ) );
            this.grad = maybeAdd( this.grad, broadcast );
        };
        return out;
    }
    add( other ) {
        other = other instanceof Matrix ? other : new Matrix( other );
        const out = new Matrix( undefined, [ this, other ], '+' );
        out._forward = () => {
            this._forward();
            other._forward();
            out.data = this.data.add( other.data );
        };
        out._backward = () => {
            this.grad = maybeAdd( this.grad, out.grad );
            other.grad = maybeAdd( other.grad, out.grad );
        };
        return out;
    }
    sub( other ) {
        return this.add( other.mul( nj.zeros( this.data.shape ).assign( -1 ) ) );
    }
    div( other ) {
        other = other instanceof Matrix ? other : new Matrix( other );
        return this.mul( other.pow( -1 ) );
    }
    pow( other ) {
        const out = new Matrix( undefined, [ this ], `**${ other }` );
        out._forward = () => {
            this._forward();
            out.data = this.data.pow( nj.zeros( this.data.shape ).assign( other ) );
        };
        out._backward = () => {
            this.grad = maybeAdd( this.grad, out.grad.multiply( this.data.pow( other - 1 ).multiply( other ) ) );
        };
        return out;
    }
    exp() {
        const out = new Matrix( undefined, [ this ], 'exp' );
        out._forward = () => {
            this._forward();
            out.data = this.data.exp();
        };
        out._backward = () => {
            this.grad = maybeAdd( this.grad, out.grad.multiply( out.data ) );
        };
        return out;
    }
    log() {
        const out = new Matrix( undefined, [ this ], 'log' );
        out._forward = () => {
            this._forward();
            out.data = this.data.log();
        };
        out._backward = () => {
            this.grad = maybeAdd( this.grad, out.grad.divide( this.data ) );
        };
        return out;
    }
    // tanh() {
    //     const out = new Matrix( undefined, [ this ], 'tanh' );
    //     out._forward = () => {
    //         this._forward();
    //         out.data = this.data.tanh();
    //     };
    //     out._backward = () => {
    //         this.grad = this.grad.add( out.grad.multiply( out.data.multiply( out.data ).multiply( -1 ).add( 1 ) ) );
    //     };
    //     return out;
    // }
    forward() {
        this._forward();
    }
    backward() {
        const reversed = [ ...this.getTopo() ].reverse();

        for ( const node of reversed ) {
            node.grad = null;
        }

        this.grad = nj.ones( this.data.shape );

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

