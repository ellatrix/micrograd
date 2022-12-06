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

