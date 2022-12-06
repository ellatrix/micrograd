class Value {
    constructor( data, _children = [], _operation = '', label = '' ) {
        this.data = data;
        this.grad = 0;
        this._backward = () => {};
        this._prev = new Set( _children );
        this._operation = _operation;
        this.label = label;
        this.group = Value.group
        return this;
    }
    add( ...others ) {
        others = others.map( other => other instanceof Value ? other : new Value( other ) );
        const out = new Value( others.reduce( ( acc, other ) => acc + other.data, this.data ), [ this, ...others ], '+' );
        out._backward = () => {
            this.grad += out.grad;
            others.forEach( other => other.grad += out.grad );
        };
        return out;
    }
    sub( other ) {
        return this.add( ( new Value( -1 ) ).multiply( other ) );
    }
    multiply( other ) {
        other = other instanceof Value ? other : new Value( other );
        const out = new Value( this.data * other.data, [ this, other ], '*' );
        out._backward = () => {
            this.grad += other.data * out.grad;
            other.grad += this.data * out.grad;
        };
        return out;
    }
    tanh() {
        const t = Math.tanh( this.data );
        const out = new Value( t, [ this ], 'tanh' );
        out._backward = () => {
            this.grad += ( 1 - t ** 2 ) * out.grad;
        };
        return out;
    }
    relu() {
        const out = new Value( Math.max( 0, this.data ), [ this ], 'ReLU' );
        out._backward = () => {
            this.grad += ( out.data > 0 ? 1 : 0 ) * out.grad;
        }
        return out;
    }
    exp() {
        const out = new Value( Math.exp( this.data ), [ this ], 'exp' );
        out._backward = () => {
            this.grad += out.data * out.grad;
        };
        return out;
    }
    pow( other ) {
        const out = new Value( this.data ** other, [ this ], `**${ other }` );
        out._backward = () => {
            this.grad += other * this.data ** ( other - 1 ) * out.grad;
        };
        return out;
    }
    div( other ) {
        other = other instanceof Value ? other : new Value( other );
        return this.multiply( other.pow( -1 ) );
    }
    log() {
        const out = new Value( Math.log( this.data ), [ this ], 'log' );
        out._backward = () => {
            this.grad += ( 1 / this.data ) * out.grad;
        }
        return out;
    }
    backward() {
        const topo = [];
        const visited = new Set();

        function buildTopo( node ) {
            if ( ! visited.has( node ) ) {
                visited.add( node );

                for ( const child of node._prev ) {
                    buildTopo( child );
                }

                topo.push( node );
            }
        }

        buildTopo( this );
        this.grad = 1;

        const reversed = topo.reverse();

        for ( const node of reversed ) {
            node._backward();
        }
    }
}

