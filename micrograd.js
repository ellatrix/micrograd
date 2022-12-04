// If we're lazy to pass labels, start from "a" and increment
function createLabel() {
    // Add # as to not conflict with passed labels
    return '#' + String.fromCharCode(createLabel.counter++);
}

createLabel.counter = 97; // "a"

class Value {
    constructor( data, _children = [], _operation = '', label = createLabel() ) {
        this.data = data;
        this.grad = 0;
        this._backward = () => {};
        this._prev = new Set( _children );
        this._operation = _operation;
        this.label = label;
        return this;
    }
    toString() {
        return `Value{data:${this.data}}`;
    }
    add( other ) {
        other = other instanceof Value ? other : new Value( other );
        const out = new Value( this.data + other.data, [ this, other ], '+' );
        out._backward = () => {
            this.grad += out.grad;
            other.grad += out.grad;
        };
        return out;
    }
    sub( other ) {
        other = other instanceof Value ? other : new Value( other );
        return this.add( other.multiply( -1 ) );
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
            console.log( this )
        };
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

