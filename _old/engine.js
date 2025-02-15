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
    constructor( data, _children = [], _operation = '', label = '' ) {
        if ( typeof data === 'function' ) {
            this._forward = data;
        } else {
            this.data = data;
        }
        this.grad = 0;
        this._prev = new Set( new Map( _children ).keys() );
        this._backward = () => {
            // Beware: Map removes duplicate keys, but we want to accumulate
            // gradients.
            for ( const [ node, gradFn ] of _children ) {
                node.grad += gradFn();
            }
        };
        this._operation = _operation;
        this.label = label;
        this.group = Value.group
        return this;
    }
    add( ...others ) {
        others = others.map( other => other instanceof Value ? other : new Value( other ) );
        const out = new Value(
            () => others.reduce( ( sum, other ) => sum + other.data, this.data ),
            [ this, ...others ].map( other => [ other, () => out.grad ] ), '+' );
        return out;
    }
    sub( other ) {
        return this.add( ( new Value( -1 ) ).mul( other ) );
    }
    mul( other ) {
        other = other instanceof Value ? other : new Value( other );
        const out = new Value( () => this.data * other.data, [
            [ this, () => other.data * out.grad ],
            [ other, () => this.data * out.grad ]
        ], '*' );
        return out;
    }
    tanh() {
        const out = new Value( () => Math.tanh( this.data ), [
            [ this, () => ( 1 - Math.tanh( this.data ) ** 2 ) * out.grad ]
        ], 'tanh' );
        return out;
    }
    relu() {
        const out = new Value( () => Math.max( 0, this.data ), [
            [ this, () => ( out.data > 0 ? 1 : 0 ) * out.grad ]
        ], 'ReLU' );
        return out;
    }
    exp() {
        const out = new Value( () => Math.exp( this.data ), [
            [ this, () => out.data * out.grad ]
        ], 'exp' );
        return out;
    }
    pow( other ) {
        const out = new Value( () => this.data ** other, [
            [ this, () => other * this.data ** ( other - 1 ) * out.grad ]
        ], `**${ other }` );
        return out;
    }
    div( other ) {
        other = other instanceof Value ? other : new Value( other );
        return this.mul( other.pow( -1 ) );
    }
    log() {
        const out = new Value( () => Math.log( this.data ), [
            [ this, () => ( 1 / this.data ) * out.grad ]
        ], 'log' );
        return out;
    }
    forward() {
        for ( const node of getTopologicalOrder( this ) ) {
            if ( node._forward ) {
                node.data = node._forward();
            }
        }
    }
    backward() {
        const reversed = getTopologicalOrder( this ).reverse();

        for ( const node of reversed ) {
            node.grad = 0;
        }

        this.grad = 1;

        for ( const node of reversed ) {
            node._backward();
        }
    }
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
    softmaxCrossEntropy( onehotLabels ) {
        const out = new Matrix( undefined, [ this ], 'softmaxCrossEntropy' );
        out._forward = () => {
            this._forward();
            const logits = this.data;
            const normLogits = logits.subtract( logits.max( 1, true ) );
            const counts = normLogits.exp();
            const probs = counts.divide( nj.array( counts.tolist().map( row => {
                const sum = row.reduce( ( a, b ) => a + b, 0 );
                return row.map( () => sum );
            } ) ) );
            const logProbs = probs.log();
            const relevantLogProbs = logProbs.multiply( onehotLabels );
            const removeEmpty = nj.array( relevantLogProbs.tolist().map( row => {
                const sum = row.reduce( ( a, b ) => a + b, 0 );
                return row.map( () => sum );
            } ) );
            const mean = removeEmpty.mean();
            out.data = nj.array( -mean );
        };
        out._backward = () => {
            const softmax = nj.array( this.data.tolist().map( row => {
                return nj.softmax( nj.array( row ) ).tolist();
            } ) );
            this.grad = maybeAdd( this.grad, softmax.subtract( onehotLabels ).divide( softmax.shape[ 0 ] ) );
        };
        return out;
    }
    softmax() {
        const out = new Matrix( undefined, [ this ], 'softmax' );
        out._forward = () => {
            this._forward();
            out.data = nj.array( this.data.tolist().map( row => {
                return nj.softmax( nj.array( row ) ).tolist();
            } ) );
        };
        out._backward = () => {};
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

