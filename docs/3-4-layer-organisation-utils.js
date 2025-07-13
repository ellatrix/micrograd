
import { random } from './1-bigram-utils.js';
import { Value, FloatMatrix } from './3-0-makemore-MLP-utils.js';
import './3-3-batch-norm-utils.js';

export class Linear {
    constructor( fan_in, fan_out, bias = true ) {
        this.weight = new Value( new FloatMatrix( () => random() / fan_in ** 0.5, [ fan_in, fan_out ] ) );
        if ( bias ) {
            this.bias = new Value( new FloatMatrix( () => 0, [ fan_out ] ) );
        }
    }
    apply( X ) {
        return X.matMulBias( this.weight, this.bias );
    }
    params() {
        return this.bias ? [ this.weight, this.bias ] : [ this.weight ];
    }
}
export class BatchNorm1d {
    constructor( dim ) {
        this.gain = new Value( new FloatMatrix( () => 1, [ dim ] ) );
        this.bias = new Value( new FloatMatrix( () => 0, [ dim ] ) );
    }
    apply( X ) {
        return X.batchNorm( this.gain, this.bias );
    }
    params() {
        return [ this.gain, this.bias ];
    }
}
export class Tanh {
    apply( X ) {
        return X.tanh();
    }
    params() {
        return [];
    }
}
export class Embedding {
    constructor( vocabSize, embeddingDimensions ) {
        this.weight = new Value( new FloatMatrix( random, [ vocabSize, embeddingDimensions ] ) )
    }
    apply( X ) {
        return this.weight.gather( X );
    }
    params() {
        return [ this.weight ];
    }
}
export class Flatten {
    constructor() {}
    apply( X ) {
        return X.reshape( ( [ first, ...rest ] ) => [ first, rest.reduce( ( acc, curr ) => acc * curr, 1 ) ] );
    }
    params() {
        return [];
    }
}
export class Sequential {
    constructor( layers ) {
        this.layers = layers;
    }
    apply( X ) {
        return this.layers.reduce( ( acc, layer ) => layer.apply( acc ), X );
    }
    params() {
        return this.layers.flatMap( layer => layer.params() );
    }
}
