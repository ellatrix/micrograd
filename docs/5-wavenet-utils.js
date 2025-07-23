
import { random } from './1-bigram-utils.js';
import { Value, FloatMatrix } from './3-0-makemore-MLP-utils.js';

export class FlattenConsecutive {
    constructor( n ) {
        this.n = n;
    }
    apply( X ) {
        return X.reshape( ( [ b, t, c ] ) => {
            return t / this.n === 1 ? [ b, c * this.n ] : [ b, t / this.n, c * this.n ];
        });
    }
    params() {
        return [];
    }
}
export class LinearBroadcast {
    constructor( fan_in, fan_out, bias = true ) {
        this.weight = new Value( new FloatMatrix( () => random() / fan_in ** 0.5, [ fan_in, fan_out ] ) );
        if ( bias ) {
            this.bias = new Value( new FloatMatrix( () => 0, [ fan_out ] ) );
        }
    }
    apply( X ) {
        return X.matMulBiasBroadcast( this.weight, this.bias );
    }
    params() {
        return this.bias ? [ this.weight, this.bias ] : [ this.weight ];
    }
}
