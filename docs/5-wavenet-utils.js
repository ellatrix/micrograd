
import { random, transpose } from './1-bigram-utils.js';
import { matMul, FloatMatrix, createFloatMatrix, Value } from './3-0-makemore-MLP-utils.js';

Value.addOperation( 'matMulBiasBroadcast', async ( A, B, bias ) => {
    const K = A.shape.at(-1);
    const restDims = A.shape.slice(0, -1);
    const [k2, N] = B.shape;

    if (K !== k2) {
        throw new Error(`Shape mismatch: A.shape=[${A.shape}], B.shape=[${B.shape}]`);
    }

    const restSize = restDims.reduce((a, b) => a * b, 1);
    const flatA = new FloatMatrix(A).reshape( [restSize, K] );
    const result = new FloatMatrix(await matMul(flatA, B)).reshape( [...restDims, N] );

    if ( bias ) {
        if ( N !== bias.length ) {
            throw new Error('Bias vector dimension does not match the resulting matrix rows.');
        }

        // Add the biases to every row.
        for ( let m_ = restSize; m_--; ) {
            for ( let n_ = N; n_--; ) {
                result[ m_ * N + n_ ] += bias[ n_ ];
            }
        }
    }

    const out = [
        result,
        async ( grad ) => {
            const flatGrad = new FloatMatrix(grad).reshape( [restSize, N] );
            const flatGradA = await matMul(flatGrad, transpose(B));
            return new FloatMatrix(flatGradA).reshape( [...restDims, K] );
        },
        async ( grad ) => {
            const flatGrad = new FloatMatrix(grad).reshape( [restSize, N] );
            const flatGradB = await matMul(transpose(flatA), flatGrad);
            return new FloatMatrix(flatGradB).reshape( [K, N] );
        }
    ];

    if ( bias ) {
        out.push( ( grad ) => {
            const B = createFloatMatrix( [ N ] );
            for ( let m_ = restSize; m_--; ) {
                for ( let n_ = N; n_--; ) {
                    B[ n_ ] += grad[ m_ * N + n_ ];
                }
            }
            return B;
        } );
    }

    return out;
} );

// print( (await matMulBroadcast( new FloatMatrix( random, [ 4, 5, 80 ] ), new FloatMatrix( random, [ 80, 200 ] ) ) ).shape ) // [ 4, 5, 200 ];

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
        this.weight = new Value( createFloatMatrix( [ fan_in, fan_out ], () => random() / fan_in ** 0.5 ) );
        if ( bias ) {
            this.bias = new Value( createFloatMatrix( [ fan_out ], () => 0 ) );
        }
    }
    apply( X ) {
        return X.matMulBiasBroadcast( this.weight, this.bias );
    }
    params() {
        return this.bias ? [ this.weight, this.bias ] : [ this.weight ];
    }
}
