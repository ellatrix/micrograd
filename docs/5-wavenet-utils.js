
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
    // Reshape a shallow subarray, not the original!
    const flatA = A.subarray().reshape([restSize, K]);
    const result = (await matMul(flatA, B)).reshape([...restDims, N]);

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

    return [
        result,
        async ( grad ) => {
            // Reshape a shallow subarray, not the original!
            const flatGrad = grad.subarray().reshape([restSize, N]);
            const flatA = A.subarray().reshape([restSize, K]);
            const flatGradAPromise = matMul(flatGrad, transpose(B));
            const flatGradBPromise = matMul(transpose(flatA), flatGrad);
            const out = [
                (await flatGradAPromise).reshape([...restDims, K]),
                (await flatGradBPromise).reshape([K, N])
            ];
            if ( bias ) {
                const biasGrad = createFloatMatrix( [ N ] );
                for ( let m_ = restSize; m_--; ) {
                    for ( let n_ = N; n_--; ) {
                        biasGrad[ n_ ] += grad[ m_ * N + n_ ];
                    }
                }
                out.push( biasGrad );
            }
            return out;
        },
    ];
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
    #params = [];
    constructor( fan_in, fan_out, bias = true ) {
        this.#params = [
            new Value( createFloatMatrix( [ fan_in, fan_out ], () => random() / fan_in ** 0.5 ) )
        ];
        if ( bias ) { 
            this.#params.push( new Value( createFloatMatrix( [ fan_out ] ) ) );
        }
    }
    apply( X ) {
        return X.matMulBiasBroadcast( ...this.#params );
    }
    params() {
        return this.#params;
    }
}
