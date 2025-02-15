
export class FloatMatrix extends Float32Array {
    constructor( data, shape = data?.shape || [] ) {
        const length = shape.reduce( ( a, b ) => a * b, 1 );

        if  ( typeof data === 'function' ) {
            data = Array.from( { length }, data );
        }

        super( data || length );

        if ( this.length !== length ) {
            throw new Error( 'Shape does not match data length.' );
        }

        this.shape = shape;
    }
}

export function random() {
    return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
}

export function oneHot( a, length ) {
    const B = new FloatMatrix( null, [ a.length, length ] );
    for ( let i = a.length; i--; ) B[ i * length + a[ i ] ] = 1;
    return B;
}

export function matMul(A, B) {
    const [ m, n ] = A.shape;
    const [ p, q ] = B.shape;
    const C = new FloatMatrix( null, [ m, q ] );

    if ( n !== p ) {
        throw new Error( 'Matrix dimensions do not match.' );
    }

    for ( let m_ = m; m_--; ) {
        for ( let q_ = q; q_--; ) {
            let sum = 0;
            for ( let n_ = n; n_--; ) {
                sum += A[m_ * n + n_] * B[n_ * q + q_];
            }
            C[m_ * q + q_] = sum;
        }
    }

    return C;
}

export function softmaxByRow( A ) {
    const [m, n] = A.shape;
    const B = new FloatMatrix( null, A.shape );
    for ( let m_ = m; m_--; ) {
        let max = -Infinity;
        for ( let n_ = n; n_--; ) {
            const value = A[m_ * n + n_];
            if (value > max) max = value;
        }
        let sum = 0;
        for ( let n_ = n; n_--; ) {
            const i = m_ * n + n_;
            // Subtract the max to avoid overflow
            sum += B[i] = Math.exp(A[i] - max);
        }
        for ( let n_ = n; n_--; ) {
            B[m_ * n + n_] /= sum;
        }
    }
    return B;
}

export function negativeLogLikelihood( probs, ys ) {
    const [m, n] = probs.shape;
    let sum = 0;
    for ( let m_ = m; m_--; ) {
        // Sum the logProbs (log likelihoods) of the correct label.
        sum += Math.log( probs[ m_ * n + ys[ m_ ] ] );
    }
    const mean = sum / m;
    // Mean negative log likelihood.
    return - mean;
}

export function softmaxCrossEntropyGradient( probs, ys ) {
    const [m, n] = probs.shape;
    const gradient = new FloatMatrix( probs );
    for ( let m_ = m; m_--; ) {
        // Subtract 1 for the gradient of the correct label.
        gradient[ m_ * n + ys[ m_ ] ] -= 1;
        for ( let n_ = n; n_--; ) {
            // Divide by the number of rows.
            gradient[ m_ * n + n_ ] /= m;
        }
    }
    return gradient;
}

export function transpose( A ) {
    const [ m, n ] = A.shape;
    const B = new FloatMatrix( null, [ n, m ] );

    for ( let m_ = m; m_--; ) {
        for ( let n_ = n; n_--; ) {
            B[n_ * m + m_] = A[m_ * n + n_];
        }
    }

    return B;
}

export function sample(probs) {
    const sample = Math.random();
    let total = 0;
    for ( let i = probs.length; i--; ) {
        total += probs[ i ];
        if ( sample < total ) return i;
    }
}
