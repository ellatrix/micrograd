function matMul(A, B, bias) {
    const [ m, n ] = A.shape;
    const [ p, q ] = B.shape;

    if ( n !== p ) {
        throw new Error( 'Matrix dimensions do not match.' );
    }

    if ( bias && bias.length !== m ) {
        throw new Error('Bias vector dimension does not match the resulting matrix rows.');
    }

    const C = empty( [ m, q ] );

    for ( let m_ = m; m_--; ) {
        for ( let q_ = q; q_--; ) {
            let sum = 0;
            for ( let n_ = n; n_--; ) {
                sum += A[m_ * n + n_] * B[n_ * q + q_];
            }
            if ( bias ) sum += bias[ m_ ];
            C[m_ * q + q_] = sum;
        }
    }

    return C;
}
