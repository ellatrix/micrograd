function matMul(A, B) {
    const [ m, n ] = A.shape;
    const [ p, q ] = B.shape;

    if ( n !== p ) {
        throw new Error( 'Matrix dimensions do not match.' );
    }

    const C = empty( [ m, q ] );

    for ( let m_ = m; m_--; ) {
        for ( let q_ = q; q_--; ) {
            let sum = 0;
            for ( let n_ = n; n_--; ) sum += A[m_ * n + n_] * B[n_ * q + q_];
            C[m_ * q + q_] = sum;
        }
    }

    return C;
}
