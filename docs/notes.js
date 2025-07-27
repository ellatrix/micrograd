class AttentionHeadOperation extends Operation {
    constructor( [ B, T, C ] ) {
        this.wei = this.createBuffer(B * T * T);
        this.output = this.createBuffer(B * T * C);
        this.dWei = this.createBuffer(B * T * T);
        this.dK = this.createBuffer(B * T * C);
        this.dQ = this.createBuffer(B * T * C);
        this.dV = this.createBuffer(B * T * C);
    }

    async forward( k, q, v ) {
        const [ B, T, C ] = k.shape;
        const scale = C ** -0.5;
        await batchMatMul(this.wei, q, k, false, true);
        await rowTrilSoftmax(this.wei, scale);
        await batchMatMul(this.output, this.wei, v);
    }

    async backward( dOut ) {
        const [ B, T, C ] = dOut.shape;
        const scale = C ** -0.5;
        for ( let b_ = B; b_--; ) {
            const weiBatch = wei.subarray(b_ * T * T, (b_ + 1) * T * T).reshape([ T, T ]);
    }
}

export class Head {
    constructor( nEmbed, headSize ) {
        this.K = new LinearBroadcast( nEmbed, headSize, false );
        this.Q = new LinearBroadcast( nEmbed, headSize, false );
        this.V = new LinearBroadcast( nEmbed, headSize, false );
        this.mask = new FloatMatrix( [ headSize, headSize ], ( i, j ) => i < j ? -Infinity : 0 );
    }
    apply( X ) {
        const k = this.K.apply( X );
        const q = this.Q.apply( X );
        const v = this.V.apply( X );
        // (B, T, C) @ ( (B, T, C)ᵀ -> (B, C, T) ) -> (B, T, T)
        return k.batchMatMul( q, false, true )
            .batchSoftmaxRowMasked( T )
            // (B, T, T) @ (B, T, C) -> (B, T, C)
            .batchMatMul( v );
    }
    params() {
        return [ ...this.K.params(), ...this.Q.params(), ...this.V.params() ];
    }
}