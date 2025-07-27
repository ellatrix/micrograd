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
