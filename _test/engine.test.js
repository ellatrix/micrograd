const table = document.createElement('table');

document.body.appendChild(table);

function t(value) {
    return tf.tensor(value.data, value.data.shape);
}

function deepFlatMap(value) {
    if (Array.isArray(value)) {
        return value.flatMap(deepFlatMap);
    }
    return [value];
}

function addRow(op, compare) {
    const row = document.createElement('tr');
    const opCell = document.createElement('td');
    opCell.textContent = op;
    row.appendChild(opCell);

    compare.forEach(([mgValues, tfValues]) => {
        if (Array.isArray(mgValues) && mgValues.length !== tfValues.size) {
            throw new Error('Not same size.');
        }
        if (Array.isArray(mgValues) && mgValues.shape.toString() !== tfValues.shape.toString()) {
            throw new Error('Not same shape.');
        }
        tfValues = tfValues.arraySync();
        let diff;
        if (mgValues.length) {
            tfValues = deepFlatMap(tfValues);
            diff = Math.max(...[...mgValues].map((v, i) => Math.abs(v - tfValues[i])));
        } else {
            diff = Math.abs(mgValues - tfValues);
        }
        const diffCell = document.createElement('td');
        if (diff === 0) {
            diffCell.textContent = '0';
        } else {
            const magnitude = Math.floor(Math.log10(diff));
            diffCell.textContent = `1e${magnitude}`;
        }
        row.appendChild(diffCell);
    });

    table.appendChild(row);
}

function random() {
    return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
}

async function test_matrix_ops() {
    {
        const op = 'matMulBiasBroadcast';
        const x = new Value( createFloatMatrix( [ 4, 5, 8 ], random ) );
        const y = new Value( createFloatMatrix( [ 8, 20 ], random ) );
        const z = await x.matMulBiasBroadcast( y );
        await z.forward();
        await z.backward();
        const f = ( x, y ) => x.matMul( y.expandDims(0).tile([4, 1, 1]) );
        const [ tfGradX, tfGradY ] = tf.grads( f )( [ t(x), t(y) ] );
        addRow( op, [
            [ z.data, f( t(x), t(y) ) ],
            [ x.grad, tfGradX ],
            [ y.grad, tfGradY ]
        ] );
    }

    {
        const op = 'matMulBiasBroadcast';
        const x = new Value( createFloatMatrix( [ 4, 5, 8 ], random ) );
        const y = new Value( createFloatMatrix( [ 8, 20 ], random ) );
        const b = new Value( createFloatMatrix( [ 20 ], random ) );
        const z = await x.matMulBiasBroadcast( y, b );
        await z.forward();
        await z.backward();
        const f = ( x, y, b ) => x.matMul( y.expandDims(0).tile([4, 1, 1]) ).add( b );
        const [ tfGradX, tfGradY, tfGradB ] = tf.grads( f )( [ t(x), t(y), t(b) ] );
        addRow( op, [
            [ z.data, f( t(x), t(y), t(b) ) ],
            [ x.grad, tfGradX ],
            [ y.grad, tfGradY ],
            [ b.grad, tfGradB ]
        ] );
    }

    {
        const op = 'matMul';
        const x = new Value( new FloatMatrix( [ 1, 2, 3, 4 ] ).reshape( [ 2, 2 ] ) );
        const y = new Value( new FloatMatrix( [ 5, 6, 7, 8 ] ).reshape( [ 2, 2 ] ) );
        const z = x.matMulBias( y, new FloatMatrix( [ 0, 0 ] ).reshape( [ 2 ] ) );
        await z.forward();
        await z.backward();
        const f = ( x, y ) => x[ op ]( y );
        const [ tfGradX, tfGradY ] = tf.grads( f )( [ t( x ), t( y ) ] );
        addRow( op, [
            [ z.data, f( t( x ), t( y ) ) ],
            [ x.grad, tfGradX ],
            [ y.grad, tfGradY ]
        ] );
    }

    {
        const op = 'matMulBias';
        const x = new Value( new FloatMatrix( [ 1, 2, 3, 4 ] ).reshape( [ 2, 2 ] ) );
        const y = new Value( new FloatMatrix( [ 5, 6, 7, 8 ] ).reshape( [ 2, 2 ] ) );
        const b = new Value( new FloatMatrix( [ 1, 1 ] ).reshape( [ 2 ] ) );
        const z = x.matMulBias( y, b );
        await z.forward();
        await z.backward();
        const f = ( x, y, b ) => x.matMul( y ).add( b );
        const [ tfGradX, tfGradY, tfGradB ] = tf.grads( f )( [ t( x ), t( y ), t( b ) ] );
        addRow( op, [
            [ z.data, f( t( x ), t( y ), t( b ) ) ],
            [ x.grad, tfGradX ],
            [ y.grad, tfGradY ],
            [ b.grad, tfGradB ]
        ] );
    }

    {
        const op = 'softmaxCrossEntropy';
        const x = new Value( new FloatMatrix( [ 1, 2, 3, 4 ] ).reshape( [ 2, 2 ] ) );
        const y = new IntMatrix( [ 1, 0 ] ).reshape( [ 2 ] );
        const z = x[ op ]( y );
        await z.forward();
        await z.backward();
        const f = ( x ) => tf.losses[ op ]( [[0,1],[1,0]], x );
        const tfGradX = tf.grad( f )( t( x ) );
        addRow( op, [
            [ z.data, f( t( x ) ) ],
            [ x.grad, tfGradX ]
        ] );
    }

    function batchNorm(x, gain, bias, epsilon = 1e-5) {
        const moments = tf.moments(x, 0);
        const mean = moments.mean;
        let correctedVariance = moments.variance;
        // const normalized = tf.div(tf.sub(x, mean), tf.sqrt(tf.add(variance, epsilon)));
        // return tf.add(tf.mul(normalized, gain), bias);
        // Apply Bessel's correction to variance
        const n = x.shape[0];
        correctedVariance = tf.mul(correctedVariance, n / (n - 0));
        return tf.batchNorm(
            x,
            mean,
            correctedVariance,
            bias,
            gain,
            epsilon
        );
    }

    function layerNorm(x, gain, bias, epsilon = 1e-5) {
        const moments = tf.moments(x, -1, true); // compute mean & variance across last dimension, keepDims=true
        const mean = moments.mean;
        const variance = moments.variance;
    
        const normalized = tf.div(tf.sub(x, mean), tf.sqrt(tf.add(variance, epsilon)));
        return tf.add(tf.mul(normalized, gain), bias);
    }

    {
        const op = 'batchNorm';
        const A = new Value( new FloatMatrix( [ 0.5, 0.5, 0.1, 0.9 ] ).reshape( [ 2, 2 ] ) );
        const gain = new Value( new FloatMatrix( [ 0.1, 0.1 ] ).reshape( [ 2 ] ) );
        const bias = new Value( new FloatMatrix( [ 0.2, 0.2 ] ).reshape( [ 2 ] ) )
        const bnout = A[ op ]( gain, bias );
        await bnout.forward();
        await bnout.backward();
        const [ tfGradX, tfGradY, tfGradZ ] = tf.grads( batchNorm )( [ t( A ), t( gain ), t( bias ) ] );
        addRow( op, [
            [ bnout.data, batchNorm( t( A ), t( gain ), t( bias ) ) ],
            [ A.grad, tfGradX ],
            [ gain.grad, tfGradY ],
            [ bias.grad, tfGradZ ]
        ] );
    }

    {
        const op = 'tanh';
        const A = new Value( new FloatMatrix( [ 0.5, 0.5, 0.1, 0.9 ] ).reshape( [ 2, 2 ] ) );
        const tanhout = A[ op ]();
        await tanhout.forward();
        await tanhout.backward();
        const f = ( x ) => x[ op ]();
        addRow( op, [
            [ tanhout.data, f( t( A ) ) ],
            [ A.grad, tf.grad( f )( t( A ) ) ]
        ] );
    }

    {
        const op = 'attentionHead';
        const k = new Value( createFloatMatrix( [ 4, 8, 16 ], random ) );
        const q = new Value( createFloatMatrix( [ 4, 8, 16 ], random ) );
        const v = new Value( createFloatMatrix( [ 4, 8, 16 ], random ) );
        const z = q
            // (B, T, C) @ ( (B, T, C)ᵀ -> (B, C, T) ) -> (B, T, T)
            .batchMatMul( k, false, true )
            .scale( 16 ** -0.5 )
            .batchSoftmaxRowTril()
            // (B, T, T) @ (B, T, C) -> (B, T, C)
            .batchMatMul( v );
        function f( k, q, v ) {
            const [ B, T, C ] = k.shape;
            let wei = tf.matMul(q, k, false, true);
            wei = tf.mul(wei, C ** -0.5);
            const mask = tf.linalg.bandPart(tf.ones([T, T]), -1, 0);
            const expandedMask = tf.expandDims(mask, 0);
            const broadcastedMask = tf.tile(expandedMask, [B, 1, 1]);
            const negInf = tf.fill([B, T, T], -Infinity);
            wei = tf.where(broadcastedMask.cast('bool'), wei, negInf);
            wei = tf.softmax(wei, -1);
            const out = tf.matMul(wei, v);
            return out;
        }
        await z.forward();
        await z.backward();
        const [ tfGradK, tfGradQ, tfGradV ] = tf.grads( f )( [ t(k), t(q), t(v) ] );
        addRow( op, [
            [ z.data, f( t(k), t(q), t(v) ) ],
            [ k.grad, tfGradK ],
            [ q.grad, tfGradQ ],
            [ v.grad, tfGradV ]
        ] );
    }

    {
        const op = 'batchMatMul';
        const k = new Value( createFloatMatrix( [ 4, 8, 16 ], random ) );
        const q = new Value( createFloatMatrix( [ 4, 16, 8 ], random ) );
        const z = k.batchMatMul( q );
        function f( k, q ) {
            return tf.matMul(k, q);
        }
        await z.forward();
        await z.backward();
        const [ tfGradK, tfGradQ ] = tf.grads( f )( [ t(k), t(q) ] );
        addRow( op, [
            [ z.data, f( t(k), t(q) ) ],
            [ k.grad, tfGradK ],
            [ q.grad, tfGradQ ]
        ] );
    }

    {
        const op = 'batchMatMulAT';
        const k = new Value( createFloatMatrix( [ 4, 8, 16 ], random ) );
        const q = new Value( createFloatMatrix( [ 4, 8, 16 ], random ) );
        const z = k.batchMatMul( q, true, false );
        function f( k, q ) {
            return tf.matMul(k, q, true, false);
        }
        await z.forward();
        await z.backward();
        const [ tfGradK, tfGradQ ] = tf.grads( f )( [ t(k), t(q) ] );
        addRow( op, [
            [ z.data, f( t(k), t(q) ) ],
            [ k.grad, tfGradK ],
            [ q.grad, tfGradQ ]
        ] );
    }

    {
        const op = 'batchMatMulBT';
        const k = new Value( createFloatMatrix( [ 4, 8, 16 ], random ) );
        const q = new Value( createFloatMatrix( [ 4, 8, 16 ], random ) );
        const z = k.batchMatMul( q, false, true );
        function f( k, q ) {
            return tf.matMul(k, q, false, true);
        }
        await z.forward();
        await z.backward();
        const [ tfGradK, tfGradQ ] = tf.grads( f )( [ t(k), t(q) ] );
        addRow( op, [
            [ z.data, f( t(k), t(q) ) ],
            [ k.grad, tfGradK ],
            [ q.grad, tfGradQ ]
        ] );
    }

    {
        const op = 'batchSoftmaxRowTril';
        const A = new Value( createFloatMatrix( [ 4, 8, 8 ], random ) );
        const z = A.batchSoftmaxRowTril();
        function f( A ) {
            const [ B, T, C ] = A.shape;
            const mask = tf.linalg.bandPart(tf.ones([T, T]), -1, 0);
            const expandedMask = tf.expandDims(mask, 0);
            const broadcastedMask = tf.tile(expandedMask, [B, 1, 1]);
            const negInf = tf.fill([B, T, T], -Infinity);
            const wei = tf.where(broadcastedMask.cast('bool'), A, negInf);
            return tf.softmax(wei, -1);
        }
        await z.forward();
        await z.backward();
        const [ tfGradA ] = tf.grads( f )( [ t(A) ] );
        console.log( t(z).arraySync(), f( t(A) ).arraySync());
        addRow( op, [
            [ z.data, f( t(A) ) ],
            [ A.grad, tfGradA ]
        ] );
    }

    {
        const op = 'concatLastDim';
        const k = new Value( createFloatMatrix( [ 4, 8, 2 ], random ) );
        const q = new Value( createFloatMatrix( [ 4, 8, 2 ], random ) );
        const v = new Value( createFloatMatrix( [ 4, 8, 2 ], random ) );
        const z = k.concatLastDim( q, v );
        function f( k, q, v ) {
            return tf.concat([k, q, v], -1);
        }
        await z.forward();
        await z.backward();
        const [ tfGradK, tfGradQ, tfGradV ] = tf.grads( f )( [ t(k), t(q), t(v) ] );
        addRow( op, [
            [ z.data, f( t(k), t(q), t(v) ) ],
            [ k.grad, tfGradK ],
            [ q.grad, tfGradQ ],
            [ v.grad, tfGradV ]
        ] );
    }

    {
        const op = 'add';
        const k = new Value( createFloatMatrix( [ 4, 8, 2 ], random ) );
        const q = new Value( createFloatMatrix( [ 4, 8, 2 ], random ) );
        const z = k.add( q );
        function f( k, q ) {
            return tf.add(k, q);
        }
        await z.forward();
        await z.backward();
        const [ tfGradK, tfGradQ ] = tf.grads( f )( [ t(k), t(q) ] );
        addRow( op, [
            [ z.data, f( t(k), t(q) ) ],
            [ k.grad, tfGradK ],
            [ q.grad, tfGradQ ]
        ] );
    }

    {
        const op = 'expandAndTile';
        const x = new Value( createFloatMatrix( [ 8, 2 ], random ) );
        const z = x.expandAndTile( 4 );
        function f( x ) {
            return tf.tile(x.expandDims(0), [4, 1, 1]);
        }
        await z.forward();
        await z.backward();
        const [ tfGradX ] = tf.grads( f )( [ t(x) ] );
        addRow( op, [
            [ z.data, f( t(x) ) ],
            [ x.grad, tfGradX ]
        ] );
    }

    {
        const op = 'layerNorm';
        const A = new Value( createFloatMatrix( [ 4, 8, 2 ], random ) );
        const gain = new Value( new FloatMatrix( [ 0.1, 0.1 ] ).reshape( [ 2 ] ) );
        const bias = new Value( new FloatMatrix( [ 0.2, 0.2 ] ).reshape( [ 2 ] ) )
        const bnout = A[ op ]( gain, bias );
        await bnout.forward();
        await bnout.backward();
        const [ tfGradX, tfGradY, tfGradZ ] = tf.grads( layerNorm )( [ t( A ), t( gain ), t( bias ) ] );
        addRow( op, [
            [ bnout.data, layerNorm( t( A ), t( gain ), t( bias ) ) ],
            [ A.grad, tfGradX ],
            [ gain.grad, tfGradY ],
            [ bias.grad, tfGradZ ]
        ] );
    }

    {
        const op = 'relu';
        const A = new Value( createFloatMatrix( [ 4, 8, 2 ], random ) );
        const z = A[ op ]();
        function f( A ) {
            return tf.relu(A);
        }
        await z.forward();
        await z.backward();
        const [ tfGradX ] = tf.grads( f )( [ t(A) ] );
        addRow( op, [
            [ z.data, f( t(A) ) ],
            [ A.grad, tfGradX ]
        ] );
    }

    {
        const op = 'gather';
        const indices = new IntMatrix( [ 0, 1, 2, 3, 4, 5 ] ).reshape( [ 2, 3 ] );
        const Embedding = new Value( createFloatMatrix( [ 27, 2 ], random ) );
        const z = Embedding.gather( indices );
        function f( Embedding ) {
            return tf.gather(Embedding, indices);
        }
        await z.forward();
        await z.backward();
        const [ tfGradX ] = tf.grads( f )( [ t(Embedding) ] );
        addRow( op, [
            [ z.data, f( t(Embedding) ) ],
            [ Embedding.grad, tfGradX ]
        ] );
    }
}

test_matrix_ops();
