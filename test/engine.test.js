const table = document.createElement('table');

document.body.appendChild(table);

function addRow(op, compare) {
    const row = document.createElement('tr');
    const opCell = document.createElement('td');
    opCell.textContent = op;
    row.appendChild(opCell);

    compare.forEach(([mgValues, tfValues]) => {
        tfValues = tfValues.arraySync();
        let diff;
        if (mgValues.length) {
            tfValues = tfValues.flatMap( ( v ) => v );
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

async function test_matrix_ops() {
    [ 'matMul' ].forEach( async op => {
        const v1 = [ 1, 2, 3, 4 ];
        const v2 = [ 5, 6, 7, 8 ];
        const shape = [ 2, 2 ];
        const x = new Value( new FloatMatrix( v1, [...shape] ) );
        const y = new Value( new FloatMatrix( v2, [...shape] ) );
        const z = x.matMulBias( y, new FloatMatrix( [ 0, 0 ], [ 2 ] ) );
        await z.forward();
        await z.backward();
        const f = ( x, y ) => x[ op ]( y );
        const tfArgs = [ tf.tensor2d( v1, shape ), tf.tensor2d( v2, shape ) ];
        const [ tfGradX, tfGradY ] = tf.grads( f )( tfArgs );
        addRow( 'matMul', [
            [ z.data, f( ...tfArgs ) ],
            [ x.grad, tfGradX ],
            [ y.grad, tfGradY ]
        ] );
    } );

    [ 'softmaxCrossEntropy' ].forEach( async op => {
        const v1 = [ 1, 2, 3, 4 ];
        const v2 = [ 1, 0 ];
        const shape = [ 2, 2 ];
        const x = new Value( new FloatMatrix( v1, [...shape] ) );
        const y = new IntMatrix( v2, [ 2 ] );
        const z = x[ op ]( y );
        await z.forward();
        await z.backward();
        const f = ( x ) => tf.losses.softmaxCrossEntropy( [[0,1],[1,0]], x );
        const tfArgs = [ tf.tensor2d( v1, shape ) ];
        const tfGradX = tf.grad( f )( ...tfArgs );
        addRow( 'softmaxCrossEntropy', [
            [ z.data, f( ...tfArgs ) ],
            [ x.grad, tfGradX ]
        ] );
    } );

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

    [ 'batchNorm' ].forEach( async op => {
        const v1 = [ 0.5, 0.5, 0.1, 0.9 ];
        const v2 = [ 0.1, 0.1 ];
        const v3 = [ 0.2, 0.2 ];
        const A = new Value( new FloatMatrix( v1, [ 2, 2 ] ) );
        const gain = new Value( new FloatMatrix( v2, [ 2 ] ) );
        const bias = new Value( new FloatMatrix( v3, [ 2 ] ) )
        const bnout = A[ op ]( gain, bias );
        await bnout.forward();
        await bnout.backward();
        const tfArgs = [ tf.tensor2d( v1, [2,2] ), tf.tensor1d( v2 ), tf.tensor1d( v3 ) ];
        const [ tfGradX, tfGradY, tfGradZ ] = tf.grads( batchNorm )( tfArgs );
        addRow( 'batchNorm', [
            [ bnout.data, batchNorm( ...tfArgs ) ],
            [ A.grad, tfGradX ],
            [ gain.grad, tfGradY ],
            [ bias.grad, tfGradZ ]
        ] );
    } );
}

test_matrix_ops();
