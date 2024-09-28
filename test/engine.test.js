const tol = 1e-3;

function isEqualWithTol( a, b ) {
    return a.reduce( ( acc, x, i ) => {
        const diff = Math.abs( x - b[ i ] );
        return acc && diff < tol;
    }, true );
}

async function test_matrix_ops() {
    // const {drawDot} = await import( './drawdot.js' );

    [ 'matMul' ].forEach( async op => {
        const v1 = [ 1, 2, 3, 4 ];
        const v2 = [ 5, 6, 7, 8 ];
        const shape = [ 2, 2 ];
        const x = new Value( new FloatMatrix( v1, [...shape] ) );
        const y = new Value( new FloatMatrix( v2, [...shape] ) );
        const z = x.matMulBias( y, new FloatMatrix( [ 0, 0, 0, 0 ], [...shape] ) );
        await z.forward();
        await z.backward();
        const f = ( x, y ) => x[ op ]( y );
        const mgData = z.data;
        const mgGradX = x.grad;
        const mgGradY = y.grad;
        const tfData = f( tf.tensor2d( v1, shape ), tf.tensor2d( v2, shape ) ).arraySync().flatMap( ( v ) => v );
        const grads = tf.grads( f )( [ tf.tensor2d( v1, shape ), tf.tensor2d( v2, shape ) ] );
        const tfGradX = grads[ 0 ].arraySync().flatMap( ( v ) => v );
        const tfGradY = grads[ 1 ].arraySync().flatMap( ( v ) => v );
        // z.color = isEqualWithTol( mgData, tfData ) ? 'green' : 'red';
        // x.color = isEqualWithTol( mgGradX, tfGradX ) ? 'green' : 'red';
        // y.color = isEqualWithTol( mgGradY, tfGradY ) ? 'green' : 'red';
        // const graph = await drawDot(z);
        // document.body.appendChild(graph);
        console.assert( isEqualWithTol( mgData, tfData ) );
        console.assert( isEqualWithTol( mgGradX, tfGradX ) );
        console.assert( isEqualWithTol( mgGradY, tfGradY ) );
    } );

    [ 'softmaxCrossEntropy' ].forEach( async op => {
        const v1 = [ 1, 2, 3, 4 ];
        const v2 = [ 1, 0 ];
        const shape = [ 2, 2 ];
        const x = new Value( new FloatMatrix( v1, [...shape] ) );
        const y = new IntMatrix( v2, [ 2 ] );
        const z = x[ op ]( y );
        // console.log(z)
        await z.forward();
        await z.backward();
        const f = ( x ) => tf.losses.softmaxCrossEntropy( [[0,1],[1,0]], x );
        const mgData = z.data;
        const mgGradX = x.grad;
        const tfData = f( tf.tensor2d( v1, shape ) ).arraySync();
        const tfGradX = tf.grad( f )( tf.tensor2d( v1, shape ) ).arraySync().flatMap( ( v ) => v );
        // z.color = Math.abs( mgData - tfData ) < tol ? 'green' : 'red';
        // console.log(mgData, tfData)
        // x.color = isEqualWithTol( mgGradX, tfGradX ) ? 'green' : 'red';
        // console.log(mgGradX, tfGradX)
        // const graph = await drawDot(z);
        // document.body.appendChild(graph);
        console.assert( Math.abs( mgData - tfData ) < tol );
        console.assert( isEqualWithTol( mgGradX, tfGradX ) );
    } );

    [ 'softmaxCrossEntropy' ].forEach( async op => {
        const v1 = [ 1, 2, 3, 4 ];
        const v2 = [ 1, 0 ];
        const shape = [ 2, 2 ];
        const x = new Value( new FloatMatrix( v1, [...shape] ) );
        const y = new IntMatrix( v2, [ 2 ] );
        const z = x[ op ]( y );
        console.log(z)
        await z.forward();
        await z.backward();
        const f = ( x ) => tf.losses.softmaxCrossEntropy( [[0,1],[1,0]], x );
        const mgData = z.data;
        const mgGradX = x.grad;
        const tfData = f( tf.tensor2d( v1, shape ) ).arraySync();
        const tfGradX = tf.grad( f )( tf.tensor2d( v1, shape ) ).arraySync().flatMap( ( v ) => v );
        // z.color = Math.abs( mgData - tfData ) < tol ? 'green' : 'red';
        // console.log(mgData, tfData)
        // x.color = isEqualWithTol( mgGradX, tfGradX ) ? 'green' : 'red';
        // console.log(mgGradX, tfGradX)
        // const graph = await drawDot(z);
        // document.body.appendChild(graph);
        console.assert( Math.abs( mgData - tfData ) < tol );
        console.assert( isEqualWithTol( mgGradX, tfGradX ) );
    } );

    function batchNorm(x, gain, bias, epsilon = 1e-5) {
        const moments = tf.moments(x, 0, true);
        const mean = moments.mean;
        const variance = moments.variance;
        const normalized = tf.div(tf.sub(x, mean), tf.sqrt(tf.add(variance, epsilon)));
        return tf.add(tf.mul(normalized, gain), bias);
    }

    [ 'batchNorm' ].forEach( async op => {
        const v1 = [ 0.5, 0.5, 0.1, 0.9 ];
        const v2 = [ 0.1, 0.1 ];
        const v3 = [ 0.2, 0.2 ];
        const A = new Value( new FloatMatrix( v1, [ 2, 2 ] ) );
        const gain = new Value( new FloatMatrix( v2, [ 2 ] ) );
        const bias = new Value( new FloatMatrix( v3, [ 2 ] ) )
        const bnout = A[ op ]( gain, bias );
        console.log(bnout)
        await bnout.forward();
        await bnout.backward();
        const mgData = bnout.data;
        // const mgGradX = A.grad;
        const tfData = batchNorm( tf.tensor2d( v1, [2,2] ), tf.tensor1d( v3 ), tf.tensor1d( v2 ) ).arraySync().flatMap( ( v ) => v );
        // const grads = tf.grads( batchNorm )( [ tf.tensor2d( v1, [2,2] ), tf.tensor1d( v3 ), tf.tensor1d( v2 ) ] );
        // const tfGradX = grads[ 0 ].arraySync().flatMap( ( v ) => v );
        // const tfGradY = grads[ 1 ].arraySync().flatMap( ( v ) => v );
        // const tfGradZ = grads[ 2 ].arraySync().flatMap( ( v ) => v );
        console.log(mgData, tfData)
        // // z.color = Math.abs( mgData - tfData ) < tol ? 'green' : 'red';
        // console.log(mgData, tfData)
        // // x.color = isEqualWithTol( mgGradX, tfGradX ) ? 'green' : 'red';
        // console.log(mgGradX, tfGradX)
        // // const graph = await drawDot(z);
        // // document.body.appendChild(graph);
        console.assert( isEqualWithTol( mgData, tfData ) );
        // console.assert( isEqualWithTol( mgGradX, tfGradX ) );
    } );
}

test_matrix_ops();
