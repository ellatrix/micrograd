function test_sanity_check() {
    const x = new Value( -4 );
    const z = x.mul( 2 ).add( 2, x );
    const q = z.relu().add( z.mul( x ) );
    const h = z.mul( z ).relu();
    const y = h.add( q ).add( q.mul( x ) );

    y.forward();
    y.backward();

    mgData = y.data;
    mgGrad = x.grad;

    const f = ( x ) => {
        const z = x.mul( 2 ).add( 2 ).add( x );
        const q = z.relu().add( z.mul( x ) );
        const h = z.mul( z ).relu();
        const y = h.add( q ).add( q.mul( x ) );
        return y;
    }

    tfData = f( tf.scalar( -4 ) ).arraySync();
    tfGrad = tf.grad( f )( tf.scalar( -4 ) ).arraySync();

    y.color = mgData === tfData ? 'green' : 'red';
    x.color = mgGrad === tfGrad ? 'green' : 'red';

    drawDot(y);

    console.assert( mgData === tfData );
    console.assert( mgGrad === tfGrad );
}

test_sanity_check();

const tol = 1e-3;

function test_more_ops() {
    const a = new Value(-4.0);
    const b = new Value(2.0);
    let c = a.add( b );
    let d = a.mul( b ).add( b.pow( 3 ) );
    c = c.add( c, 1 );
    c = c.add( 1, c, a.mul( -1 ) );
    d = d.add( d.mul( 2 ), b.add( a ).relu() )
    d = d.add( d.mul( 3 ), b.sub( a ).relu() )
    const e = c.sub( d );
    const f = e.pow( 2 );
    let g = f.div( 2 );
    g = g.add( new Value( 10 ).div( f ) );
    g.forward();
    g.backward();

    gMgData = g.data;
    aMgGrad = a.grad;
    bMgGrad = b.grad;

    const fun = ( a, b ) => {
        let c = a.add( b );
        let d = a.mul( b ).add( b.pow( 3 ) );
        c = c.add( c.add( 1 ) );
        c = c.add( c.add( 1 ).add( a.mul( -1 ) ) );
        d = d.add( d.mul( 2 ).add( b.add( a ).relu() ) )
        d = d.add( d.mul( 3 ).add( b.sub( a ).relu() ) )
        const e = c.sub( d );
        const f = e.pow( 2 );
        let g = f.div( 2 );
        g = g.add( tf.scalar( 10 ).div( f ) );
        return g;
    }

    gTfData = fun( tf.scalar( -4 ), tf.scalar( 2 ) ).arraySync();
    const grads = tf.grads( fun )( [ tf.scalar( -4 ), tf.scalar( 2 ) ] );
    aTfGrad = grads[ 0 ].arraySync();
    bTfGrad = grads[ 1 ].arraySync();

    g.color = gMgData - gTfData < tol ? 'green' : 'red';
    a.color = aMgGrad - aTfGrad < tol ? 'green' : 'red';
    b.color = bMgGrad - bTfGrad ? 'green' : 'red';

    drawDot(g);

    console.assert( gMgData - gTfData < tol );
    console.assert( aMgGrad - aTfGrad < tol );
    console.assert( bMgGrad - bTfGrad < tol );
}

test_more_ops();

function test_each_op() {
    [ 'relu', 'log', 'tanh' ].forEach( op => {
        const v = 4
        const x = new Value( v );
        const y = x[ op ]();
        y.forward();
        y.backward();
        const f = ( x ) => x[ op ]();
        const mgData = y.data;
        const mgGrad = x.grad;
        const tfData = f( tf.scalar( v ) ).arraySync();
        const tfGrad = tf.grad( f )( tf.scalar( v ) ).arraySync();
        y.color = mgData - tfData < tol ? 'green' : 'red';
        x.color = mgGrad - tfGrad < tol ? 'green' : 'red';
        drawDot(y);
        console.assert( mgData - tfData < tol );
        console.assert( mgGrad - tfGrad < tol );
    } );

    [ 'add', 'sub', 'mul', 'div' ].forEach( op => {
        const v1 = -4;
        const v2 = 2;
        const x = new Value( v1 );
        const y = new Value( v2 );
        const z = x[ op ]( y );
        z.forward();
        z.backward();
        const f = ( x, y ) => x[ op ]( y );
        const mgData = z.data;
        const mgGradX = x.grad;
        const mgGradY = y.grad;
        const tfData = f( tf.scalar( v1 ), tf.scalar( v2 ) ).arraySync();
        const grads = tf.grads( f )( [ tf.scalar( v1 ), tf.scalar( v2 ) ] );
        const tfGradX = grads[ 0 ].arraySync();
        const tfGradY = grads[ 1 ].arraySync();
        z.color = mgData === tfData ? 'green' : 'red';
        x.color = mgGradX === tfGradX ? 'green' : 'red';
        y.color = mgGradY === tfGradY ? 'green' : 'red';
        drawDot(z);
        console.assert( mgData === tfData );
        console.assert( mgGradX === tfGradX );
        console.assert( mgGradY === tfGradY );
    } );

    [ 'pow' ].forEach( op => {
        const v1 = -4;
        const v2 = 2;
        const x = new Value( v1 );
        const y = x[ op ]( v2 );
        y.forward();
        y.backward();
        const f = ( x ) => x[ op ]( v2 );
        const mgData = y.data;
        const mgGrad = x.grad;
        const tfData = f( tf.scalar( v1 ) ).arraySync();
        const tfGrad = tf.grad( f )( tf.scalar( v1 ) ).arraySync();
        y.color = mgData === tfData ? 'green' : 'red';
        x.color = mgGrad === tfGrad ? 'green' : 'red';
        drawDot(y);
        console.assert( mgData === tfData );
        console.assert( mgGrad === tfGrad );
    } );

    [ 'add' ].forEach( op => {
        const vs = [ -4, 2, 3 ];
        const xs = vs.map( v => new Value( v ) );
        const z = xs.shift()[ op ]( ...xs );
        z.forward();
        z.backward();
        const f = ( ...a ) => tf[ op + 'N' ]( a );
        const mgData = z.data;
        const mgGrads = xs.map( x => x.grad );
        const tfData = f( ...vs.map( x => tf.scalar( x ) ) ).arraySync();
        const grads = tf.grads( f )( vs.map( x => tf.scalar( x ) ) );
        const tfGrads = grads.map( x => x.arraySync() );
        z.color = mgData === tfData ? 'green' : 'red';
        xs.forEach( ( x, i ) => x.color = mgGrads[ i ] === tfGrads[ i ] ? 'green' : 'red' );
        drawDot(z);
        console.assert( mgData === tfData );
        xs.forEach( ( x, i ) => console.assert( mgGrads[ i ] === tfGrads[ i ] ) );
    } );
}

test_each_op();

function isEqualWithTol( a, b ) {
    a = a.tolist ? a.tolist() : a;
    b = b.tolist ? b.tolist() : b;
    return a.reduce( ( acc1, row, i ) => {
        return acc1 && row.reduce( ( acc2, x, j ) => {
            const diff = Math.abs( x - b[ i ][ j ] );
            return acc2 && diff < tol;
        }, true );
    }, true );
}

function test_matrix_ops() {
    [ 'add', 'mul', 'sub', 'div', 'matMul' ].forEach( op => {
        const v1 = [ [ 1, 2 ], [ 3, 4 ] ];
        const v2 = [ [ 5, 6 ], [ 7, 8 ] ];
        const x = new Matrix( nj.array( v1 ) );
        const y = new Matrix( nj.array( v2 ) );
        const z = x[ op ]( y );
        z.forward();
        z.backward();
        const f = ( x, y ) => x[ op ]( y );
        const mgData = z.data;
        const mgGradX = x.grad;
        const mgGradY = y.grad;
        const tfData = f( tf.tensor2d( v1 ), tf.tensor2d( v2 ) ).arraySync();
        const grads = tf.grads( f )( [ tf.tensor2d( v1 ), tf.tensor2d( v2 ) ] );
        const tfGradX = grads[ 0 ].arraySync();
        const tfGradY = grads[ 1 ].arraySync();
        z.color = isEqualWithTol( mgData, tfData ) ? 'green' : 'red';
        x.color = isEqualWithTol( mgGradX, tfGradX ) ? 'green' : 'red';
        y.color = isEqualWithTol( mgGradY, tfGradY ) ? 'green' : 'red';
        drawDot(z);
        console.assert( isEqualWithTol( mgData, tfData ) );
        console.assert( isEqualWithTol( mgGradX, tfGradX ) );
        console.assert( isEqualWithTol( mgGradY, tfGradY ) );
    } );

    [ 'pow' ].forEach( op => {
        const v1 = [ [ 1, 2 ], [ 3, 4 ] ];
        const v2 = 2;
        const x = new Matrix( nj.array( v1 ) );
        const z = x[ op ]( v2 );
        z.forward();
        z.backward();
        const f = ( x ) => x[ op ]( v2 );
        const mgData = z.data;
        const mgGradX = x.grad;
        const tfData = f( tf.tensor2d( v1 ) ).arraySync().map( x => x.map( y => Math.round( y ) ) );
        const tfGradX = tf.grad( f )( tf.tensor2d( v1 ) ).arraySync().map( x => x.map( y => Math.round( y ) ) );
        z.color = nj.equal( mgData, tfData ) ? 'green' : 'red';
        x.color = nj.equal( mgGradX, tfGradX ) ? 'green' : 'red';
        drawDot(z);
        console.assert( nj.equal( mgData, tfData ) );
        console.assert( nj.equal( mgGradX, tfGradX ) );
    } );

    [ 'exp', 'log' ].forEach( op => {
        const v1 = [ [ 1, 2 ], [ 3, 4 ] ];
        const x = new Matrix( nj.array( v1 ) );
        const z = x[ op ]();
        z.forward();
        z.backward();
        const f = ( x ) => x[ op ]();
        const mgData = z.data;
        const mgGradX = x.grad;
        const tfData = f( tf.tensor2d( v1 ) ).arraySync();
        const tfGradX = tf.grad( f )( tf.tensor2d( v1 ) ).arraySync();
        z.color = isEqualWithTol( mgData, tfData ) ? 'green' : 'red';
        x.color = isEqualWithTol( mgGradX, tfGradX ) ? 'green' : 'red';
        drawDot(z);
        console.assert( isEqualWithTol( mgData, tfData ) );
        console.assert( isEqualWithTol( mgGradX, tfGradX ) );
    } );

    [ 'mean' ].forEach( op => {
        const v1 = [ 1, 2, 3, 4 ];
        const x = new Matrix( nj.array( v1 ) );
        const z = x[ op ]();
        z.forward();
        z.backward();
        const f = ( x ) => x[ op ]();
        const mgData = z.data;
        const mgGradX = x.grad;
        const tfData = f( tf.tensor1d( v1 ) ).arraySync();
        const tfGradX = tf.grad( f )( tf.tensor1d( v1 ) ).arraySync();
        z.color = mgData.tolist()[ 0 ] === tfData ? 'green' : 'red';
        x.color = nj.equal( mgGradX, tfGradX ) ? 'green' : 'red';
        drawDot(z);
        console.assert( mgData.tolist()[ 0 ] === tfData );
        console.assert( nj.equal( mgGradX, tfGradX ) );
    } );

    [ 'mean' ].forEach( op => {
        const v1 = [ [ 1, 2 ], [ 3, 4 ] ];
        const x = new Matrix( nj.array( v1 ) );
        const z = x[ op ]();
        z.forward();
        z.backward();
        const f = ( x ) => x[ op ]();
        const mgData = z.data;
        const mgGradX = x.grad;
        const tfData = f( tf.tensor2d( v1 ) ).arraySync();
        const tfGradX = tf.grad( f )( tf.tensor2d( v1 ) ).arraySync();
        z.color = mgData.tolist()[ 0 ] === tfData ? 'green' : 'red';
        x.color = nj.equal( mgGradX, tfGradX ) ? 'green' : 'red';
        drawDot(z);
        console.assert( mgData.tolist()[ 0 ] === tfData );
        console.assert( nj.equal( mgGradX, tfGradX ) );
    } );

    [ 'divSumRows' ].forEach( op => {
        const v1 = [ [ 1, 2 ], [ 3, 4 ] ];
        const x = new Matrix( nj.array( v1 ) );
        const z = x[ op ]();
        z.forward();
        z.backward();
        const f = ( x ) => x.div( x.sum( 1, true ) );
        const mgData = z.data;
        const mgGradX = x.grad;
        const tfData = f( tf.tensor2d( v1 ) ).arraySync();
        const tfGradX = tf.grad( f )( tf.tensor2d( v1 ) ).arraySync();
        z.color = mgData.tolist()[ 0 ] === tfData ? 'green' : 'red';
        x.color = nj.equal( mgGradX, tfGradX ) ? 'green' : 'red';
        drawDot(z);
        console.assert( isEqualWithTol( mgData, tfData ) );
        console.log( tfGradX );
        console.assert( nj.equal( mgGradX, tfGradX ) );
    } );

    [ 'sum' ].forEach( op => {
        const v1 = [ 1, 2, 3, 4 ];
        const x = new Matrix( nj.array( v1 ) );
        const z = x[ op ]();
        z.forward();
        z.backward();
        const f = ( x ) => x[ op ]();
        const mgData = z.data;
        const mgGradX = x.grad;
        const tfData = f( tf.tensor1d( v1 ) ).arraySync();
        const tfGradX = tf.grad( f )( tf.tensor1d( v1 ) ).arraySync();
        z.color = mgData.tolist()[ 0 ] === tfData ? 'green' : 'red';
        x.color = nj.equal( mgGradX, tfGradX ) ? 'green' : 'red';
        drawDot(z);
        console.assert( mgData.tolist()[ 0 ] === tfData );
        console.assert( nj.equal( mgGradX, tfGradX ) );
    } );

    [ 'sum' ].forEach( op => {
        const v1 = [ [ 1, 2 ], [ 3, 4 ] ];
        const x = new Matrix( nj.array( v1 ) );
        const z = x[ op ]( 1 );
        z.forward();
        z.backward();
        const f = ( x ) => x[ op ]( 1, true );
        const mgData = z.data;
        const mgGradX = x.grad;
        const tfData = f( tf.tensor2d( v1 ) ).arraySync();
        const tfGradX = tf.grad( f )( tf.tensor2d( v1 ) ).arraySync();
        z.color = nj.equal( mgData, tfData ) ? 'green' : 'red';
        x.color = nj.equal( mgGradX, tfGradX ) ? 'green' : 'red';
        drawDot(z);
        console.assert( nj.equal( mgData, tfData ) );
        console.assert( nj.equal( mgGradX, tfGradX ) );
    } );
}

test_matrix_ops();
