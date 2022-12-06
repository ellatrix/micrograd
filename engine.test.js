function test_sanity_check() {
    const x = new Value( -4 );
    const z = x.multiply( 2 ).add( 2, x );
    const q = z.relu().add( z.multiply( x ) );
    const h = z.multiply( z ).relu();
    const y = h.add( q ).add( q.multiply( x ) );

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
