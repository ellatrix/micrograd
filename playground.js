deconstructedTanH: {
    break deconstructedTanH;
    // Inputs x1, x2
    const x1 = new Value(2.0); x1.label = 'x1';
    const x2 = new Value(0.0); x2.label = 'x2';
    // Weights w1, w2
    const w1 = new Value(-3.0); w1.label = 'w1';
    const w2 = new Value(1.0); w2.label = 'w2';
    // Bias of the neuron
    const b = new Value(6.8813735870195432); b.label = 'b';
    // x1*w1 + x2*w2 + b
    x1w1 = x1.mul( w1 ); x1w1.label = 'x1*w1';
    x2w2 = x2.mul( w2 ); x2w2.label = 'x2*w2';
    x1w1x2w2 = x1w1.add( x2w2 ); x1w1x2w2.label = 'x1*w1 + x2*w2';
    n = x1w1x2w2.add( b ); n.label = 'n';

    e = n.mul(2).exp();
    o = e.add(-1).div(e.add(1));
    o.label = 'o';
    o.backward();
    drawDot(o);
}

{
    const mlp = new MLP( 3, [ 4, 4, 1 ] );
    const examples = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ];
    const ys = [1.0, -1.0, -1.0, 1.0];
    const iterations = 100;
    const ypred = examples.map( ( example ) => mlp.forward( example ) );
    const losses = ys.map( ( ygt, i ) => ypred[i].sub( ygt ).pow( 2 ) );
    const firstLoss = losses.shift();
    const totalLoss = firstLoss.add( ...losses );

    totalLoss.label = 'Total Loss';
    ypred.forEach( ( y ) => y.group = 'ypred' );

    for (let i = 0; i < iterations; i++) {
        totalLoss.forward();
        console.log(`Loss after iteration ${i}: ${totalLoss.data}`);
        totalLoss.backward();
        mlp.parameters().forEach( ( p ) => p.data -= 0.1 * p.grad );
    }

    drawDot( totalLoss );
}
