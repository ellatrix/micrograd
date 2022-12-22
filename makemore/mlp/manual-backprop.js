function sample(probs) {
    const sum = probs.reduce((a, b) => a + b, 0)
    if (sum <= 0) throw Error('probs must sum to a value greater than zero')
    const normalized = probs.map(prob => prob / sum)
    const sample = Math.random()
    let total = 0
    for (let i = 0; i < normalized.length; i++) {
        total += normalized[i]
        if (sample < total) return i
    }
}

fetch('https://raw.githubusercontent.com/karpathy/makemore/master/names.txt')
.then(res => res.text()).then(text => {
    console.log( 'Data loaded.' );
    console.log( 'Creating training data...' );

    const names = text.split('\n');
    const chars = [ ...new Set( [ ...names.join('') ] ) ].sort();
    const totalChars = chars.length + 1;
    const stringToCharMap = chars.reduce( ( map, char, index ) => {
        map[ char ] = index + 1;
        return map;
    }, {} );
    stringToCharMap[ '.' ] = 0;
    indexToCharMap = [ '.', ...chars ];

    const blockSize = 3;

    function buildDataSet( words ) {
        let X = [];
        let Y = [];

        for ( const name of words ) {
            const context = '.'.repeat( blockSize ) + name + '.';
            let i = blockSize;
            while ( context[ i ] ) {
                const x = context.slice( i - blockSize, i );
                const y = context[ i ];
                X.push( [ ...x ].map( ( char ) => stringToCharMap[ char ] ) );
                Y.push( stringToCharMap[ y ] );
                i++;
            }
        }   

        return [
            tf.tensor2d( X, [ X.length, blockSize ], 'int32' ),
            tf.tensor1d( Y, 'int32' )
        ];
    }

    tf.util.shuffle( names );

    const n1 = Math.floor( names.length * 0.8 );
    const n2 = Math.floor( names.length * 0.9 );

    const [ Xtr, Ytr ] = buildDataSet( names.slice( 0, n1 ) );
    const [ Xdev, Ydev ] = buildDataSet( names.slice( n1, n2 ) );
    const [ Xtest, Ytest ] = buildDataSet( names.slice( n2 ) );

    console.log( 'Training data created.' ) 

    const embeddingDimension = 10;
    const neurons = 200;
    const C = tf.variable( tf.randomNormal( [ totalChars, embeddingDimension ] ) );
    const W1 = tf.variable( tf.randomNormal( [ embeddingDimension * blockSize, neurons ] ).mul( (5/3)/((embeddingDimension * blockSize)**0.5) ) );
    const b1 = tf.variable( tf.randomNormal( [ neurons ] ).mul( 0.01 ) );
    const W2 = tf.variable( tf.randomNormal( [ neurons, totalChars ] ).mul( 0.01 ) );
    const b2 = tf.variable( tf.randomNormal( [ totalChars ] ).mul( 0.01 ) );
    const bngain = tf.variable( tf.ones( [ 1, neurons ] ) );
    const bnbias = tf.variable( tf.zeros( [ 1, neurons ] ) );
    const parameters = [ C, W1, b1, W2, b2, bngain, bnbias ];
    const numberOfParameters = parameters.reduce( ( sum, p ) => sum + p.size, 0 );

    console.log( `Number of parameters: ${numberOfParameters}`)

    const iterations = 1000;

    function run( learningRate = 0.1 ) {
        let CGradTest, W1GradTest, b1GradTest, W2GradTest, b2GradTest, bngainGradTest, bnbiasGradTest;
        const loss = ( C, W1, b1, W2, b2, bngain, bnbias ) => {
            const n = 32;
            const ix = tf.randomUniform( [ n ], 0, Xtr.shape[ 0 ], 'int32' );
            const Xbatch = Xtr.gather( ix );
            const Ybatch = Ytr.gather( ix );
            const emb = C.gather( Xbatch ).reshape( [ -1, embeddingDimension * blockSize ] );
            const hprebn = emb.matMul( W1 ).add( b1 );
            const bnmeani = hprebn.sum( 0, true ).div( n );
            const bndiff = hprebn.sub( bnmeani );
            const bndiff2 = bndiff.square();
            const bnvar = bndiff2.sum( 0, true ).div( n - 1 );
            const bnvarInv = bnvar.add( 1e-5 ).pow( -0.5 );
            const bnraw = bndiff.mul( bnvarInv );
            const hpreact = bnraw.mul( bngain ).add( bnbias );
            const h = hpreact.tanh();
            const logits = h.matMul( W2 ).add( b2 );
            // const logitMaxes = tf.max( logits, 1, true );
            // const normLogits = logits.sub( logitMaxes );
            // const counts = tf.exp( normLogits );
            // const countsSum = tf.sum( counts, 1, true );
            // const countsSumInv = countsSum.pow( -1 );
            // const probs = counts.mul( countsSumInv );
            // const logProbs = probs.log();
            const oneHotY = tf.oneHot( Ybatch, totalChars );
            // Pluck out the relevant log-probabilities
            // const l = logProbs.mul( oneHotY ).sum( 1 ).mean().mul( -1 );
            const l = tf.losses.softmaxCrossEntropy( oneHotY, logits );
            // dLogProbs = tf.zeros( logProbs.shape );
            // dLogProbs = dLogProbs.add( oneHotY.mul( -1/n ) );
            // dProbs = probs.pow( -1 ).mul( dLogProbs );
            // dCountsSumInv = counts.mul( dProbs ).sum( 1, true );
            // dCounts = countsSumInv.mul( dProbs );
            // dCountsSum = countsSum.mul( -1 ).pow( -2 ).mul( dCountsSumInv );
            // dCounts = dCounts.add( tf.ones( counts.shape ).mul( dCountsSum ) );
            // dNormLogits = counts.mul( dCounts );
            // dLogits = dNormLogits.clone();
            // dLogitMaxes = dNormLogits.mul( -1 ).sum( 1, true );
            // dLogits = dLogits.add( tf.oneHot( tf.argMax( logits, 1 ), logits.shape[ 1 ] ).mul( dLogitMaxes ) );
            dLogits = tf.softmax( logits, 1 ).sub( oneHotY ).div( n );
            dH = dLogits.matMul( W2.transpose() );
            db2 = dLogits.sum( 0 );
            dW2 = h.transpose().matMul( dLogits );
            W2GradTest = dW2.arraySync();

            return l;
        }

        for (let i = 0; i < iterations; i++) {
            const grads = tf.grads( loss )( parameters );
            // const forward = loss();

            const [ CGrad, W1Grad, b1Grad, W2Grad, b2Grad, bngainGrad, bnbiasGrad ] = grads;

            function isEqualWithTol( a, b ) {
                return a.reduce( ( acc1, row, i ) => {
                    return acc1 && row.reduce( ( acc2, x, j ) => {
                        const diff = Math.abs( x - b[ i ][ j ] );
                        return acc2 && diff < 0.000001;
                    }, true );
                }, true );
            }

            W2Grad.print();
            tf.tensor( W2GradTest ).print()

            console.log( 'dW', isEqualWithTol( W2Grad.arraySync(), W2GradTest ) );



            // parameters.forEach( ( p, index ) => {
            //     parameters[ index ] = p.sub( grads[ index ].mul( learningRate ) );
            // } );

            return;
        }

        function getMoments( C, W1, b1 ) {
            const emb = C.gather( Xdev ).reshape( [ -1, embeddingDimension * blockSize ] );
            let hpreact = emb.matMul( W1 ).add( b1 );
            return tf.moments( hpreact, 0, true );
        }

        const moments = getMoments( ...parameters );
        
        function splitLoss( X, Y ) {
            const loss = ( C, W1, b1, W2, b2, bngain, bnbias ) => tf.tidy( () => {
                const emb = C.gather( X ).reshape( [ -1, embeddingDimension * blockSize ] );
                let hpreact = emb.matMul( W1 ).add( b1 );
                hpreact = bngain.mul( hpreact.sub( moments.mean ).div( moments.variance.add( 1e-5 ).sqrt() ) ).add( bnbias );
                const h = hpreact.tanh();
                const logits = h.matMul( W2 ).add( b2 );   
                return tf.losses.softmaxCrossEntropy( tf.oneHot( Y, totalChars ), logits );
            } );

            return loss( ...parameters ).arraySync();
        }

        console.log( `Loss on training set after ${iterations} iterations with a ${learningRate} learning rate: ${splitLoss( Xtr, Ytr )}` );
        console.log( `Loss on development set after ${iterations} iterations with a ${learningRate} learning rate: ${splitLoss( Xdev, Ydev )}` );

        for (let i = 0; i < 5; i++) {
            const out = []  
            let context = Array( blockSize ).fill( 0 );

            while ( true ) {
                const getProbs = ( C, W1, b1, W2, b2, bngain, bnbias ) => tf.tidy( () => {
                    const emb = C.gather( tf.tensor1d( context, 'int32' ) );
                    const flattenedEmb = emb.reshape( [ -1, embeddingDimension * blockSize ] );
                    let hpreact = flattenedEmb.matMul( W1 ).add( b1 );
                    hpreact = bngain.mul( hpreact.sub( moments.mean ).div( moments.variance.add( 1e-5 ).sqrt() ) ).add( bnbias );
                    const h = hpreact.tanh();
                    const logits = h.matMul( W2 ).add( b2 );
                    return logits.softmax().squeeze();
                } );
                const probs = getProbs( ...parameters );
                const ix = sample( probs.arraySync() );
                context = [ ...context.slice( 1 ), ix ];
                out.push( indexToCharMap[ ix ] );

                if ( ix === 0 ) {
                    break;
                }
            }

            console.log( out.join( '' ) );
        }
    }

    document.getElementById( 'run' ).addEventListener( 'click', () => {
        run( 0.1 );
    } );

    document.getElementById( 'run2' ).addEventListener( 'click', () => {
        run( 0.01 );
    } );
});