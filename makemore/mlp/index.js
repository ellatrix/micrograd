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

    const embeddingDimension = 3;
    const neurons = 200;
    const C = tf.variable( tf.randomNormal( [ totalChars, embeddingDimension ] ) );
    const W1 = tf.variable( tf.randomNormal( [ embeddingDimension * blockSize, neurons ] ) );
    const b1 = tf.variable( tf.randomNormal( [ neurons ] ) );
    const W2 = tf.variable( tf.randomNormal( [ neurons, totalChars ] ) );
    const b2 = tf.variable( tf.randomNormal( [ totalChars ] ) );
    const parameters = [ C, W1, b1, W2, b2 ];
    const numberOfParameters = parameters.reduce( ( sum, p ) => sum + p.size, 0 );

    console.log( `Number of parameters: ${numberOfParameters}`)

    const iterations = 1000;

    function run( learningRate = 0.1 ) {
        const optimizer = tf.train.sgd(learningRate);

        for (let i = 0; i < iterations; i++) {
            const loss = ( C, W1, b1, W2, b2 ) => tf.tidy( () => {
                const ix = tf.randomUniform( [ 64 ], 0, Xtr.shape[ 0 ], 'int32' );
                const Xbatch = Xtr.gather( ix );
                const Ybatch = Ytr.gather( ix );
                const emb = C.gather( Xbatch ).reshape( [ -1, embeddingDimension * blockSize ] );
                const h1 = emb.matMul( W1 ).add( b1 ).tanh();
                const logits = h1.matMul( W2 ).add( b2 );   
                return tf.losses.softmaxCrossEntropy( tf.oneHot( Ybatch, totalChars ), logits );
            } );

            optimizer.minimize( () => loss( ...parameters ) );

            // const grads = tf.grads( f )( parameters );

            // grads.forEach( ( grad, index ) => {
            //     parameters[ index ] = parameters[ index ].sub( grad.mul( learningRate ) );
            // } );
        }

        function lossAfterMiniBatch( C, W1, b1, W2, b2 ) {
            const emb = C.gather( Xdev ).reshape( [ -1, embeddingDimension * blockSize ] );
            const h1 = emb.matMul( W1 ).add( b1 ).tanh();
            const logits = h1.matMul( W2 ).add( b2 );   
            return tf.losses.softmaxCrossEntropy( tf.oneHot( Ydev, totalChars ), logits );
        }

        console.log( `Loss after ${iterations} iterations with a ${learningRate} learning rate: ${lossAfterMiniBatch( ...parameters ).arraySync()}` );

        for (let i = 0; i < 5; i++) {
            const out = []  
            let context = Array( blockSize ).fill( 0 );

            while ( true ) {
                const [ C, W1, b1, W2, b2 ] = parameters;
                const emb = C.gather( tf.tensor1d( context, 'int32' ) );
                const flattenedEmb = emb.reshape( [ -1, embeddingDimension * blockSize ] );
                const h1 = flattenedEmb.matMul( W1 ).add( b1 ).tanh();
                const logits = h1.matMul( W2 ).add( b2 );
                const probs = logits.softmax().squeeze();
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