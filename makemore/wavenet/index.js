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

    const blockSize = 8;

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
    const iterations = 1000;

    const layers = [
        tf.layers.embedding( {
            inputShape: [ blockSize ],
            inputDim: totalChars,
            outputDim: embeddingDimension,
        } ),
        tf.layers.flatten(),
        tf.layers.dense( {
            units: neurons,
        } ),
        tf.layers.batchNormalization( {
            axis: 1,
        } ),
        tf.layers.activation( {
            activation: 'tanh',
        } ),
        tf.layers.dense( {
            units: totalChars,
        } ),
    ];

    const model = tf.sequential( { layers } );

    model.summary();

    function run( learningRate = 0.1 ) {
        const optimizer = tf.train.sgd( learningRate );
        const loss = () => tf.tidy( () => {
            const ix = tf.randomUniform( [ 32 ], 0, Xtr.shape[ 0 ], 'int32' );
            const Xbatch = Xtr.gather( ix );
            const Ybatch = Ytr.gather( ix );

            const logits = layers.reduce( ( input, layer ) => layer.apply( input ), Xbatch );
            return tf.losses.softmaxCrossEntropy( tf.oneHot( Ybatch, totalChars ), logits );
        } )

        for (let i = 0; i < iterations; i++) {
            optimizer.minimize( loss, true, model.trainableWeights.map( ( { val } ) => val ) ).print();
        }
        
        function splitLoss( X, Y ) {
            const loss = tf.tidy( () => {
                const logits = layers.reduce( ( input, layer ) => layer.apply( input ), X );
                return tf.losses.softmaxCrossEntropy( tf.oneHot( Y, totalChars ), logits );
            } );

            return loss.arraySync();
        }

        console.log( `Loss on training set after ${iterations} iterations with a ${learningRate} learning rate: ${splitLoss( Xtr, Ytr )}` );
        console.log( `Loss on development set after ${iterations} iterations with a ${learningRate} learning rate: ${splitLoss( Xdev, Ydev )}` );

        for (let i = 0; i < 5; i++) {
            const out = []  
            let context = Array( blockSize ).fill( 0 );

            while ( true ) {
                const X = tf.tensor2d( [ context ], null, 'int32' )
                const logits = layers.reduce( ( input, layer ) => layer.apply( input ), X );
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
