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

async function main() {
    const response = await fetch('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
    const text = await response.text()
    console.log( 'Data loaded.' );

    const itos = [ ...new Set( [ ...text ] ) ].sort();
    const stoi = itos.reduce( ( map, char, index ) => {
        map[ char ] = index;
        return map;
    }, {} );

    function encode( text ) {
        return [ ...text ].map( ( char ) => stoi[ char ] );
    }

    function decode( indices ) {
        return indices.map( ( index ) => itos[ index ] ).join('');
    }

    const n = Math.floor( text.length * 0.8 );
    const Xtr = tf.tensor1d( encode( text.slice( 0, n ) ), 'int32' );
    const Xdev = tf.tensor1d( encode( text.slice( n ) ), 'int32' );

    const blockSize = 8;
    const batchSize = 32;

    function getBatch( split ) {
        const data = split === 'train' ? Xtr : Xdev;
        const ix = [ ...tf.randomUniform( [ batchSize ], 0, data.size - blockSize, 'int32' ).dataSync() ] ;
        const x = tf.stack( ix.map( ( i ) => data.slice( [ i ], [ blockSize ] ) ) );
        const y = tf.stack( ix.map( ( i ) => data.slice( [ i + 1 ], [ blockSize ] ) ) );
        return [ x, y ];
    }

    const layers = [
        tf.layers.embedding( {
            inputShape: [ blockSize ],
            inputDim: itos.length,
            outputDim: itos.length
        } ),
    ];

    const model = tf.sequential( { layers } );

    model.summary();

    const iterations = 5000;
    const evalInterval = 100;

    function run( learningRate = 0.1 ) {
        const optimizer = tf.train.adam( learningRate );
        const loss = () => tf.tidy( () => {
            const [ Xbatch, Ybatch ] = getBatch( 'train' );
            const logits = layers.reduce( ( input, layer ) => layer.apply( input ), Xbatch );
            return tf.losses.softmaxCrossEntropy( tf.oneHot( Ybatch, itos.length ), logits );
        } );

        function estimateLoss(split) {
            const evalIters = 200;
            let totalLoss = 0;

            for (let k = 0; k < evalIters; k++) {
                const [Xbatch, Ybatch] = getBatch(split);
                const logits = layers.reduce((input, layer) => layer.apply(input), Xbatch);
                const loss = tf.losses.softmaxCrossEntropy( tf.oneHot( Ybatch, itos.length ), logits );
                totalLoss += loss.dataSync()[0];
            }

            return totalLoss / evalIters;
        }

        for (let i = 0; i < iterations; i++) {
            optimizer.minimize( loss, true, model.trainableWeights.map( ( { val } ) => val ) );

            if (i % evalInterval === 0) {
                const trainLoss = estimateLoss("train");
                const valLoss = estimateLoss("val");
                console.log(`step ${i}: train loss ${trainLoss}, val loss ${valLoss}`);
            }
        }

        function generate( seed, length ) {
            let context = encode( seed );
            const out = [ ...context ];

            while ( out.length < length ) {
                const X = tf.tensor2d( [ context ], null, 'int32' );
                const logits = layers.reduce( ( input, layer ) => layer.apply( input ), X );
                const probs = logits.softmax().squeeze().arraySync();
                const ix = sample( probs[ probs.length - 1 ] );
                context = [ ...context.slice( 1 ), ix ];
                out.push( ix );
            }

            return decode( out );
        }

        console.log( generate( 'The list', 100 ) );
    }

    document.getElementById( 'run' ).addEventListener( 'click', () => {
        run( 0.01 );
    } );

    document.getElementById( 'run2' ).addEventListener( 'click', () => {
        run( 0.001 );
    } );
}

main();
