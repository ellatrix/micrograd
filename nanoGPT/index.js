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

    const itos = [ ...new Set( text ) ].sort();
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
    const nEmbed = 32;

    function getBatch( split ) {
        const data = split === 'train' ? Xtr : Xdev;
        const ix = [ ...tf.randomUniform( [ batchSize ], 0, data.size - blockSize, 'int32' ).dataSync() ] ;
        const x = tf.stack( ix.map( ( i ) => data.slice( [ i ], [ blockSize ] ) ) );
        const y = tf.stack( ix.map( ( i ) => data.slice( [ i + 1 ], [ blockSize ] ) ) );
        return [ x, y ];
    }

    const tokenEmbeddingTable = tf.layers.embedding( {
        inputShape: [ blockSize ],
        inputDim: itos.length,
        outputDim: nEmbed,
    } );

    const positionEmbeddingTable = tf.layers.embedding( {
        inputShape: [ blockSize ],
        inputDim: blockSize,
        outputDim: nEmbed,
    } );

    class Head {
        constructor(headSize, blockSize, dropout = 0.1) {
          this.key = tf.layers.dense({
            units: headSize,
            useBias: false
          });
          this.query = tf.layers.dense({
            units: headSize,
            useBias: false
          });
          this.value = tf.layers.dense({
            units: headSize,
            useBias: false
          });
          this.tril = tf.linalg.bandPart(tf.ones([blockSize, blockSize]), -1, 0);
          this.dropout = dropout;
        }
      
        apply(x) {
            const [ B, T, C ] = x.shape;
            const key = this.key.apply(x);
            const query = this.query.apply(x);
            const value = this.value.apply(x);
      
          let wei = tf.matMul(query, tf.transpose(key, [0, 2, 1]));
          wei = wei.mul(tf.sqrt(tf.scalar(parseFloat(key.shape[key.shape.length - 1], 10))).reciprocal());

      
          const mask = tf.sub(1, this.tril.slice([0, 0], [x.shape[1], x.shape[1]]));
          const infMask = mask.mul(tf.scalar(-1e9));
          wei = wei.add(infMask);
          wei = tf.softmax(wei, -1);
          wei = tf.dropout(wei, this.dropout);
      
          const out = tf.matMul(wei, value);
          return out;
        }
      }

    const saHead = new Head( nEmbed, blockSize );

    const lmHead = tf.layers.dense( {
        units: itos.length,
        activation: 'softmax',
    } );

    const model = tf.sequential( { layers: [
        tokenEmbeddingTable,
        positionEmbeddingTable,
        saHead.key,
        saHead.query,
        saHead.value,
        lmHead,
    ] } );

    model.summary();

    const iterations = 200;
    const evalInterval = 100;

    function run( learningRate = 0.1 ) {
        const optimizer = tf.train.adam( learningRate );
        const loss = () => tf.tidy( () => {
            const [ Xbatch, Ybatch ] = getBatch( 'train' );
            const tokEmb = tokenEmbeddingTable.apply( Xbatch );
            const postEmd = positionEmbeddingTable.apply( Xbatch );
            const x = tf.add( tokEmb, postEmd );
            const x2 = saHead.apply( x );
            const logits = lmHead.apply( x2 );
            return tf.losses.softmaxCrossEntropy( tf.oneHot( Ybatch, itos.length ), logits );
        } );

        function estimateLoss(split) {
            const evalIters = 200;
            let totalLoss = 0;

            for (let k = 0; k < evalIters; k++) {
                const [Xbatch, Ybatch] = getBatch(split);
                const tokEmb = tokenEmbeddingTable.apply( Xbatch );
                const postEmd = positionEmbeddingTable.apply( Xbatch );
                const x = tf.add( tokEmb, postEmd );
                const x2 = saHead.apply( x );
                const logits = lmHead.apply( x2 );
                const loss = tf.losses.softmaxCrossEntropy( tf.oneHot( Ybatch, itos.length ), logits );
                totalLoss += loss.dataSync()[0];
            }

            return totalLoss / evalIters;
        }

        for (let i = 0; i < iterations; i++) {
            optimizer.minimize( loss, true, model.trainableWeights.map( ( { val } ) => val ) )

            if (i && i % evalInterval === 0) {
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
                const tokEmb = tokenEmbeddingTable.apply( X );
                const logits = lmHead.apply( tokEmb );
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
