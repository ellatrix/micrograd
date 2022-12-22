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
    const names = text.split('\n');
    const chars = [ ...new Set( [ ...names.join('') ] ) ].sort();
    const totalChars = chars.length + 1;
    const stringToCharMap = chars.reduce( ( map, char, index ) => {
        map[ char ] = index + 1;
        return map;
    }, {} );
    stringToCharMap[ '.' ] = 0;
    indexToCharMap = [ '.', ...chars ];

    // Inputs.
    let xs = [];
    // Targets, or labels.
    let ys = [];

    for ( const name of names ) {
        const exploded = '.' + name + '.';
        i = 1;
        while ( exploded[ i ] ) {
            const bigram = exploded[i - 1] + exploded[i];
            const indexOfChar1 = stringToCharMap[ exploded[ i - 1 ] ];
            const indexOfChar2 = stringToCharMap[ exploded[ i ] ];
            xs.push( indexOfChar1 );
            ys.push( indexOfChar2 );
            i++;
        }
    }

    xs = tf.tensor1d( xs, 'int32' );
    ys = tf.tensor1d( ys, 'int32' );

    let W = tf.randomNormal( [ 27, 27 ] );
    const xenc = tf.oneHot( xs, totalChars );
    const yenc = tf.oneHot( ys, totalChars );

    console.log( yenc.shape, xenc.matMul( W ).shape );
    
    const f = ( x ) => tf.losses.softmaxCrossEntropy( yenc, xenc.matMul( x ) );

    const iterations = 200;

    let lastLoss = Infinity;

    function run() {
        for (let i = 0; i < iterations; i++) {
            const loss = f( W ).arraySync();

            const action = loss <= lastLoss ? 'log' : 'error';

            console[action](`Loss after iteration ${i}: ${loss}`);

            lastLoss = loss;

            const grad = tf.grad( f )( W );

            W = W.sub( grad.mul( 50 ) );
        }

        for (let i = 0; i < 5; i++) {
            const out = []  
            let ix = 0;

            while ( true ) {
                const xenc = tf.oneHot( [ ix ], totalChars );
                const logits = xenc.matMul( W );
                const counts = logits.exp();
                const probs = counts.div( counts.sum( 1, true ) ).squeeze();
                // const probs = logits.softmax();
                ix = sample( probs.arraySync() );
                // ix = tf.multinomial( probs.squeeze(), 1 ).arraySync()[ 0 ];

                out.push( indexToCharMap[ ix ] );

                if ( ix === 0 ) {
                    break;
                }
            }

            console.log( out.join( '' ) );
        }
    }

    const button = document.createElement( 'button' );
    button.textContent = 'Run';
    button.onclick = run;

    document.body.appendChild( button );
});