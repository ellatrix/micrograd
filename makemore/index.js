function zeros( rows, cols ) {
    return Array.from( { length: rows }, () => Array.from( { length: cols }, () => 0 ) );
}

function oneHot( ns, length ) {
    return ns.map( ( n ) => Array.from( { length }, ( _, i ) => new Value( n === i ? 1 : 0 ) ) );
}

function random( rows, cols ) {
    return Array.from( { length: rows }, () => Array.from( { length: cols }, () => new Value( randomMinMax( -1, 1 ) ) ) );
}

function matrixDotProduct( a, b ) {
    const rows = a.length;
    const cols = b[ 0 ].length;
    return Array.from( { length: rows }, ( _, i ) => Array.from( { length: cols }, ( _, j ) => {
        const row = a[ i ];
        const col = b.map( ( r, k ) => r[ j ].mul( row[ k ] ) );
        return col.shift().add( ...col );
    } ) );
}

function matrixExp( a ) {
    return a.map( ( row ) => row.map( ( x ) => x.exp() ) );
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

    // Inputs.
    const xs = [];
    // Targets, or labels.
    const ys = [];

    for ( const name of names.slice( 0, 100 ) ) {
        const exploded = [ '.', ...Array.from( name ), '.' ];
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

    const xenc = oneHot( xs, totalChars );
    const W = random( 27, 27 );
    const iterations = 100;

    for (let i = 0; i < iterations; i++) {
        // Forward pass.
        const logits = matrixDotProduct( xenc, W ); // log counts
        // Softmax.
        const counts = matrixExp( logits );
        const probs = counts.map( ( row ) => {
            const rowSum = ( new Value( 0 ) ).add( ...row );
            return row.map( ( x ) => x.div( rowSum ) );
        } ); // normalized probabilities
        const relevantProbs = probs.map( ( row, j ) => row[ ys[ j ] ] );
        let loss = relevantProbs.map( ( x ) => x.log() );

        loss = loss.shift().add( ...loss );
        loss = loss.div( -relevantProbs.length );

        console.log(`Loss after iteration ${i}: ${loss.data}`);

        W.forEach( ( row ) => row.forEach( ( x ) => x.grad = 0 ) );

        loss.backward();

        W.forEach( ( row ) => row.forEach( ( x ) => x.data -= 50 * x.grad ) );
    }
});