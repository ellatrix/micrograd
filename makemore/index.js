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
    return a.map( ( row, i ) => row.map( ( _, j ) => a[ i ].reduce( ( sum, _, k ) => sum.add( a[ i ][ k ].multiply( b[ k ][ j ] ) ), new Value( 0 ) ) ) );
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

    for ( const name of names.slice( 0, 1 ) ) {
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
    const logits = matrixDotProduct( xenc, W ); // log counts
    // Softmax.
    const counts = matrixExp( logits );
    const probs = counts.map( ( row ) => row.map( ( x ) => x.div( row.reduce( ( sum, x ) => sum.add( x ), new Value( 0 ) ) ) ) ); // normalized probabilities
    const relevantProbs = probs.map( ( row, i ) => row[ ys[ i ] ] );
    let loss = relevantProbs.map( ( x ) => x.log() );

    loss = loss.reduce( ( sum, x ) => sum.add( x ) ).div( loss.length );
    loss = loss.multiply( -1 );

    W.forEach( ( row ) => row.forEach( ( x ) => x.grad = 0 ) );

    loss.backward();

    drawDot( loss );

    console.log( loss );
});