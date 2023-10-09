function makeTable( Weights, indexToCharMap ) {
    Weights = softmaxByRow( Weights );
    const [ m, n ] = Weights.shape;
    const table = document.getElementById( 'table' );
    table.innerHTML = '';

    for ( let m_ = m; m_--; ) {
        const row = document.createElement( 'tr' );
        for ( let n_ = n; n_--; ) {
            const cell = document.createElement( 'td' );
            cell.textContent = Weights[ m_ * n + n_ ].toFixed(2);
            cell.style.backgroundColor = `rgba( 0, 0, 0, ${ Weights[ m_ * n + n_ ] } )`;
            row.prepend( cell );
        }

        // Add row head
        const cell = document.createElement( 'td' );
        cell.textContent = indexToCharMap[ m_ ];
        row.prepend( cell );
        table.prepend( row );
    }

    const row = document.createElement( 'tr' );

    for ( let n_ = n; n_--; ) {
        const cell = document.createElement( 'td' );
        cell.textContent = indexToCharMap[ n_ ];
        row.prepend( cell );
    }

    const cell = document.createElement( 'td' );
    row.prepend( cell );
    table.prepend( row );
}