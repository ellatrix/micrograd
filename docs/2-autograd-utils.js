
export function getTopologicalOrder( node ) {
    const result = [];
    const visited = new Set();

    function visit( node ) {
        if ( visited.has( node ) || ! node._prev ) return;
        visited.add( node );
        for ( const child of node._prev ) visit( child );
        result.push( node );
    }

    visit( node );

    return result;
}
