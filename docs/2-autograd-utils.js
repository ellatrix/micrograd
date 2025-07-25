
export function getTopologicalOrder( startNode ) {
    const result = [];
    const visited = new Set();
    const visiting = new Set();

    function visit( node ) {
        if ( visited.has( node ) || ! node._prev ) return;
        if ( visiting.has( node ) ) {
            throw new Error("Cycle detected in computation graph.");
        }
        visiting.add( node );
        for ( const child of node._prev ) visit( child );
        visiting.delete( node );
        visited.add( node );
        result.push( node );
    }

    visit( startNode );

    return result;
}
