function trace( root ) {
	const nodes = new Set();
	const edges = new Set();

	function build( node ) {
		if ( ! nodes.has( node ) ) {
			nodes.add( node );

			for ( const child of node._prev ) {
				edges.add( [ child, node ] );
				build( child );
			}
		}
	}

	build( root );

	return { nodes, edges };
}

function drawDot(root) {
	const { nodes, edges } = trace( root );
	const graph = new graphlib.Graph( { rankdir: 'LR' } );

	graph.rankdir = 'LR';

	for ( const node of nodes ) {
		// const uid = Math.random().toString(36).substr(2, 9);
		graph.setNode( node.label, { shape: 'record', label: `${ node.label } | data: ${ node.data } | grad: ${ node.grad }` } );

		if ( node._operation ) {
			graph.setNode( node.label + node._operation, { label: `${ node._operation }` } );
			graph.setEdge( node.label + node._operation, node.label );
		}
	}

	for ( const [ node, child ] of edges ) {
		graph.setEdge( node.label, child.label + child._operation );
	}

	const viz = new Viz();
	let written = graphlibDot.write( graph );
	written = written.slice( 0, -2 ) + 'rankdir="LR"}';
    viz.renderSVGElement( written )
        .then(function(element) {
        	document.body.appendChild(element);
        });
}