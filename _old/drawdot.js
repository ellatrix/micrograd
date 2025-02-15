function createId() {
    return createId.counter++;
}

createId.counter = 0;

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
	const graph = new graphlib.Graph( { compound: true } );

	for ( const node of nodes ) {
		node._id = createId();
		graph.setNode( node._id, { color: node.color || '', shape: 'record', label: `${ node.label ? node.label + ' | ' : '' }data: ${ node.data } | grad: ${ node.grad }` } );

		if ( node._operation ) {
			graph.setNode( node._id + node._operation, { label: `${ node._operation }` } );
			graph.setEdge( node._id + node._operation, node._id );
		}

		if ( node.group ) {
			graph.setNode( 'cluster_' + node.group, { color: 'black', label: node.group } );
			graph.setParent( node._id, 'cluster_' + node.group );

			// if ( node._operation ) {
			// 	graph.setParent( node._id + node._operation, 'cluster_' + node.group );
			// }
		}
	}

	for ( const [ node, child ] of edges ) {
		graph.setEdge( node._id, child._id + child._operation );
	}

	const viz = new Viz();
	let written = graphlibDot.write( graph );
	written = written.slice( 0, -2 ) + 'rankdir="LR"}';
    viz.renderSVGElement( written )
        .then(function(element) {
        	document.body.appendChild(element);
        });
}