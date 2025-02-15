import { Graph } from 'https://esm.sh/graphlib';
import { instance } from 'https://esm.sh/@viz-js/viz';
import * as graphlibDot from 'https://esm.sh/graphlib-dot';
const ids = new WeakMap();

function createId(node) {
    let id = ids.get(node);
    if (!id) {
        id = createId.counter++;
        ids.set(node, id);
    }
    return id;
}

createId.counter = 0;

function trace( root ) {
	const nodes = new Set();
	const edges = new Set();

	function build( node ) {
		if ( ! nodes.has( node ) ) {
			nodes.add( node );

			for ( const child of node._prev ?? [] ) {
				edges.add( [ child, node ] );
				build( child );
			}
		}
	}

	build( root );

	return { nodes, edges };
}

export async function drawDot(root) {
	const { nodes, edges } = trace( root );
	const graph = new Graph( { compound: true } );
    graph.setGraph({ rankdir: "LR" });

	for ( const node of nodes ) {
		node._id = createId(node);
		graph.setNode( node._id, {
            shape: 'record',
			color: node.color ?? '',
            // `grad` will be important later.
            label: [
                node.label,
                [ 'data', node.data ].join(': '),
                node.grad !== undefined ? [ 'grad', node.grad ].join(': ') : null
            ].filter( Boolean ).join(' | ')
        } );

		if ( node._op ) {
			graph.setNode( node._id + node._op, { label: node._op } );
			graph.setEdge( node._id + node._op, node._id );
		}
	}

	for ( const [ node, child ] of edges ) {
		graph.setEdge( node._id, child._id + child._op );
	}

	const viz = await instance();
    const dotString = graphlibDot.write(graph);
    return viz.renderSVGElement(dotString);
}