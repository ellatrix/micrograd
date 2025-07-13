---
layout: default
title: '3.1. makemore: Learning Rate'
permalink: '/makemore-learning-rate'
---

## What is a good learning rate?

Let's try to find a good learning rate between 0.001 and 1, which are likely to
be at the low and high end respectively. We could try some between,
exponentially trying a higher learning rate.

<script>
function linspace(start, end, num) {
    const step = (end - start) / (num - 1);
    return Array.from({ length: num }, (_, i) => start + (step * i));
}
export const learningRateExponents = linspace( -3, 0, 1000 );
export const learningRates = learningRateExponents.map( ( exponent ) => Math.pow( 10, exponent ) );
export const graph = document.createElement( 'div' );
</script>

Important: we should reset the parameters!

<script>
resetParams();
export const losses = [];
export const loss = await lossFn( X, Y );
</script>

<script data-iterations="1000">
const indices = Int32Array.from( { length: hyperParameters.batchSize }, () => Math.random() * X.shape[ 0 ] );
indices.shape = [ indices.length ];
const Xbatch = gather( X, indices );
const Ybatch = gather( Y, indices );
const loss = await lossFn( Xbatch, Ybatch );
losses.push( loss.data );
await loss.backward();
for ( const param of params ) {
    for ( let i = param.data.length; i--; ) {
        param.data[ i ] -= learningRates[ losses.length ] * param.grad[ i ];
    }
}

await Plotly.react(graph, [{
    x: [...learningRateExponents],
    y: [...losses],
}], {
    title: 'Loss vs Learning Rate',
    xaxis: {
        title: 'Learning Rate Exponent',
    },
    yaxis: {
        title: 'Loss',

    },
    width: 500,
    height: 500
});
export default graph;
</script>

The learning rate of 0.1 looks good, where 0.001 looks way too low, and 1 makes
the loss explode.

To do: learning rate diagnostic end of part 3 video: https://www.youtube.com/watch?v=P6sfmUTpUmc.