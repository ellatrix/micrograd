// Inputs x1, x2
const x1 = new Value(2.0); x1.label = 'x1';
const x2 = new Value(0.0); x2.label = 'x2';
// Weights w1, w2
const w1 = new Value(-3.0); w1.label = 'w1';
const w2 = new Value(1.0); w2.label = 'w2';
// Bias of the neuron
const b = new Value(6.8813735870195432); b.label = 'b';
// x1*w1 + x2*w2 + b
x1w1 = x1.multiply( w1 ); x1w1.label = 'x1*w1';
x2w2 = x2.multiply( w2 ); x2w2.label = 'x2*w2';
x1w1x2w2 = x1w1.add( x2w2 ); x1w1x2w2.label = 'x1*w1 + x2*w2';
n = x1w1x2w2.add( b ); n.label = 'n';

e = n.multiply(2).exp();
o = e.add(-1).div(e.add(1));
o.label = 'o';
o.backward();
drawDot(o);
