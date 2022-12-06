function randomMinMax(min, max) {
    return Math.random() * (max - min) + min;
}

class Neuron {
  constructor( numberOfInputs ) {
    this.bias = new Value( randomMinMax( -1, 1 ) );
    this.bias.label = 'bias';
    this.weights = Array.from( { length: numberOfInputs }, () => new Value( randomMinMax( -1, 1 ) ) );
    const id = createId();
    this.weights.forEach( input => input.group = 'neuron_weights' + id );
  }
  forward( inputs ) {
      const weightedInputs = inputs.map( ( input, index ) => this.weights[ index ].mul( input ) );
      return this.bias.add( ...weightedInputs ).tanh();
  }
  parameters() {
    return this.weights.concat( this.bias );
  }
}

class Layer {
  constructor( numberOfInputs, numberOfNeurons ) {
    this.neurons = Array.from( { length: numberOfNeurons }, () => new Neuron( numberOfInputs ) );
  }
  forward( inputs ) {
      const outs = this.neurons.map( ( neuron ) => neuron.forward( inputs ) );
      return outs.length === 1 ? outs[ 0 ] : outs;
  }
  parameters() {
    return this.neurons.reduce( ( params, neuron ) => params.concat( neuron.parameters() ), [] );
  }
}

class MLP {
  constructor( numberOfInputs, numberOfOutputs ) {
    const size = [ numberOfInputs ].concat( numberOfOutputs );
    this.layers = numberOfOutputs.map( ( numberOfNeurons, index ) =>
        new Layer( size[ index ], numberOfNeurons )
    );
  }
  forward( inputs ) {
    const id = createId();
    inputs = inputs.map( ( input ) => new Value( input ) );
    inputs.forEach( input => input.group = 'input' + id );
    const out = this.layers.reduce( ( x, layer ) => layer.forward( x ), inputs );
    return out;
  }
  parameters() {
    return this.layers.reduce( ( params, layer ) => params.concat( layer.parameters() ), [] );
  }
  zeroGrad() {
    this.parameters().forEach((p) => p.grad = 0);
  }
}

function createId() {
  return createId.counter++;
}

createId.counter = 0;
