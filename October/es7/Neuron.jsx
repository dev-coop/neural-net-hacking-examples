import _ from 'lodash';

///////////////////////////////////////////////////

class Connection {
  constructor(source, target) {
    this.source = source;
    this.target = target;
    this.weight = 0.5;
  }
}

///////////////////////////////////////////////////

class Neuron {
  input = 0;
  output = 0;
  incoming = [];
  outgoing = [];
  activationFn = inputVal => 1 / (1 + Math.exp(-inputVal));

  activate(value) {
    this.input = value || _.sum(this.incoming, connection => {
      return connection.source.output * connection.weight;
    });
    return this.output = this.activationFn(this.input);
  }

  connect(target) {
    const connection = new Connection(this, target);
    this.outgoing.push(connection);
    target.incoming.push(connection);
  }
}

///////////////////////////////////////////////////
// The Test

const neuronA = new Neuron();
const neuronB = new Neuron();

neuronA.connect(neuronB);

neuronA.activate(10);
console.log(neuronA.output);

neuronB.activate();
console.log(neuronB.output);
