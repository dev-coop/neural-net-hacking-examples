import _ from 'lodash'

///////////////////////////////////////////////////

class Connection {
  constructor(source, target) {
    this.source = source
    this.target = target
    this.weight = 0.4 // Math.random()
  }
}

///////////////////////////////////////////////////

class Neuron {
  isBias = false
  input = 0
  output = 0
  error = null
  incoming = []
  outgoing = []
  learningRate = 0.3

  // sigmoid (range 0, +1)
  activationFn = inputVal => 1 / (1 + Math.exp(-inputVal))
  activationPrime = inputVal => {
    const val = 1 / (1 + Math.exp(-this.input))
    return val * (1 - val)
  }

  isOutput() {
    return _.isEmpty(this.outgoing)
  }

  activate(value) {
    if (this.isBias) return this.output = 1

    this.input = value || _.sum(this.incoming, ({source, weight}) => {
        return source.output * weight
      })
    this.output = this.activationFn(this.input)
    return this.output
  }

  train(targetOutput) {
    const inputDerivative = this.activationPrime(this.input)

    // set delta
    if (this.isOutput()) {
      this.error = targetOutput - this.output
      this.delta = -this.error * inputDerivative
    } else {
      this.delta = _.sum(this.outgoing, ({target, weight}) => {
        return inputDerivative * weight * target.delta
      })
    }

    // update weights
    _.each(this.outgoing, connection => {
      const gradient = this.output * connection.target.delta

      connection.weight -= gradient * this.learningRate
    })
  }

  connect(targetNeuron) {
    const connection = new Connection(this, targetNeuron)
    this.outgoing.push(connection)
    targetNeuron.incoming.push(connection)
  }
}

///////////////////////////////////////////////////

class Layer {
  constructor(size) {
    this.neurons = _.times(size, n => new Neuron())
  }

  activate(values) {
    _.each(this.neurons, (neuron, index) => {
      neuron.activate(values && values[index])
    })
  }

  connect(targetLayer) {
    if (!_.some(this.neurons, 'isBias')) {
      const bias = new Neuron()
      bias.isBias = true
      this.neurons.push(bias)
    }

    _.each(this.neurons, source => {
      _.each(targetLayer.neurons, target => {
        source.connect(target)
      })
    })
  }
}

///////////////////////////////////////////////////

class Network {
  constructor(sizes) {
    this.layers = _.map(sizes, size => new Layer(size))
    this.inputLayer = _.first(this.layers)
    this.outputLayer = _.last(this.layers)
    this.hiddenLayers = _.slice(this.layers, 1, this.layers.length - 1)

    _.each(this.layers, (layer, index) => {
      const nextIndex = ++index;
      const targetLayer = this.layers[nextIndex]
      targetLayer && layer.connect(targetLayer)
    })
  }

  activate(inputValues) {
    this.inputLayer.activate(inputValues)
    _.invoke(this.hiddenLayers, 'activate')
    this.outputLayer.activate()
  }
}

///////////////////////////////////////////////////
// The Test

const neuronA = new Neuron()
const neuronB = new Neuron()

neuronA.connect(neuronB)

const epochs = 9999
const trainingLogs = 100

_.times(epochs, (n) => {
  neuronA.activate(2)
  neuronB.activate()

  neuronB.train(1)
  neuronA.train()

  if (n === 0 || n % (epochs / trainingLogs) === 0) {
    console.log(`epoch: ${n} error: ${neuronB.error}`)
  }
})
