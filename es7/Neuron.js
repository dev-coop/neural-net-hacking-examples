import _ from 'lodash'

///////////////////////////////////////////////////
// Data

const DATA = {
  ORGate: [
    {input: [0, 0], output: [0]},
    {input: [0, 1], output: [1]},
    {input: [1, 0], output: [1]},
    {input: [1, 1], output: [1]},
  ]
}

///////////////////////////////////////////////////
// Connection

class Connection {
  constructor(source, target) {
    this.source = source
    this.target = target
    this.weight = 0.4 // Math.random()
  }
}

///////////////////////////////////////////////////
// Neuron

class Neuron {
  isBias = false
  input = 0
  output = 0
  delta = 0
  incoming = []
  outgoing = []
  learningRate = 0.3

  // sigmoid (range 0, +1)
  activationFn = inputVal => 1 / (1 + Math.exp(-inputVal))

  isOutput() {
    return _.isEmpty(this.outgoing)
  }

  isInput() {
    return _.isEmpty(this.incoming)
  }

  activate(value) {
    if (this.isBias) return this.output = 1

    this.input = value || _.sum(this.incoming, ({source, weight}) => source.output * weight)
    return this.output = this.activationFn(this.input)
  }

  train(targetOutput) {
    if (!this.isBias && !this.isInput()) {
      if (this.isOutput()) {
        // this is the derivative of the error function, not simply the difference in output
        // http://whiteboard.ping.se/MachineLearning/BackProp
        this.delta = this.output - targetOutput
      } else {
        this.delta = _.sum(this.outgoing, ({target, weight}) => weight * target.delta)
      }
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
// Layer

class Layer {
  constructor(size) {
    this.neurons = _.times(size, n => new Neuron())
  }

  activate(inputValues = []) {
    return _.map(this.neurons, (neuron, index) => neuron.activate(inputValues[index]))
  }

  train(targetOutputs = []) {
    _.each(this.neurons, (neuron, index) => neuron.train(targetOutputs[index]))
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
// Network

class Network {
  constructor(sizes) {
    this.layers = _.map(sizes, size => new Layer(size))
    this.inputLayer = _.first(this.layers)
    this.outputLayer = _.last(this.layers)
    this.hiddenLayers = _.slice(this.layers, 1, this.layers.length - 1)
    this.error = 0

    this.errorFn = (targetOutputs) => {
      return _.sum(this.outputLayer.neurons, (neuron, i) => {
          return 0.5 * Math.pow(targetOutputs[i] - neuron.output, 2)
        }) / this.outputLayer.neurons.length
    }

    _.each(this.layers, (layer, index) => {
      const nextIndex = ++index;
      const targetLayer = this.layers[nextIndex]
      targetLayer && layer.connect(targetLayer)
    })
  }

  activate(inputValues) {
    this.inputLayer.activate(inputValues)
    _.invoke(this.hiddenLayers, 'activate')
    return this.outputLayer.activate()
  }

  train(targetOutputs) {
    this.outputLayer.train(targetOutputs)

    // set the new network error after training
    this.error = this.errorFn(targetOutputs)

    // hidden layers in reverse
    for (let i = this.hiddenLayers.length; i > 1; i--) {
      this.hiddenLayers[i].train()
    }

    this.inputLayer.train()
  }
}

///////////////////////////////////////////////////
// Trainer

class Trainer {
  constructor(network, data) {
    this.network = network
    this.data = data
  }

  train(options) {
    const {epochs, logFreq} = options
    const {network, data} = this

    _.times(epochs, epoch => {
      const avgError = _.sum(data, sample => {
        network.activate(sample.input)
        network.train(sample.output)
        return network.error / data.length
      })

      if (epoch % logFreq === 0 || epoch + 1 === epochs) {
        console.log(`epoch: ${epoch} error: ${avgError}`)
      }
    })
  }
}

///////////////////////////////////////////////////
// The Test

const network = new Network([2, 1])
const trainer = new Trainer(network, DATA.ORGate)

trainer.train({epochs: 10000, logFreq: 1000})
