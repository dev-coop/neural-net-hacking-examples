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
  input = 0
  output = 0
  incoming = []
  outgoing = []
  // sigmoid
  activationFn = inputVal => 1 / (1 + Math.exp(-inputVal))

  activate(value) {
    this.input = value || _.sum(this.incoming, ({source, weight}) => {
        return source.output * weight
      })
    this.output = this.activationFn(this.input)
    return this.output
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

const net = new Network([3, 2, 1])

net.activate([1, 2, 3])

console.log('INPUT')
_.each(net.inputLayer.neurons, neuron => {
  console.log(`in ${neuron.input} out ${neuron.output}`)
})

console.log('HIDDEN')
_.each(net.hiddenLayers, layer => {
  _.each(layer.neurons, neuron => {
    console.log(`in ${neuron.input} out ${neuron.output}`)
  })
})

console.log('OUTPUT')
_.each(net.outputLayer.neurons, neuron => {
  console.log(`in ${neuron.input} out ${neuron.output}`)
})
