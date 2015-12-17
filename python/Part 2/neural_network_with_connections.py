import math

from itertools import izip_longest


class NeuronConnection(object):

    def __init__(self, neuron, weight=0.4):
        self.neuron = neuron
        self.weight = weight


class Neuron(object):

    def __init__(self, input=None):
        self.input = input
        self.output = None
        self.incoming = []
        self.outgoing = []

    def connect_child(self, child_neuron, weight=0.4):
        """Connect a child neuron to our output, then connect ourselves as input to child neuron"""
        self.outgoing.append(NeuronConnection(child_neuron, weight=weight))
        child_neuron.incoming.append(NeuronConnection(self, weight=weight))

    def activate(self, value=None):
        self.input = value or self.sum_inputs
        self.output = 1 / (1 + math.exp(-self.input))
        return self.output

    @property
    def sum_inputs(self):
        if self.input:
            # If we're an input
            return self.input
        else:
            # Otherwise we're a normal neuron
            sum = 0
            for connection in self.incoming:
                if connection.neuron.output is None:
                    connection.neuron.activate()
                sum += connection.neuron.output * connection.weight
            return sum


class Layer(object):
    def __init__(self, size=10):
        self.size = size
        self.neurons = [Neuron() for _ in range(size)]

    def activate(self, values=None):
        values = values or []
        for neuron, value in izip_longest(self.neurons, values):
            neuron.activate(value)

    def connect_with_layer(self, layer):
        [n1.connect_child(n2) for n2 in layer.neurons for n1 in self.neurons]


class Network(object):
    def __init__(self, sizes):
        # add all layers
        self.layers = [Layer(size=size) for size in sizes]
        for index, layer in enumerate(self.layers):
            if index + 1 < len(self.layers):
                layer.connect_with_layer(self.layers[index + 1])

    def activate(self, input_values):
        # activate input layer
        self.layers[0].activate(input_values)
        for layer in self.layers[1:]:
            layer.activate()


if __name__ == "__main__":
    network = Network([3, 2, 2])
    network.activate([1, 2, 3])

    print "INPUT"
    for neuron in network.layers[0].neurons:
        print "In:", neuron.input, "Out:", neuron.output

    print "HIDDEN"
    for layer in network.layers[1:-1]:
        for neuron in layer.neurons:
            print "In:", neuron.input, "Out:", neuron.output

    print "OUTPUT"
    for neuron in network.layers[-1].neurons:
        print "In:", neuron.input, "Out:", neuron.output
