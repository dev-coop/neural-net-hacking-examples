import math

from itertools import izip_longest


class NeuronConnection(object):

    def __init__(self, parent, child, weight=0.4):
        self.parent = parent
        self.child = child
        self.weight = weight


class Neuron(object):

    def __init__(self, input=None, output=None, is_bias=False):
        self.input = input
        self.output = output
        self.incoming = []
        self.outgoing = []
        self.is_bias = is_bias
        self.error = 0
        self.delta = None
        self.learning_rate = 0.3

    def connect_child(self, child_neuron, weight=0.4):
        """Connect a child neuron to our output, then connect ourselves as input to child neuron"""
        connection = NeuronConnection(self, child_neuron, weight=weight)
        self.outgoing.append(connection)
        child_neuron.incoming.append(connection)

    def activate(self, value=None):
        if self.is_bias:
            return self.output
        self.input = value or self.sum_inputs
        self.output = 1 / (1 + math.exp(-self.input))
        return self.output

    def train(self, target_output=0):
        # self.activate()
        input_derivative = 1 / (1 + math.exp(-self.input))
        input_derivative *= (1 - input_derivative)

        if target_output:
            # Only output layer needs error set
            self.error = target_output - self.output

        if self.is_output:
            self.delta = -self.error * input_derivative
        else:
            self.delta = sum([input_derivative * n.weight * n.child.delta for n in self.outgoing])

        for connection in self.outgoing:
            gradient = self.output * connection.child.delta
            connection.weight -= gradient * self.learning_rate

    @property
    def is_output(self):
        # We have no outgoing neurons, so we're the output layer!
        return not self.outgoing

    @property
    def sum_inputs(self):
        # if self.input:
        #     # If we're an input
        #     return self.input
        # else:
        # Otherwise we're a normal neuron
        sum = 0
        for connection in self.incoming:
            if connection.parent.output is None:
                connection.parent.activate()
            sum += connection.parent.output * connection.weight
        return sum


class Layer(object):
    def __init__(self, size=10):
        self.size = size
        self.neurons = [Neuron() for _ in range(size)]

    def activate(self, values=None):
        values = values or []
        for neuron, value in izip_longest(self.neurons, values):
            neuron.activate(value)

    def train(self, target_output):
        [n.train(target_output) for n in self.neurons]

    def connect_with_layer(self, layer):
        # Let's add our bias neuron now -- we know we're not on output layer because
        # we're trying to attach to another layer!
        self.neurons.append(Neuron(output=1, is_bias=True))

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

    def train(self, target_output):
        pass


if __name__ == "__main__":
    a = Neuron()
    b = Neuron()

    a.connect_child(b)

    for _ in range(9999):
        a.activate(2)
        b.activate()

        b.train(1)
        a.train()

        print "b error:", b.error



    # network = Network([3, 2, 1])
    # network.activate([1, 2, 3])
    #
    # print "INPUT"
    # for neuron in network.layers[0].neurons:
    #     print "In:", neuron.input, "Out:", neuron.output
    #
    # print "HIDDEN"
    # for layer in network.layers[1:-1]:
    #     for neuron in layer.neurons:
    #         print "In:", neuron.input, "Out:", neuron.output
    #
    # print "OUTPUT"
    # for neuron in network.layers[-1].neurons:
    #     print "In:", neuron.input, "Out:", neuron.output
