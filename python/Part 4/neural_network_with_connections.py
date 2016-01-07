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
        if not self.is_bias:
            if self.is_output:
                # this is the derivative of the error function
                self.delta = self.output - target_output
            else:
                self.delta = sum([n.weight * n.child.delta for n in self.outgoing])

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

    def train(self, target_outputs=list()):
        # [n.train(target_output) for n in self.neurons]
        [n.train(o) for n, o in izip_longest(self.neurons, target_outputs)]

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

    def train(self, target_outputs):
        # train output layer, pass it expected outputs
        self.layers[-1].train(target_outputs)
        # train the rest in reverse, not passed expected outputs because they don't care
        [layer.train() for layer in reversed(self.layers[0:-1])]

    def calculate_error(self, target_outputs):
        error = 0
        for n, output in izip_longest(self.layers[-1].neurons, target_outputs):
            error += 0.5 * math.pow(n.output - output, 2)
        return error


class Trainer(object):

    def __init__(self, network, training_data):
        self.network = network
        self.training_data = training_data

    def train(self, epochs=10001, log_frequency=1000):
        for epoch in xrange(epochs):
            accumulated_error = 0
            for input_data, target_output in self.training_data:
                self.network.activate(input_data)  # pass current loop input data
                self.network.train(target_output)  # pass current loop target data
                accumulated_error += self.network.calculate_error(target_output)
            accumulated_error /= len(self.training_data)
            if epoch == 0 or epoch % log_frequency == 0 or epoch == epochs - 1:
                print "Epoch ", epoch, "error =", accumulated_error


if __name__ == "__main__":
    OR_GATE = (
        # (input, output)
        ([0, 0], [0]),
        ([1, 0], [1]),
        ([1, 1], [1]),
        ([0, 1], [1]),
    )

    network = Network([2, 1])
    trainer = Trainer(network, OR_GATE)
    trainer.train()
