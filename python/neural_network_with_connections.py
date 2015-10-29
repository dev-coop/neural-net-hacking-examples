import math

class NeuronConnection(object):

    def __init__(self, neuron, weight=1.0):
        self.neuron = neuron
        self.weight = weight


class Neuron(object):

    def __init__(self, input=None):
        self.input = input
        self.incoming_neurons = []
        self.outgoing_neurons = []

    def connect_child(self, child_neuron, weight=1.0):
        '''Connect a child neuron to our output, then connect ourselves as input to child neuron'''
        self.outgoing_neurons.append(NeuronConnection(child_neuron, weight=weight))
        child_neuron.incoming_neurons.append(NeuronConnection(self, weight=weight))

    def activate(self):
        self.output = 1 / (1 + math.exp(-self.sum_inputs))
        return self.output

    @property
    def sum_inputs(self):
        sum = 0
        if self.input:
            # If we're an input
            return self.input
        else:
            # Otherwise we're a normal neuron
            for connection in self.incoming_neurons:
                sum += connection.neuron.sum_inputs * connection.weight
        return sum


# TODO: NOT gate
# inputs 0 or 1 returns opposite, 1 and 0 respectively
neuron = Neuron(input=1)
neuron_2 = Neuron()

neuron.connect_child(neuron_2, weight=1)

neuron_2.activate()

print "Output:", neuron_2.output
