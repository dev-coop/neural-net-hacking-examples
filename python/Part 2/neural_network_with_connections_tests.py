import unittest

from neural_network_with_connections import Neuron, Layer

from unittest import TestCase


class NeuralNetworkTests(TestCase):

    def test_connection_adds_to_incoming_and_outgoing_arrays(self):
        neuron = Neuron(input=0)
        neuron_2 = Neuron()
        neuron.connect_child(neuron_2, weight=1)
        # Make sure there's only one connection back and forth
        assert [neuron == connection.neuron for connection in neuron_2.incoming].count(True) == 1
        assert [neuron_2 == connection.neuron for connection in neuron.outgoing].count(True) == 1

    def test_activate_applies_proper_math(self):
        '''Let's make sure the sigmoid function is working properly, 0 * 0 weight should be 0.5'''
        neuron = Neuron(input=0)
        neuron.activate()
        assert neuron.output == 0.5

    def test_activate_applies_proper_math_over_many_neurons(self):
        neuron = Neuron(input=0)
        neuron_2 = Neuron()
        neuron.connect_child(neuron_2, weight=1)
        neuron_2.activate()
        # sucks comparing floats in python apparently, absolute difference check
        assert abs(neuron_2.output - 0.622459331202) < 0.000000001

    def test_layer_connections_are_appropriate(self):
        layer_1 = Layer(size=3)
        layer_2 = Layer(size=2)
        layer_1.connect_with_layer(layer_2)

        for neuron in layer_1.neurons:
            # Should have proper # of connections
            assert len(neuron.outgoing) == len(layer_2.neurons)
            # Should connect to proper neurons
            for child_neuron in layer_2.neurons:
                assert any([child_neuron == connection.neuron for connection in neuron.outgoing])


if __name__ == '__main__':
    unittest.main()
