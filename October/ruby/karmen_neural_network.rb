class Neuron
  attr_accessor :input, :output

  def activation_function(input)
    Math.sin(input)
  end

  def activate
    @output = activation_function(input)
  end
end


neuron = Neuron.new
neuron.input = 1
neuron.activate
puts neuron.output
