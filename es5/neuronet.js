
var Connection = function(source, target){
    this.source = source;
    this.target = target;
    this.weight = 0.5;
}

var Neuron = function(){
    var self = this;
    this.input = 0;
    this.output = 0;
    this.incoming = [];
    this.outgoing = [];
    this.activationFn = function(input){
        return 1 / (1 + Math.exp(-input));
    };
    this.activate = function(inputValue){
        var output = inputValue || self.sum();
        this.output = this.activationFn(output);
        return this.output;
    };
    this.sum = function(){
        var sum = 0;
        self.incoming.map(function(conn){
            sum += conn.source.output * conn.weight;
        })
        return sum;
    };
    this.connect = function(target){
        var conn = new Connection(this, target);
        this.outgoing.push(conn);
        target.incoming.push(conn);
    };
}

var neuronA = new Neuron();
var neuronB = new Neuron();

neuronA.connect(neuronB);

neuronA.activate(5);
console.log(neuronA.output);

neuronB.activate();
console.log(neuronB.output);
