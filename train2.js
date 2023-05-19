const { NeuralNetwork } = require('./neural_network2')
// Create a new neural network with 2 input layers, 1 hidden layer, and 1 output layer.
const neuralNetwork = new NeuralNetwork(2, 1, 1, sigmoid, 0.01);

// Create some training data.
const trainingData = [
	{
		inputs: [1, 0],
		outputs: [0]
	},
	{
		inputs: [0, 1],
		outputs: [1]
	}
];

// Train the neural network.
neuralNetwork.train(trainingData, 1000);

// Make a prediction.
const prediction = neuralNetwork.predict([1, 0]);

// Print the prediction.
console.log(prediction, 'a'); // 0

function sigmoid(x) {
	let result = [];

	for (let i = 0; i < x.length; i++) {
		result.push(1 / (1 + Math.exp(-x[i])));
	}

	return result;
}