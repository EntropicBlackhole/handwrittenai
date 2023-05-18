const neuralNetwork = new NeuralNetwork(2, 1, 1, 'sigmoid');

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

neuralNetwork.train(trainingData, 1000);

const prediction = neuralNetwork.predict([1, 0]);

console.log(prediction); // 0
