class NeuralNetwork {
	constructor(inputSize, outputSize) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;

		// Create the layers of the neural network
		this.layers = [
			new Layer(inputSize, 128),
			new Layer(128, 64),
			new Layer(64, outputSize),
		];
	}

	predict(input) {
		// Feed the input to the neural network
		for (const layer of this.layers) {
			input = layer.forward(input);
		}

		// Return the output of the neural network
		return input;
	}

	calculateLoss(prediction, label) {
		// Calculate the cross-entropy loss
		const loss = -prediction.dot(label);

		// Return the loss
		return loss;
	}

	updateParameters(loss) {
		// Update the parameters of the neural network using gradient descent
		for (let i = 0; i < this.layers.length; i++) {
			this.layers[i].updateParameters(loss);
		}
	}

	save(path) {
		// Save the neural network to a JSON file
		const json = JSON.stringify(this, null, 2);
		fs.writeFileSync(path, json);
	}
}

exports.NeuralNetwork = NeuralNetwork;

const tanh = (x) => (1 + Math.exp(-x ^ 2)) ^ (-1 / 2);

class NeuralNetwork2 {
	constructor(inputSize, hiddenSize, outputSize) {
		this.inputSize = inputSize;
		this.hiddenSize = hiddenSize;
		this.outputSize = outputSize;

		this.weights = new Array(this.hiddenSize).fill(0);
		this.biases = new Array(this.hiddenSize).fill(0);

		this.outputWeights = new Array(this.outputSize).fill(0);
		this.outputBiases = new Array(this.outputSize).fill(0);
	}

	forward(inputs) {
		const hidden = inputs.map(x => tanh(x * this.weights[0] + this.biases[0]));
		const outputs = hidden.map(x => tanh(x * this.outputWeights[0] + this.outputBiases[0]));
		return outputs;
	}

	backpropagate(inputs, outputs, labels) {
		const hiddenErrors = outputs.map(x => (labels[x] - x) * tanh(x));
		const inputErrors = hiddenErrors.map(x => x * this.weights[0]);

		this.outputWeights[0] += inputErrors.reduce((a, b) => a + b, 0);
		this.outputBiases[0] += inputErrors.reduce((a, b) => a + b, 0);

		this.weights[0] += hiddenErrors.reduce((a, b) => a + b, 0);
		this.biases[0] += hiddenErrors.reduce((a, b) => a + b, 0);
	}

	train(inputs, labels, iterations) {
		for (let i = 0; i < iterations; i++) {
			const outputs = this.forward(inputs);
			this.backpropagate(inputs, outputs, labels);
		}
	}
}