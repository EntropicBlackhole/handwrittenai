class NeuralNetwork {

	constructor(inputLayers, outputLayers, hiddenLayers, activationFunction) {
		this.inputLayers = inputLayers;
		this.outputLayers = outputLayers;
		this.hiddenLayers = hiddenLayers;
		this.activationFunction = activationFunction;

		this.weights = [];
		this.biases = [];

		for (let i = 0; i < this.inputLayers; i++) {
			this.weights.push(new Array(this.hiddenLayers));
			this.biases.push(new Array(this.hiddenLayers));
		}

		for (let i = 0; i < this.hiddenLayers; i++) {
			this.weights.push(new Array(this.outputLayers));
			this.biases.push(new Array(this.outputLayers));
		}
	}

	predict(inputs) {
		// Forward pass
		let outputs = inputs;

		for (let i = 0; i < this.hiddenLayers; i++) {
			outputs = this.activationFunction(outputs.dot(this.weights[i]) + this.biases[i]);
		}

		return outputs;
	}

	calculateLoss(inputs, outputs) {
		// Calculate the loss between the predicted outputs and the ground truth outputs
		let loss = 0;

		for (let i = 0; i < outputs.length; i++) {
			loss += Math.pow(outputs[i] - inputs[i], 2);
		}

		return loss;
	}

	saveToJSON() {
		// Save the neural network to a JSON file
		let data = {
			inputLayers: this.inputLayers,
			outputLayers: this.outputLayers,
			hiddenLayers: this.hiddenLayers,
			activationFunction: this.activationFunction,
			weights: this.weights,
			biases: this.biases
		};

		return JSON.stringify(data);
	}
	train(trainingData, epochs) {
		for (let epoch = 0; epoch < epochs; epoch++) {
			for (let i = 0; i < trainingData.length; i++) {
				// Forward pass
				let outputs = trainingData[i].inputs;

				for (let j = 0; j < this.hiddenLayers; j++) {
					outputs = this.activationFunction(outputs.dot(this.weights[j]) + this.biases[j]);
				}

				// Calculate the loss
				let loss = this.calculateLoss(outputs, trainingData[i].outputs);

				// Update the weights and biases
				for (let j = 0; j < this.hiddenLayers; j++) {
					let gradient = outputs.dot(this.weights[j].transpose()) * (outputs - trainingData[i].outputs);
					this.weights[j] = this.weights[j] - this.learningRate * gradient;
					this.biases[j] = this.biases[j] - this.learningRate * gradient.mean();
				}
			}
		}
	}

}

exports.NeuralNetwork = NeuralNetwork;