class NeuralNetwork {

	constructor(inputLayers, outputLayers, hiddenLayers, activationFunction, learningRate) {
		this.inputLayers = inputLayers;
		this.outputLayers = outputLayers;
		this.hiddenLayers = hiddenLayers;
		this.activationFunction = activationFunction;
		this.learningRate = learningRate;

		this.weights = [];
		this.biases = [];
		
		for (let i = 0; i < this.inputLayers; i++) {
			this.weights.push(new Array(this.hiddenLayers));
			this.biases.push(new Array(this.hiddenLayers));
			for (let j = 0; j < this.weights[i].length; j++) {
				this.weights[i][j] = Math.random() * 2 - 1;
				// console.log('added weight for input layer\ni:', i, 'j:', j)
			}
			for (let j = 0; j < this.biases[i].length; j++) {
				this.biases[i][j] = Math.random() * 2 - 1;
				// console.log('added bias for input layer\ni:', i, 'j:', j)
			}
		}
		let weightLength = this.weights.length
		let biasesLength = this.biases.length
		// console.log("Weights:", this.weights)
		// console.log("Biases:", this.biases)
		for (let i = 0; i < this.hiddenLayers; i++) {
			this.weights.push(new Array(this.outputLayers));
			this.biases.push(new Array(this.outputLayers));
			for (let j = 0; j < this.weights[i].length; j++) {
				this.weights[i+weightLength][j] = Math.random() * 2 - 1;
				// console.log('added weight for hidden layer\ni:', i, 'j:', j)
			}
			for (let j = 0; j < this.biases[i].length; j++) {
				this.biases[i+biasesLength][j] = Math.random() * 2 - 1;
				// console.log('added bias for hidden layer\ni:', i, 'j:', j)
			}
		}
		console.log("Weights:", this.weights)
		console.log("Biases:", this.biases)
	}

	predict(inputs) {
		// Forward pass
		let outputs = inputs;

		for (let i = 0; i < this.hiddenLayers; i++) {
			outputs = this.activationFunction(dot(outputs, this.weights[i]) + this.biases[i]);
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

	train(trainingData, epochs) {
		let bestLoss = Infinity;

		for (let epoch = 0; epoch < epochs; epoch++) {
			for (let i = 0; i < trainingData.length; i++) {
				// Forward pass
				let outputs = trainingData[i].inputs;
				
				for (let j = 0; j < this.hiddenLayers; j++) {
					outputs = this.activationFunction(dot(outputs, this.weights[j]) + this.biases[j]);
				}
				console.log(outputs)
				// Calculate the loss
				let loss = this.calculateLoss(outputs, trainingData[i].outputs);

				// Update the weights and biases
				for (let j = 0; j < this.hiddenLayers; j++) {
					// console.log(this.weights)
					let gradient = dot(outputs - trainingData[i].outputs, transpose(this.weights[j]));
					this.weights[j] = this.weights[j] - this.learningRate * gradient;
					this.biases[j] = this.biases[j] - this.learningRate * mean(gradient);
				}
				// console.log(this.weights)
				// Update the best loss
				if (loss < bestLoss) {
					bestLoss = loss;
				}
			}
		}

		return bestLoss;
	}


}

exports.NeuralNetwork = NeuralNetwork;

function dot(array1, array2) {
	let result = [];

	for (let i = 0; i < array1.length; i++) {
		result.push(array1[i] * array2[i]);
	}

	return result;
}

function transpose(array) {
	// console.log(array)
	let result = [];

	for (let i = 0; i < array.length; i++) {
		// console.log(array[i], i)
		result.push([]);
		for (let j = 0; j < array[i].length; j++) {
			result[i].push(array[j][i]);
		}
	}

	return result;
}

function mean(array) {
	let sum = 0;

	for (let i = 0; i < array.length; i++) {
		sum += array[i];
	}

	return sum / array.length;
}
