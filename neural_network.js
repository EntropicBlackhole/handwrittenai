class NeuralNetwork {
	constructor(layers, activations) {
		if (layers.length < 2) {
			throw new Error('Invalid number of layers');
		}
		if (activations.length !== layers.length - 1) {
			throw new Error('Invalid number of activations');
		}
		this.layers = [];
		for (let i = 0; i < layers.length - 1; i++) {
			const numInputs = layers[i];
			const numOutputs = layers[i + 1];
			const activation = activations[i];
			this.layers.push(new Layer(numInputs, numOutputs, activation));
		}
	}

	predict(inputs) {
		let outputs = inputs;
		for (let i = 0; i < this.layers.length; i++) {
			outputs = this.layers[i].forward(outputs);
		}
		return outputs;
	}

	train(inputs, labels, options = {}) {
		const defaults = {
			batchSize: inputs.length,
			epochs: 1,
			learningRate: 0.1,
			verbose: false,
			shuffle: true,
			validationSplit: 0,
			callback: null,
		};
		options = Object.assign(defaults, options);
		const numExamples = inputs.length;
		let indices = new Array(numExamples);
		for (let i = 0; i < numExamples; i++) {
			indices[i] = i;
		}
		for (let epoch = 0; epoch < options.epochs; epoch++) {
			if (options.shuffle) {
				shuffleArray(indices);
			}
			let batchStart = 0;
			while (batchStart < numExamples) {
				const batchEnd = Math.min(batchStart + options.batchSize, numExamples);
				const batchIndices = indices.slice(batchStart, batchEnd);
				const batchInputs = batchIndices.map(i => inputs[i]);
				const batchLabels = batchIndices.map(i => labels[i]);
				this.trainBatch(batchInputs, batchLabels, options.learningRate);
				batchStart = batchEnd;
			}
			if (options.validationSplit > 0) {
				const numTrainExamples = Math.floor(numExamples * (1 - options.validationSplit));
				const trainInputs = inputs.slice(0, numTrainExamples);
				const trainLabels = labels.slice(0, numTrainExamples);
				const valInputs = inputs.slice(numTrainExamples);
				const valLabels = labels.slice(numTrainExamples);
				const trainLoss = this.loss(trainInputs, trainLabels);
				const valLoss = this.loss(valInputs, valLabels);
				if (options.verbose) {
					console.log(`Epoch ${epoch + 1}: train loss = ${trainLoss}, validation loss = ${valLoss}`);
				}
				if (options.callback) {
					options.callback(epoch, trainLoss, valLoss);
				}
			} else {
				const trainLoss = this.loss(inputs, labels);
				if (options.verbose) {
					console.log(`Epoch ${epoch + 1}: train loss = ${trainLoss}`);
				}
				if (options.callback) {
					options.callback(epoch, trainLoss);
				}
			}
		}
	}
	trainBatch(inputs, labels, learningRate) {
		for (let i = 0; i < inputs.length; i++) {
			const input = inputs[i];
			const label = labels[i];
			let output = input;
			for (let j = 0; j < this.layers.length; j++) {
				output = this.layers[j].forward(output);
			}
			let error = this.loss.backward(output, label);
			for (let j = this.layers.length - 1; j >= 0; j--) {
				error = this.layers[j].backward(error, output);
				output = this.layers[j].inputs;
			}
			for (let j = 0; j < this.layers.length; j++) {
				this.layers[j].updateWeights(learningRate);
			}
		}
	}

	loss(inputs, labels) {
		let sum = 0;
		for (let i = 0; i < inputs.length; i++) {
			const input = inputs[i];
			const label = labels[i];
			const output = this.predict(input);
			sum += this.lossFunction(output, label);
		}
		return sum / inputs.length;
	}

	setLossFunction(lossFunction) {
		this.lossFunction = lossFunction;
	}

	save(filename) {
		const data = {
			layers: this.layers.map(layer => layer.serialize()),
		};
		const json = JSON.stringify(data);
		fs.writeFileSync(filename, json, 'utf8');
	}

	static load(filename) {
		const json = fs.readFileSync(filename, 'utf8');
		const data = JSON.parse(json);
		const layers = data.layers.map(layerData => Layer.deserialize(layerData));
		const network = new NeuralNetwork([], []);
		network.layers = layers;
		return network;
	}
}

class Layer {
	constructor(numInputs, numOutputs, activation) {
		this.weights = new Matrix(numInputs, numOutputs, () => Math.random() * 2 - 1);
		this.bias = new Matrix(1, numOutputs, () => Math.random() * 2 - 1);
		this.activation = activation;
		this.inputs = null;
		this.outputs = null;
		this.derivatives = null;
	}

	forward(inputs) {
		this.inputs = Matrix.fromArray(inputs);
		this.outputs = Matrix.dot(this.inputs, this.weights).add(this.bias).map(this.activation.forward.bind(this.activation));
		return this.outputs.toArray();
	}

	backward(error, output) {
		this.derivatives = Matrix.fromArray(output).map(this.activation.backward.bind(this.activation)).multiply(error);
		return Matrix.dot(this.derivatives, this.weights.transpose()).toArray();
	}

	updateWeights(learningRate) {
		const inputTranspose = this.inputs.transpose();
		const weightDeltas = Matrix.dot(inputTranspose, this.derivatives);
		this.weights = this.weights.subtract(weightDeltas.multiply(learningRate));
		this.bias = this.bias.subtract(this.derivatives.multiply(learningRate));
	}

	serialize() {
		return {
			weights: this.weights.toArray(),
			bias: this.bias.toArray(),
			activation: this.activation.constructor.name,
		};
	}

	static deserialize(data) {
		const layer = new Layer(data.weights.length, data.weights[0].length, new window[data.activation]());
		layer.weights = Matrix.fromArray(data.weights);
		layer.bias = Matrix.fromArray(data.bias);
		return layer;
	}
}

class Matrix {
	constructor(rows, cols, init = 0) {
		this.rows = rows;
		this.cols = cols;
		this.data = [];
		for (let i = 0; i < rows; i++) {
			this.data.push([]);
			for (let j = 0; j < cols; j++) {
				this.data[i].push(typeof init === 'function' ? init() : init);
			}
		}
	}

	static fromArray(arr) {
		return new Matrix(arr.length, 1, (i, j) => arr[i]);
	}

	toArray() {
		const arr = [];
		for (let i = 0; i < this.rows; i++) {
			for (let j = 0; j < this.cols; j++) {
				arr.push(this.data[i][j]);
			}
		}
		return arr;
	}

	randomize() {
		this.data = this.data.map(row => row.map(() => Math.random() * 2 - 1));
	}

	map(fn) {
		return new Matrix(this.rows, this.cols, (i, j) => fn(this.data[i][j], i, j));
	}

	add(other) {
		if (!(other instanceof Matrix)) {
			other = new Matrix(this.rows, this.cols, () => other);
		}
		return new Matrix(this.rows, this.cols, (i, j) => this.data[i][j] + other.data[i][j]);
	}

	subtract(other) {
		if (!(other instanceof Matrix)) {
			other = new Matrix(this.rows, this.cols, () => other);
		}
		return new Matrix(this.rows, this.cols, (i, j) => this.data[i][j] - other.data[i][j]);
	}

	multiply(other) {
		if (!(other instanceof Matrix)) {
			other = new Matrix(this.rows, this.cols, () => other);
		}
		if (this.cols !== other.rows) {
			throw new Error('Invalid matrix dimensions');
		}
		return new Matrix(this.rows, other.cols, (i, j) => {
			let sum = 0;
			for (let k = 0; k < this.cols; k++) {
				sum += this.data[i][k] * other.data[k][j];
			}
			return sum;
		});
	}

	transpose() {
		return new Matrix(this.cols, this.rows, (i, j) => this.data[j][i]);
	}

	static dot(a, b) {
		return a.multiply(b);
	}
}
exports.NeuralNetwork = NeuralNetwork;
exports.Layer = Layer;
exports.Matrix = Matrix;
function shuffleArray(array) {
	for (let i = array.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[array[i], array[j]] = [array[j], array[i]];
	}
}

/*
class NeuralNetwork {
	constructor(inputSize, outputSize, hiddenSizes, activation) {
		// Initialize the layers of the neural network
		let activationFunction = activation;
		this.layers = [];
		let prevSize = inputSize;
		for (let i = 0; i < hiddenSizes.length; i++) {
			const hiddenSize = hiddenSizes[i];
			this.layers.push(new Layer(prevSize, hiddenSize, activationFunction));
			prevSize = hiddenSize;
		}
		this.layers.push(new Layer(prevSize, outputSize, activationFunction));
	}

	predict(input) {
		let output = input;
		for (let i = 0; i < this.layers.length; i++) {
			output = this.layers[i].forward(output);
			console.log(`Layer ${i} output: ${output}`);
		}

		return output;
	}


	calculateLoss(output, label) {
		let loss = 0;
		for (let i = 0; i < output.length; i++) {
			console.log(`Label[${i}]: ${label[i]}`);
			loss -= label[i] * Math.log(output[i] + 1e-15);
			loss -= (1 - label[i]) * Math.log(1 - output[i] + 1e-15);
		}
		console.log(`Loss: ${loss}`);

		return loss;
	}


	updateParameters(loss, input) {
		// Update the parameters of the neural network using backpropagation
		let delta = loss;
		let prevOutput = null;
		for (let i = this.layers.length - 1; i >= 0; i--) {
			const layer = this.layers[i];
			if (prevOutput === null) {
				prevOutput = input;
			} else {
				prevOutput = this.layers[i + 1].output;
			}
			delta = layer.backward(delta, prevOutput);
		}
	}

	save(path) {
		// Save the neural network to a JSON file
		const json = JSON.stringify(this, null, 2);
		fs.writeFileSync(path, json);
	}
}


class Layer {
	constructor(numInputs, numOutputs, activation) {
		this.weights = new Array(numOutputs).fill(null).map(() => new Array(numInputs).fill(0));
		this.bias = new Array(numOutputs).fill(0);
		this.learningRate = 0.1;
		this.inputs = null;
		this.activation = activation;
	}

	forward(inputs) {
		this.inputs = inputs;
		let output = new Array(this.weights.length).fill(0);
		for (let j = 0; j < this.weights.length; j++) {
			for (let i = 0; i < inputs.length; i++) {
				output[j] += inputs[i] * this.weights[j][i];
			}
			output[j] += this.bias[j];
		}
		output = this.activation.forward(output);
		return output;
	}

	backward(error, output) {
		let delta = this.activation.backward(output);
		for (let j = 0; j < this.weights[0].length; j++) {
			let grad = 0;
			for (let i = 0; i < this.weights.length; i++) {
				grad += delta[i] * this.inputs[i][j];
			}
			this.weights[j] -= this.learningRate * grad;
		}
		let nextError = new Array(this.inputs.length).fill(0);
		for (let i = 0; i < this.inputs.length; i++) {
			for (let j = 0; j < this.weights[0].length; j++) {
				nextError[i] += delta[j] * this.weights[i][j];
			}
		}
		return nextError;
	}
}

exports.NeuralNetwork = NeuralNetwork;


/*
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
*/