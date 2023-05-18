const fs = require("fs");
const sharp = require("sharp");
const { NeuralNetwork } = require("./neural_network.js");
class Sigmoid {
  forward(x) {
    return 1 / (1 + Math.exp(-x));
  }

  backward(y) {
    return y * (1 - y);
  }
}

class ReLU {
  forward(x) {
    return Math.max(0, x);
  }

  backward(y) {
    return y > 0 ? 1 : 0;
  }
}

class Tanh {
  forward(x) {
    return Math.tanh(x);
  }

  backward(y) {
    return 1 - y * y;
  }
}
/*
// Get the path to the image file
const imagePath = "image.png";

// Read the image file
const imageBuffer = fs.readFileSync(imagePath);

// Convert the image to a grayscale image
const grayImage = sharp(imageBuffer).grayscale();

// Normalize the image so that the pixel values range from 0 to 1
const normalizedImage = grayImage.div(255.0);

// Resize the image to a fixed size
const resizedImage = normalizedImage.resize([28, 28]);

// Extract the pixel values from the image
const pixelValues = resizedImage.data;

// Convert the pixel values to numerical values
const numericalValues = pixelValues.map((v) => v / 255.0);
*/
// Create a new neural network
// const nn = new NeuralNetwork([784, 128, 64, 10], [new ReLU(), new ReLU(), new Softmax()]);
const neuralNetwork = new NeuralNetwork(784, 10, [3], new ReLU());

// Train the neural network
const trainingData = fs.readFileSync("training-data.csv", "utf8");
for (const trainingExample of trainingData.split("\n")) {
  // Split the training example into the input and the label
  const trainingDataLine = trainingExample.split(",");
  let label = trainingDataLine[trainingDataLine.length-1];
  trainingDataLine.shift()
  let input = JSON.parse(`[${trainingDataLine}]`)

  // Convert the input to a numerical value
  // const numericalInput = input.map((v) => v / 255.0);

  // Feed the numerical input to the neural network
  const prediction = neuralNetwork.predict(input);

  // Calculate the loss
  const oneHotLabel = oneHotEncode(label, 10);
  const loss = neuralNetwork.calculateLoss(prediction, oneHotLabel);
  // const loss = neuralNetwork.calculateLoss(prediction, label);

  // Update the neural network parameters
  neuralNetwork.updateParameters(loss, input);
}

// Save the neural network
neuralNetwork.save("neural-network.json");

// Predict the value of a number by again converting the PNG and passing the numeric values
const prediction = neuralNetwork.predict(numericalValues);

// Print the prediction
console.log(prediction);

function oneHotEncode(label, numClasses) {
  const oneHot = [];
  for (let i = 0; i < numClasses; i++) {
    oneHot.push(i === label ? 1 : 0);
  }
  return oneHot;
}

