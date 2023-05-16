const fs = require("fs");
const sharp = require("sharp");
const NeuralNetwork = require("./neural_network.js");

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

// Create a new neural network
const neuralNetwork = new NeuralNetwork(784, 10);

// Train the neural network
for (const trainingExample of fs.readFileSync("training-data.csv", "utf8").split("\n")) {
  // Split the training example into the input and the label
  const [input, label] = trainingExample.split(",");

  // Convert the input to a numerical value
  const numericalInput = input.map((v) => v / 255.0);

  // Feed the numerical input to the neural network
  const prediction = neuralNetwork.predict(numericalInput);

  // Calculate the loss
  const loss = neuralNetwork.calculateLoss(prediction, label);

  // Update the neural network parameters
  neuralNetwork.updateParameters(loss);
}

// Save the neural network
neuralNetwork.save("neural-network.json");

// Predict the value of a number by again converting the PNG and passing the numeric values
const prediction = neuralNetwork.predict(numericalValues);

// Print the prediction
console.log(prediction);