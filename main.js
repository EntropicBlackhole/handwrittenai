const fs = require("fs");
const sharp = require("sharp");
const neuro = require("neuro.js");
start() 

async function start() {
  const trainingData = JSON.parse(fs.readFileSync('./training_data.json'));
  let imageToGet = process.argv[2];
  const imageBuffer = fs.readFileSync(imageToGet);
  const grayImage = sharp(imageBuffer).grayscale();
  
  const normalizedImage = grayImage.normalize();
  const resizedImage = normalizedImage.resize(14, 14)
  fs.writeFileSync('./test.png', await resizedImage.toBuffer())
  let imageBuffer2 = await resizedImage.raw().toBuffer()
  const pixelValues = JSON.parse(JSON.stringify(imageBuffer2)).data
  const numericalValues = pixelValues.map((v) => v / 255.0);
  var digitClassifier = new neuro.classifiers.NeuralNetwork();
  digitClassifier.fromJSON(trainingData)
  console.log(digitClassifier.classify(numericalValues)); 
}

