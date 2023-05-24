const fs = require("fs");
const sharp = require("sharp");
const neuro = require("neuro.js");
train()

async function train() {
    const imageDirPath = "images";
    const trainingData = [];
    for (const imagePath of fs.readdirSync(imageDirPath)) {
        const imageBuffer = fs.readFileSync('images/' + imagePath);
        const grayImage = sharp(imageBuffer).grayscale();
        const normalizedImage = grayImage.normalize();
        const resizedImage = normalizedImage;
        let imageBuffer2 = await resizedImage.raw().toBuffer()
        const pixelValues = JSON.parse(JSON.stringify(imageBuffer2)).data
        const numericalValues = pixelValues.map((v) => v / 255.0);
        const label = imagePath.split("-")[1].split(".")[0];
        trainingData.push({ input: numericalValues, output: parseInt(label) / 10 });
    }
    const digitClassifier = new neuro.classifiers.NeuralNetwork(); console.log("training")
    await digitClassifier.trainBatch(trainingData); 
    fs.writeFileSync('./training_data.json', JSON.stringify(digitClassifier.toJSON(), null, 2))
}