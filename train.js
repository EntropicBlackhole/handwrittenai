const fs = require("fs");
const sharp = require("sharp");
const neuro = require("neuro.js");
train()

async function train() {
    let firstTime = Date.now();
    const imageDirPath = "dataset";
    const trainingData = [];
    for (const digitDirPath of fs.readdirSync(imageDirPath)) {
        let count = 0;
        for (const imagePath of fs.readdirSync(imageDirPath + "/" + digitDirPath)) {
            count++;
            
            if (count > 100) break
            console.log(imageDirPath + "/" + digitDirPath + "/" + imagePath)
            const imageBuffer = fs.readFileSync(imageDirPath + "/" + digitDirPath + "/" + imagePath);
            const grayImage = sharp(imageBuffer).grayscale();
            const normalizedImage = grayImage.normalize().flatten({ background: '#F0A703' })
            const resizedImage = normalizedImage.resize(28, 28);
            fs.writeFileSync('./test.png', await resizedImage.toBuffer())
            const imageBuffer2 = await resizedImage.raw().toBuffer()
            const pixelValues = JSON.parse(JSON.stringify(imageBuffer2)).data
            const numericalValues = pixelValues.map((v) => v / 255.0);
            const label = digitDirPath
            console.log(parseInt(label) / 10)
            trainingData.push({ input: numericalValues, output: parseInt(label) / 10 });
        }
    }
    let secondTime = Date.now();
    console.log("Finished parsing everything, took", secondTime - firstTime, "ms");
    const digitClassifier = new neuro.classifiers.NeuralNetwork(); 
    console.log("training")
    console.log(trainingData)
    await digitClassifier.trainBatch(trainingData); 
    console.log("trained")
    let thirdTime = Date.now();
    console.log("Finished training, took", thirdTime-secondTime, "ms, and", thirdTime-firstTime, "ms since the start")
    fs.writeFileSync('./training_data.json', JSON.stringify(digitClassifier.toJSON(), null, 2))
}