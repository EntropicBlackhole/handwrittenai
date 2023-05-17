const fs = require("fs");
const sharp = require("sharp");
start()


async function start() {
    // Get the path to the directory containing the PNG images

    const imageDirPath = "images";

    // Create an array to store the training data
    const trainingData = [];

    // Iterate over the images in the directory
    for (const imagePath of fs.readdirSync(imageDirPath)) {
        // Read the image file
        const imageBuffer = fs.readFileSync('images/' + imagePath);

        // Convert the image to a grayscale image
        const grayImage = sharp(imageBuffer).grayscale();
        // Normalize the image so that the pixel values range from 0 to 1
        const normalizedImage = grayImage.normalize();
        // console.log(normalizedImage)
        // Resize the image to a fixed size
        const resizedImage = normalizedImage.resize(28, 28);
        //start(resizedImage)
        //return
        // Extract the pixel values from the image
        let imageBuffer2 = await resizedImage.raw().toBuffer()
        const pixelValues = JSON.parse(JSON.stringify(imageBuffer2)).data
        // Convert the pixel values to numerical values
        const numericalValues = pixelValues.map((v) => v / 255.0);
        // console.log(numericalValues)
        // Get the label of the image
        const label = imagePath.split("-")[1].split(".")[0];

        // Add the training example to the training data array
        trainingData.push([numericalValues.toBuffer(), label]);
    }

    // Save the training data to a CSV file
    fs.writeFileSync("training-data.csv", trainingData.join("\n"));

}