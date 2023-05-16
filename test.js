const Canvas = require('canvas')
const canvas = Canvas.createCanvas(224, 140)
const fs = require('fs')
let ctx = canvas.getContext("2d")
Canvas.loadImage("./digit-handwritten-dataset.png").then((image) => {
	ctx.drawImage(image, 0, 0, 224, 140)
	for (let i = 0; i < 10; i++) {
		for (let j = 0; j < 16; j++) {
			let newCanvas = Canvas.createCanvas(14, 14)
			let newCtx = newCanvas.getContext("2d")
			newCtx.putImageData(ctx.getImageData(j * 14, i * 14, 14, 14), 0, 0)
			fs.writeFileSync(`./images/${j+1}-${i}.png`, newCanvas.toBuffer())
		}
	}
});