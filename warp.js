const { readFileSync } = require('fs');
const { extname } = require('path');
const { Point, Warper } = require('./warper');

const imageFromFile = file => {
    const data = readFileSync(file);
    const img = new Image();
    img.src = 'data:image/' + extname(file) + ';base64,' + data.toString('base64');
    return img;
};

const OVERLAY_PADDING = 0.5;

module.exports.loadImage = path => imageFromFile(path)

module.exports.warp = (fromData, toData, sourceImg, applyWarp) => {
    // Scale the avatar to have the same size face detection box as the target
    // Also apply some whitespace padding around
    const xScale = toData.width / fromData.width;
    const yScale = toData.height / fromData.height;
    const xPad = sourceImg.width * OVERLAY_PADDING;
    const yPad = sourceImg.height * OVERLAY_PADDING;
    const overlay = document.createElement('canvas');
    overlay.width = (sourceImg.width * xScale) + (xPad * 2);
    overlay.height = (sourceImg.height * yScale) + (yPad * 2);
    overlay.getContext('2d').drawImage(
        sourceImg,
        xPad,
        yPad,
        sourceImg.width * xScale,
        sourceImg.height * yScale,
    );

    // Get the pixel data and warp it
    const sourcePixels = overlay.getContext('2d').getImageData(0, 0, overlay.width, overlay.height);
    const warp = new Warper(sourcePixels);
    const warpedPixels = warp.warp(
        fromData.points.map(point => new Point(
            (point[0] * xScale) + xPad,
            (point[1] * yScale) + yPad,
        )),
        toData.points.map(point => new Point(
            point[0] - toData.x + (avatarData.x * xScale) + xPad,
            point[1] - toData.y + (avatarData.y * yScale) + yPad,
        )),
    );
    overlay.getContext('2d').putImageData(applyWarp ? warpedPixels : sourcePixels, 0, 0);

    // Return the overlay and the scaling applied
    return { overlay, xScale, yScale, xPad, yPad };
}
