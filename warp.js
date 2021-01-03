require('@tensorflow/tfjs-node');
const faceapi = require('face-api.js');
const { readFileSync } = require('fs');
const { extname } = require('path');
const { Point, Warper } = require('./warper');

// Force faceapi to be in a browser (electron) context
faceapi.env.monkeyPatch({
    Canvas: HTMLCanvasElement,
    Image: HTMLImageElement,
    ImageData: ImageData,
    Video: HTMLVideoElement,
    createCanvasElement: () => document.createElement('canvas'),
    createImageElement: () => document.createElement('img'),
});

const imageFromFile = file => {
    const data = readFileSync(file);
    const img = new Image();
    img.src = 'data:image/' + extname(file) + ';base64,' + data.toString('base64');
    return img;
};

const SCORE_MIN = 0.25;
const OVERLAY_PADDING = 0.5;

module.exports.loadImage = path => imageFromFile(path)

module.exports.loadNets = async () => {
    // Tiny
    await faceapi.nets.tinyFaceDetector.loadFromDisk('./weights');
    await faceapi.nets.faceLandmark68TinyNet.loadFromDisk('./weights');

    // Full
    await faceapi.nets.ssdMobilenetv1.loadFromDisk('./weights');
    await faceapi.nets.faceLandmark68Net.loadFromDisk('./weights');
};

const faceDetectionOpts = useTiny => useTiny
    ? new faceapi.TinyFaceDetectorOptions({ scoreThreshold: SCORE_MIN })
    : new faceapi.SsdMobilenetv1Options({ minConfidence: SCORE_MIN });

module.exports.detect = async (img, useTiny = true) => {
    // Perform the detection
    const detectStart = Date.now();
    const detection = await faceapi
        .detectSingleFace(img, faceDetectionOpts(useTiny))
        .withFaceLandmarks(useTiny);
    const detectEnd = Date.now();

    // Output data
    return {
        data: detection,
        score: detection && detection.detection ? detection.detection.score : 0,
        points: detection && detection.landmarks ? detection.landmarks.positions.map(point => [point.x, point.y]) : [],
        width: detection && detection.detection ? detection.detection.box.width : 0,
        height: detection && detection.detection ? detection.detection.box.height : 0,
        x: detection && detection.detection ? detection.detection.box.x : 0,
        y: detection && detection.detection ? detection.detection.box.y : 0,
        duration: detectEnd - detectStart,
        success: !!(detection && detection.detection && detection.landmarks),
    };
};

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

    // TODO: This warp actually works but is a bit misaligned

    // Return the overlay and the scaling applied
    return { overlay, xScale, yScale, xPad, yPad };
}
