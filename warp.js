require('@tensorflow/tfjs-node');
const faceapi = require('face-api.js');
const { readFileSync } = require('fs');
const { join, extname } = require('path');
const { Point, Warper } = require('./imgwarp');

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

module.exports.AVATAR = imageFromFile(join(__dirname, 'avatar.png'));
module.exports.AVATAR_GHOST = imageFromFile(join(__dirname, 'avatar_ghost.png'));

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
    // Scale the avatar to the detection
    const xScale = toData.width / fromData.width;
    const yScale = toData.height / fromData.height;
    const overlay = document.createElement('canvas');
    overlay.width = toData.width;
    overlay.height = toData.height;
    overlay.getContext('2d').drawImage(sourceImg, 0, 0, overlay.width, overlay.height);

    // Get the pixel data and warp it
    const sourcePixels = overlay.getContext('2d').getImageData(0, 0, overlay.width, overlay.height);
    const warp = new Warper(sourcePixels);
    const warpedPixels = warp.warp(
        fromData.points.map(point => new Point(point[0] * xScale, point[1] * yScale)),
        toData.points.map(point => new Point(point[0] - toData.x, point[1] - toData.y)),
    );

    // TODO: This warp actually works but is a bit misaligned

    // Draw the output
    overlay.getContext('2d').putImageData(applyWarp ? warpedPixels : sourcePixels, 0, 0);
    return overlay;
}
