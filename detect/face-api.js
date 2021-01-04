/*
    "@tensorflow/tfjs-node": "^1.7.0",
    "face-api.js": "^0.22.2"
 */

const { SCORE_MIN } = require('./config');
const { join } = require('path');
const faceapi = require('face-api.js');
require('@tensorflow/tfjs-node');

// Force faceapi to be in a browser (electron) context
faceapi.env.monkeyPatch({
    Canvas: HTMLCanvasElement,
    Image: HTMLImageElement,
    ImageData: ImageData,
    Video: HTMLVideoElement,
    createCanvasElement: () => document.createElement('canvas'),
    createImageElement: () => document.createElement('img'),
});

module.exports.load = async () => {
    const dir = join(__dirname, 'face-api-weights');

    // Tiny
    await faceapi.nets.tinyFaceDetector.loadFromDisk(dir);
    await faceapi.nets.faceLandmark68TinyNet.loadFromDisk(dir);

    // Full
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(dir);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(dir);
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
