require('@tensorflow/tfjs-node');
const faceapi = require('face-api.js');
const { readFileSync } = require('fs');
const { join, extname } = require('path');
const warp = require('ndarray-idw-warp');
const ndarrayFromCanvas = require('ndarray-from-canvas');
const canvasFromNdarray = require('canvas-from-ndarray');

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

module.exports.warp = (fromData, toData, sourceImg) => {
    // Scale the avatar to the detection
    const xScale = toData.width / fromData.width;
    const yScale = toData.height / fromData.height;
    const overlay = document.createElement('canvas');
    overlay.width = toData.width;
    overlay.height = toData.height;
    overlay.getContext('2d').drawImage(sourceImg, 0, 0, overlay.width, overlay.height);

    // Build the warp map (scaled)
    const warpMap = fromData.points.map((point, i) => [
        point[0] * xScale,
        point[1] * yScale,
        toData.points[i][0],
        toData.points[i][1],
    ]);

    // Pin all our edges not warped
    const edges = [];
    for (let x = 0; x < overlay.width; x++) {
        edges.push([x, 0, x, 0]); // top
        edges.push([x, overlay.height - 1, x, overlay.height - 1]); // bottom
    }
    for (let y = 0; y < overlay.width; y++) {
        edges.push([0, y, 0, y]); // left
        edges.push([overlay.width - 1, y, overlay.width - 1, y]); // right
    }
    console.log(edges);

    // Get the pixel data and warp it
    const sourcePixels = ndarrayFromCanvas(overlay);
    const warpedPixels = warp(sourcePixels, [...edges, ...warpMap]); // TODO: Help! This line is super slow & borked
    console.log(sourcePixels);
    console.log(warpedPixels);

    // Draw the output
    return canvasFromNdarray(warpedPixels, overlay);
}
