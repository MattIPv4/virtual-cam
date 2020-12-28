require('@tensorflow/tfjs-node');
const canvas = require('canvas');
const faceapi = require('face-api.js');
const { writeFile } = require('fs').promises;

const { Canvas, Image, ImageData, loadImage } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const log = msg => console.log(`[${(new Date()).toISOString()}]: ${msg}`);

const getTop = data => data
    .map((a) => a.y)
    .reduce((a, b) => Math.min(a, b));

const getMeanPosition = data => data
    .map((a) => [a.x, a.y])
    .reduce((a, b) => [a[0] + b[0], a[1] + b[1]])
    .map(a => a / data.length);

const main = async () => {
    // Load the nets we need
    log('Loading nets...');
    await faceapi.nets.tinyFaceDetector.loadFromDisk('./weights');
    await faceapi.nets.faceLandmark68TinyNet.loadFromDisk('./weights');
    await faceapi.nets.faceExpressionNet.loadFromDisk('./weights');

    // Load the sample face in
    log('Loading face...');
    const face = await loadImage('test/default.jpg');
    const displaySize = { width: face.width, height: face.height };
    const out = faceapi.createCanvasFromMedia(face);
    faceapi.matchDimensions(out, displaySize);

    // Perform the detection
    log('Detecting face...');
    const detectStart = Date.now();
    const detection = await faceapi
        .detectSingleFace(face, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.2 }))
        .withFaceLandmarks(true)
        .withFaceExpressions();
    log(`Detected face in ${Date.now() - detectStart} ms`);

    // Get rotation
    log('Calculating rotation...');
    const eyeRight = getMeanPosition(detection.landmarks.getRightEye());
    const eyeLeft = getMeanPosition(detection.landmarks.getLeftEye());
    const nose = getMeanPosition(detection.landmarks.getNose());
    const mouth = getMeanPosition(detection.landmarks.getMouth());
    const jaw = getTop(detection.landmarks.getJawOutline());
    const pitch = (jaw - mouth[1]) / detection.detection.box.height + 0.5;
    const yaw = (eyeLeft[0] + (eyeRight[0] - eyeLeft[0]) / 2 - nose[0]) /
        detection.detection.box.width;
    const roll = (Math.atan2(eyeLeft[1] - eyeRight[1], eyeLeft[0] - eyeRight[0]) / (2 * Math.PI)) - 0.5;

    // Output data
    const data = {
        score: detection.detection.score,
        x: detection.detection.box.x,
        y: detection.detection.box.y,
        width: detection.detection.box.width,
        height: detection.detection.box.height,
        yaw,
        pitch,
        roll,
    };
    console.log(data);

    // Output
    log('Generating output...');
    const resizedDetection = faceapi.resizeResults(detection, displaySize);
    faceapi.draw.drawDetections(out, resizedDetection);
    faceapi.draw.drawFaceLandmarks(out, resizedDetection);
    faceapi.draw.drawFaceExpressions(out, resizedDetection);
    await writeFile('test/output.jpg', out.toBuffer('image/jpeg'));

    log('Done!');
};

main();
