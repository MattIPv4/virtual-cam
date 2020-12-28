require('@tensorflow/tfjs-node');
const canvas = require('canvas');
const faceapi = require('face-api.js');
const { writeFile } = require('fs').promises;

const { Canvas, Image, ImageData, loadImage } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const USE_TINY = false;
const SCORE_MIN = 0.1;

const log = msg => console.log(`[${(new Date()).toISOString()}]: ${msg}`);

const getTop = data => data
    .map((a) => a.y)
    .reduce((a, b) => Math.min(a, b));

const getMeanPosition = data => data
    .map((a) => [a.x, a.y])
    .reduce((a, b) => [a[0] + b[0], a[1] + b[1]])
    .map(a => a / data.length);

const loadNets = async () => {
    log('Loading nets...');
    await faceapi.nets.ssdMobilenetv1.loadFromDisk('./weights');
    await faceapi.nets.tinyFaceDetector.loadFromDisk('./weights');
    await faceapi.nets.faceLandmark68Net.loadFromDisk('./weights');
    await faceapi.nets.faceLandmark68TinyNet.loadFromDisk('./weights');
    await faceapi.nets.faceExpressionNet.loadFromDisk('./weights');
};

const faceDetectionOpts = () => USE_TINY
    ? new faceapi.TinyFaceDetectorOptions({ scoreThreshold: SCORE_MIN })
    : new faceapi.SsdMobilenetv1Options({ minConfidence: SCORE_MIN });

const processImage = async img => {
    // Perform the detection
    log('Detecting face...');
    const detectStart = Date.now();
    const detection = await faceapi
        .detectSingleFace(img, faceDetectionOpts())
        .withFaceLandmarks(USE_TINY)
        .withFaceExpressions();
    log(`Detected face in ${Date.now() - detectStart} ms`);

    // Get rotation
    log('Calculating rotation...');
    const eyeRight = getMeanPosition(detection.landmarks.getRightEye());
    const eyeLeft = getMeanPosition(detection.landmarks.getLeftEye());
    const eyeMidpoint = [(eyeLeft[0] + eyeRight[0]) / 2, (eyeLeft[1] + eyeRight[1]) / 2];
    const nose = getMeanPosition(detection.landmarks.getNose());
    const mouth = getMeanPosition(detection.landmarks.getMouth());
    const jaw = getTop(detection.landmarks.getJawOutline());
    const pitch = (jaw - mouth[1]) / detection.detection.box.height + 0.5;
    const yaw = (eyeLeft[0] + (eyeRight[0] - eyeLeft[0]) / 2 - nose[0]) / detection.detection.box.width;
    const rawRoll = Math.atan2(eyeMidpoint[1] - nose[1], eyeMidpoint[0] - nose[0]);
    const roll = rawRoll < 0 ? (2 * Math.PI) - Math.abs(rawRoll) : rawRoll;

    // Output data
    return {
        score: detection.detection.score,
        x: detection.detection.box.x,
        y: detection.detection.box.y,
        width: detection.detection.box.width,
        height: detection.detection.box.height,
        eyeRight,
        eyeLeft,
        eyeMidpoint,
        nose,
        mouth,
        jaw,
        yaw,
        pitch,
        roll,
    };
};

const main = async () => {
    // Load the nets we need
    await loadNets();

    // Load the sample face in
    log('Loading face...');
    const face = await loadImage('test/rotated.jpg');
    const data = await processImage(face);
    console.log(data);

    // Output
    log('Generating output...');
    const out = faceapi.createCanvasFromMedia(face);
    const ctx = out.getContext('2d');

    ctx.beginPath();
    ctx.rect(data.x, data.y, data.width, data.height);
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 5;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(data.eyeMidpoint[0], data.eyeMidpoint[1], 10, 0, 2 * Math.PI);
    ctx.fillStyle = 'green';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(data.eyeRight[0], data.eyeLeft[1], 10, 0, 2 * Math.PI);
    ctx.fillStyle = 'yellow';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(data.eyeLeft[0], data.eyeLeft[1], 10, 0, 2 * Math.PI);
    ctx.fillStyle = 'pink';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(data.nose[0], data.nose[1], 10, 0, 2 * Math.PI);
    ctx.fillStyle = 'cyan';
    ctx.fill();

    ctx.save();
    ctx.beginPath();
    ctx.translate(data.x + data.width / 2, data.y + data.height / 2);
    ctx.rotate(data.roll);
    ctx.rect(-data.width / 2, -data.height / 2, data.width, data.height);
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 5;
    ctx.stroke();

    await writeFile('test/output.jpg', out.toBuffer('image/jpeg'));

    log('Done!');
};

main();
