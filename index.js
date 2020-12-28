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

const getDistance = (fromData, toData) => [toData[0] - fromData[0], toData[1] - fromData[1]];

const getAbsDistance = (fromData, toData) => {
    const distance = getDistance(fromData, toData);
    return Math.sqrt(Math.pow(distance[0], 2) + Math.pow(distance[1], 2))
};

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

const DEFAULT_TRANSFORMS = {
    scale: 2.25,
    translate: { x: -75, y: 150 },
    rotate: 0.185,
};

const transforms = (base, updated) => {
    // const baseAvgSize = (base.width + base.height) / 2;
    const baseAvgSize = (getAbsDistance(base.eyeLeft, base.nose) + getAbsDistance(base.eyeRight, base.nose)) / 2;
    // const updatedAvgSize = (updated.width + updated.height) / 2;
    const updatedAvgSize = (getAbsDistance(updated.eyeLeft, updated.nose) + getAbsDistance(updated.eyeRight, updated.nose)) / 2;
    const scale = updatedAvgSize / baseAvgSize;

    const baseCenter = [base.x + base.width / 2, base.y + base.height / 2];
    const updatedCenter = [updated.x + updated.width / 2, updated.y + updated.height / 2];
    const translate = { x: updatedCenter[0] - baseCenter[0], y: updatedCenter[1] - baseCenter[1] };

    const rotate = updated.roll - base.roll;

    return {
        scale,
        translate,
        rotate,
    };
};

const plottedFace = (baseCanvas, detectionData) => {
    const ctx = baseCanvas.getContext('2d');
    ctx.save();

    ctx.beginPath();
    ctx.rect(detectionData.x, detectionData.y, detectionData.width, detectionData.height);
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 5;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(detectionData.eyeMidpoint[0], detectionData.eyeMidpoint[1], 10, 0, 2 * Math.PI);
    ctx.fillStyle = 'green';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(detectionData.eyeRight[0], detectionData.eyeRight[1], 10, 0, 2 * Math.PI);
    ctx.fillStyle = 'yellow';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(detectionData.eyeLeft[0], detectionData.eyeLeft[1], 10, 0, 2 * Math.PI);
    ctx.fillStyle = 'pink';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(detectionData.nose[0], detectionData.nose[1], 10, 0, 2 * Math.PI);
    ctx.fillStyle = 'cyan';
    ctx.fill();

    ctx.save();
    ctx.beginPath();
    ctx.translate(detectionData.x + detectionData.width / 2, detectionData.y + detectionData.height / 2);
    ctx.rotate(detectionData.roll);
    ctx.rect(-detectionData.width / 2, -detectionData.height / 2, detectionData.width, detectionData.height);
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 5;
    ctx.stroke();
    ctx.restore();

    ctx.restore();
    return baseCanvas;
};

const outputFace = async (baseCanvas, faceTransforms) => {
    const avatar = await loadImage('test/head_t.png');
    const ctx = baseCanvas.getContext('2d');

    ctx.save();
    ctx.globalAlpha = 0.75;
    ctx.rect(0, 0, baseCanvas.width, baseCanvas.height);
    ctx.fillStyle = '#000';
    ctx.fill();
    ctx.restore();

    ctx.save();
    ctx.globalAlpha = 0.75;
    ctx.translate(
        (baseCanvas.width / 2) + faceTransforms.translate.x + DEFAULT_TRANSFORMS.translate.x,
        (baseCanvas.height / 2) + faceTransforms.translate.y + DEFAULT_TRANSFORMS.translate.y,
    );
    ctx.rotate(faceTransforms.rotate + DEFAULT_TRANSFORMS.rotate);
    ctx.scale(faceTransforms.scale * DEFAULT_TRANSFORMS.scale, faceTransforms.scale * DEFAULT_TRANSFORMS.scale);
    ctx.drawImage(
        avatar,
        (-avatar.width / 2),
        (-avatar.height / 2),
    )
    ctx.restore();

    return baseCanvas;
};

const main = async () => {
    // Load the nets we need
    await loadNets();

    log('Loading baseline face...');
    const baselineFace = await loadImage('test/default.jpg');
    const baselineData = await processImage(baselineFace);
    const baselineTransforms = transforms(baselineData, baselineData); // All zeros
    const baselineOutput = plottedFace(await outputFace(faceapi.createCanvasFromMedia(baselineFace), baselineTransforms), baselineData);
    await writeFile('test/output-default.jpg', baselineOutput.toBuffer('image/jpeg'));

    log('Loading rotated face...');
    const rotatedFace = await loadImage('test/rotated.jpg');
    const rotatedData = await processImage(rotatedFace);
    const rotatedTransforms = transforms(baselineData, rotatedData);
    const rotatedOutput = plottedFace(await outputFace(faceapi.createCanvasFromMedia(rotatedFace), rotatedTransforms), rotatedData);
    await writeFile('test/output-rotated.jpg', rotatedOutput.toBuffer('image/jpeg'));

    log('Done!');
};

main();
