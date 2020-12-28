require('@tensorflow/tfjs-node');
const faceapi = require('face-api.js');
const { readFileSync } = require('fs');
const { join, extname } = require('path');

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

const USE_TINY = true;
const SCORE_MIN = 0.25;
const AVATAR = imageFromFile(join(__dirname, 'test', 'head_t.png'));

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

module.exports.loadNets = async () => {
    await faceapi.nets.ssdMobilenetv1.loadFromDisk('./weights');
    await faceapi.nets.tinyFaceDetector.loadFromDisk('./weights');
    await faceapi.nets.faceLandmark68Net.loadFromDisk('./weights');
    await faceapi.nets.faceLandmark68TinyNet.loadFromDisk('./weights');
    await faceapi.nets.faceExpressionNet.loadFromDisk('./weights');
};

const faceDetectionOpts = () => USE_TINY
    ? new faceapi.TinyFaceDetectorOptions({ scoreThreshold: SCORE_MIN })
    : new faceapi.SsdMobilenetv1Options({ minConfidence: SCORE_MIN });

module.exports.processImage = async img => {
    // Perform the detection
    const detectStart = Date.now();
    const detection = await faceapi
        .detectSingleFace(img, faceDetectionOpts())
        .withFaceLandmarks(USE_TINY)
        .withFaceExpressions();
    const detectEnd = Date.now();

    // Failed to detect
    if (!detection || !detection.detection || !detection.landmarks) {
        return {
            score: detection && detection.detection ? detection.detection.score : 0,
            duration: detectEnd - detectStart,
            success: false,
        };
    }

    // Get rotation
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
        duration: detectEnd - detectStart,
        success: true,
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

module.exports.plottedFace = (baseCanvas, detectionData) => {
    if (!detectionData.success) return baseCanvas;

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

const DEFAULT_TRANSFORMS = {
    scale: 0.9,
    translate: { x: -25, y: 50 },
    rotate: 0.15,
};

module.exports.transforms = (base, updated) => {
    try {
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
    } catch (_) {
        return {
            scale: 1,
            translate: { x: 0, y: 0 },
            rotate: 0,
        };
    }
};

module.exports.outputFace = (baseCanvas, faceTransforms, opacity) => {
    // Calculate base avatar scale
    const defaultScale = (baseCanvas.height * DEFAULT_TRANSFORMS.scale) / AVATAR.height;

    // Render the avatar
    const ctx = baseCanvas.getContext('2d');
    ctx.save();
    ctx.globalAlpha = opacity;
    ctx.translate(
        (baseCanvas.width / 2) + faceTransforms.translate.x + (DEFAULT_TRANSFORMS.translate.x * defaultScale),
        (baseCanvas.height / 2) + faceTransforms.translate.y + (DEFAULT_TRANSFORMS.translate.y * defaultScale),
    );
    ctx.rotate(faceTransforms.rotate + DEFAULT_TRANSFORMS.rotate);
    ctx.scale(faceTransforms.scale * defaultScale * -1, faceTransforms.scale * defaultScale);
    ctx.drawImage(
        AVATAR,
        (-AVATAR.width / 2),
        (-AVATAR.height / 2),
    )
    ctx.restore();

    return baseCanvas;
};
