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

module.exports.AVATAR = imageFromFile(join(__dirname, 'avatar.png'));
module.exports.AVATAR_GHOST = imageFromFile(join(__dirname, 'avatar_ghost.png'));

const getTop = data => data
    .map((a) => a.y)
    .reduce((a, b) => Math.min(a, b));

const getMeanPosition = data => data
    .map((a) => [a.x, a.y])
    .reduce((a, b) => [a[0] + b[0], a[1] + b[1]])
    .map(a => a / data.length);

const getLowestPosition = data => data
    .map((a) => [a.x, a.y])
    .sort((a, b) => b[1] - a[1])[0];

const getDistance = (fromData, toData) => [toData[0] - fromData[0], toData[1] - fromData[1]];

const getAbsDistance = (fromData, toData) => {
    const distance = getDistance(fromData, toData);
    return Math.sqrt(Math.pow(distance[0], 2) + Math.pow(distance[1], 2))
};

module.exports.loadNets = async () => {
    if (!USE_TINY) await faceapi.nets.ssdMobilenetv1.loadFromDisk('./weights');
    if (USE_TINY) await faceapi.nets.tinyFaceDetector.loadFromDisk('./weights');
    if (!USE_TINY) await faceapi.nets.faceLandmark68Net.loadFromDisk('./weights');
    if (USE_TINY) await faceapi.nets.faceLandmark68TinyNet.loadFromDisk('./weights');
    // await faceapi.nets.faceExpressionNet.loadFromDisk('./weights');
};

const faceDetectionOpts = () => USE_TINY
    ? new faceapi.TinyFaceDetectorOptions({ scoreThreshold: SCORE_MIN })
    : new faceapi.SsdMobilenetv1Options({ minConfidence: SCORE_MIN });

module.exports.processImage = async img => {
    // Perform the detection
    const detectStart = Date.now();
    const detection = await faceapi
        .detectSingleFace(img, faceDetectionOpts())
        .withFaceLandmarks(USE_TINY);
        // .withFaceExpressions();
    const detectEnd = Date.now();

    // Failed to detect
    if (!detection || !detection.detection || !detection.landmarks) {
        return {
            score: detection && detection.detection ? detection.detection.score : 0,
            duration: detectEnd - detectStart,
            success: false,
        };
    }

    // Get positions and rotations
    const eyeRight = getMeanPosition(detection.landmarks.getRightEye());
    const eyeLeft = getMeanPosition(detection.landmarks.getLeftEye());
    const eyeMidpoint = [(eyeLeft[0] + eyeRight[0]) / 2, (eyeLeft[1] + eyeRight[1]) / 2];
    const nose = getMeanPosition(detection.landmarks.getNose());
    const mouth = getMeanPosition(detection.landmarks.getMouth());
    const jaw = getLowestPosition(detection.landmarks.getJawOutline());
    const pitch = (getTop(detection.landmarks.getJawOutline()) - mouth[1]) / detection.detection.box.height + 0.5;
    const yaw = (eyeLeft[0] + (eyeRight[0] - eyeLeft[0]) / 2 - nose[0]) / detection.detection.box.width;
    const rawRoll = Math.atan2(eyeMidpoint[1] - nose[1], eyeMidpoint[0] - nose[0]);
    const roll = rawRoll < 0 ? (2 * Math.PI) - Math.abs(rawRoll) : rawRoll;

    // Get the center
    const center = {
        x: detection.detection.box.x + (detection.detection.box.width / 2),
        y: detection.detection.box.y + (detection.detection.box.height / 2),
    };
    // const center = {
    //     x: nose[0],
    //     y: nose[1],
    // };

    // Get the corners
    const corners = [
        { // top left
            x: detection.detection.box.x,
            y: detection.detection.box.y,
        },
        { // top right
            x: detection.detection.box.x + detection.detection.box.width,
            y: detection.detection.box.y,
        },
        { // bottom right
            x: detection.detection.box.x + detection.detection.box.width,
            y: detection.detection.box.y + detection.detection.box.height,
        },
        { // bottom left
            x: detection.detection.box.x,
            y: detection.detection.box.y + detection.detection.box.height,
        },
    ];

    // Rotate a point about the center
    const rotatedPoint = ({ x, y }) => {
        // translate point to origin
        const tempX = x - center.x;
        const tempY = y - center.y;

        // now apply rotation
        const rotatedX = tempX * Math.cos(roll) - tempY * Math.sin(roll);
        const rotatedY = tempX * Math.sin(roll) + tempY * Math.cos(roll);

        // translate back
        return { x: rotatedX + center.x, y: rotatedY + center.y };
    };

    // Output data
    return {
        score: detection.detection.score,
        duration: detectEnd - detectStart,
        success: true,
        width: detection.detection.box.width,
        height: detection.detection.box.height,
        center,
        corners,
        eyeRight,
        eyeLeft,
        eyeMidpoint,
        nose,
        mouth,
        jaw,
        jawPoints: detection.landmarks.getJawOutline(),
        yaw,
        pitch,
        roll,
        rolledCorners: corners.map(corner => rotatedPoint(corner)),
    };
};

module.exports.plottedFace = (baseCanvas, detectionData) => {
    if (!detectionData.success) return baseCanvas;

    const ctx = baseCanvas.getContext('2d');
    ctx.save();

    // Original box
    ctx.beginPath();
    ctx.moveTo(detectionData.corners[0].x, detectionData.corners[0].y);
    ctx.lineTo(detectionData.corners[1].x, detectionData.corners[1].y);
    ctx.lineTo(detectionData.corners[2].x, detectionData.corners[2].y);
    ctx.lineTo(detectionData.corners[3].x, detectionData.corners[3].y);
    ctx.closePath();
    ctx.strokeStyle = '#f00';
    ctx.lineWidth = 5;
    ctx.stroke();

    // Rotated box
    ctx.beginPath();
    ctx.moveTo(detectionData.rolledCorners[0].x, detectionData.rolledCorners[0].y);
    ctx.lineTo(detectionData.rolledCorners[1].x, detectionData.rolledCorners[1].y);
    ctx.lineTo(detectionData.rolledCorners[2].x, detectionData.rolledCorners[2].y);
    ctx.lineTo(detectionData.rolledCorners[3].x, detectionData.rolledCorners[3].y);
    ctx.closePath();
    ctx.strokeStyle = '#00f';
    ctx.lineWidth = 5;
    ctx.stroke();

    // Old rotated box (how avatar is applied)
    // ctx.save();
    // ctx.translate(detectionData.corners[0].x + detectionData.width / 2, detectionData.corners[0].y + detectionData.height / 2);
    // ctx.rotate(detectionData.roll);
    // ctx.rect(-detectionData.width / 2, -detectionData.height / 2, detectionData.width, detectionData.height);
    // ctx.strokeStyle = '#00f';
    // ctx.lineWidth = 5;
    // ctx.stroke();
    // ctx.restore();

    // Eye points
    ctx.beginPath();
    ctx.arc(detectionData.eyeMidpoint[0], detectionData.eyeMidpoint[1], 10, 0, 2 * Math.PI);
    ctx.fillStyle = '#0f0';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(detectionData.eyeRight[0], detectionData.eyeRight[1], 10, 0, 2 * Math.PI);
    ctx.fillStyle = '#9f0';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(detectionData.eyeLeft[0], detectionData.eyeLeft[1], 10, 0, 2 * Math.PI);
    ctx.fillStyle = '#0f9';
    ctx.fill();

    // Nose
    ctx.beginPath();
    ctx.arc(detectionData.nose[0], detectionData.nose[1], 10, 0, 2 * Math.PI);
    ctx.fillStyle = '#0ff';
    ctx.fill();

    // Center
    ctx.beginPath();
    ctx.arc(detectionData.center.x, detectionData.center.y, 10, 0, 2 * Math.PI);
    ctx.fillStyle = '#9ff';
    ctx.fill();

    // Mouth
    ctx.beginPath();
    ctx.arc(detectionData.mouth[0], detectionData.mouth[1], 10, 0, 2 * Math.PI);
    ctx.fillStyle = '#ff0';
    ctx.fill();

    // Jaw/chin
    for (const point of detectionData.jawPoints) {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 10, 0, 2 * Math.PI);
        ctx.fillStyle = '#f9f';
        ctx.fill();
    }
    ctx.beginPath();
    ctx.arc(detectionData.jaw[0], detectionData.jaw[1], 10, 0, 2 * Math.PI);
    ctx.fillStyle = '#f0f';
    ctx.fill();

    // Neck "attachment" point (avg of bottom two corners)
    ctx.beginPath();
    ctx.arc(
        (detectionData.rolledCorners[0].x + detectionData.rolledCorners[3].x) / 2,
        (detectionData.rolledCorners[0].y + detectionData.rolledCorners[3].y) / 2,
        10, 0, 2 * Math.PI
    );
    ctx.fillStyle = '#f99';
    ctx.fill();

    ctx.restore();
    return baseCanvas;
};

const DEFAULT_TRANSFORMS = {
    scale: 0.9,
    translate: { x: -25, y: 50 },
    rotate: -0.15,
};


// TODO: These need to do a far far better job of warping the avatar onto the track of the face
// Does this even need a baseline, or do we just define a load of points in advance in the avatar itself somehow?

module.exports.transforms = (base, updated) => {
    try {
        // Calculate scale based on distance between eyes and nose
        const baseAvgSize = (getAbsDistance(base.eyeLeft, base.nose) + getAbsDistance(base.eyeRight, base.nose)) / 2;
        const updatedAvgSize = (getAbsDistance(updated.eyeLeft, updated.nose) + getAbsDistance(updated.eyeRight, updated.nose)) / 2;
        const scale = updatedAvgSize / baseAvgSize;

        // Calculate translation & rotation
        const translate = { x: updated.center.x - base.center.x, y: updated.center.y - base.center.y };
        const rotate = updated.roll - base.roll;

        return {
            scale,
            translate,
            rotate,
        };
    } catch (_) {
        // If anything goes wrong, default to no transforms
        return {
            scale: 1,
            translate: { x: 0, y: 0 },
            rotate: 0,
        };
    }
};

module.exports.outputFace = (baseCanvas, faceTransforms, faceImage, opacity) => {
    // Calculate base avatar scale
    const defaultScale = (baseCanvas.height * DEFAULT_TRANSFORMS.scale) / faceImage.height;

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
        faceImage,
        (-faceImage.width / 2),
        (-faceImage.height / 2),
    )
    ctx.restore();

    return baseCanvas;
};
