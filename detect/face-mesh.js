/*
    "@tensorflow-models/face-landmarks-detection": "0.0.2",
    "@tensorflow/tfjs-backend-cpu": "^2.8.2",
    "@tensorflow/tfjs-converter": "^2.8.2",
    "@tensorflow/tfjs-core": "^2.8.2",
 */

// THIS IS SUPER SLOW

const { SCORE_MIN } = require('./config');
const faceLandmarksDetection = require('@tensorflow-models/face-landmarks-detection');
require('@tensorflow/tfjs-backend-cpu');

let model;

module.exports.load = async () => {
    model = await faceLandmarksDetection.load(
        faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
        {
            maxFaces: 1,
            scoreThreshold: SCORE_MIN,
        },
    );
};

module.exports.detect = async img => {
    // Perform the detection
    const detectStart = Date.now();
    const prediction = await model.estimateFaces({
        input: img,
        returnTensors: false,
        flipHorizontal: false,
        predictIrises: false,
    });
    const detectEnd = Date.now();

    // Output data
    return {
        data: prediction[0],
        score: prediction.length > 0 ? prediction[0].faceInViewConfidence : 0,
        points: prediction.length > 0 ? prediction[0].scaledMesh.map(coords => coords.slice(0, 2)) : [],
        width: prediction.length > 0 ? prediction[0].boundingBox.bottomRight[0] - prediction[0].boundingBox.topLeft[0] : 0,
        height: prediction.length > 0 ? prediction[0].boundingBox.bottomRight[1] - prediction[0].boundingBox.topLeft[1] : 0,
        x: prediction.length > 0 ? prediction[0].boundingBox.topLeft[0] : 0,
        y: prediction.length > 0 ? prediction[0].boundingBox.topLeft[1] : 0,
        duration: detectEnd - detectStart,
        success: prediction.length > 0,
    };
};
