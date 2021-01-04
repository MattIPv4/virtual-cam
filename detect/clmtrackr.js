/*
    "clmtrackr": "^1.1.2",
 */

// THIS IS AWFUL AND DOES NOT WORK

const { SCORE_MIN } = require('./config');
const clm = require('clmtrackr');

module.exports.load = async () => {};

module.exports.detect = async img => {
    // Perform the detection
    const detectStart = Date.now();

    // Create the tracker
    const ctracker = new clm.tracker({
        stopOnConvergence: true,
        scoreThreshold: SCORE_MIN,
        maxIterationsPerAnimFrame: 1000,
    });
    ctracker.init();

    // Listen for all the events it can emit
    document.addEventListener('clmtrackrIteration', () => {
        console.log('iteration');
    }, false);
    const promises = [
        new Promise(resolve => {
            document.addEventListener('clmtrackrNotFound', () => {
                ctracker.stop();
                resolve('not found');
            }, false);
        }),
        new Promise(resolve => {
            document.addEventListener('clmtrackrLost', () => {
                ctracker.stop();
                resolve('lost');
            }, false);
        }),
        new Promise(resolve => {
            document.addEventListener('clmtrackrConverged', () => {
                ctracker.stop();
                resolve('converged');
            }, false);
        }),
    ];

    // Run until it finds a result
    ctracker.start(img);
    console.log(await Promise.any(promises));
    const detection = ctracker.getCurrentPosition();
    const detectEnd = Date.now();

    // TODO: Output data
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

