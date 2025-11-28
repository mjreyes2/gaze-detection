import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";

let model, video, rafID;
let amountStraightEvents = 0;
const VIDEO_SIZE = 500;
let positionXLeftIris;
let positionYLeftIris;
let event;

const normalize = (val, max, min) =>
  Math.max(0, Math.min(1, (val - min) / (max - min)));

const isFaceRotated = (landmarks) => {
  const leftCheek = landmarks.leftCheek;
  const rightCheek = landmarks.rightCheek;
  const midwayBetweenEyes = landmarks.midwayBetweenEyes;

  const xPositionLeftCheek = video.width - leftCheek[0][0];
  const xPositionRightCheek = video.width - rightCheek[0][0];
  const xPositionMidwayBetweenEyes = video.width - midwayBetweenEyes[0][0];

  const widthLeftSideFace = xPositionMidwayBetweenEyes - xPositionLeftCheek;
  const widthRightSideFace = xPositionRightCheek - xPositionMidwayBetweenEyes;

  const difference = widthRightSideFace - widthLeftSideFace;

  if (widthLeftSideFace < widthRightSideFace && Math.abs(difference) > 5) {
    return true;
  } else if (
    widthLeftSideFace > widthRightSideFace &&
    Math.abs(difference) > 5
  ) {
    return true;
  }
  return false;
};

async function renderPrediction() {
  const predictions = await model.estimateFaces({
    input: video,
    returnTensors: false,
    flipHorizontal: false,
    predictIrises: true,
  });

  if (predictions.length > 0) {
    let normalizedX = 0;
    let normalizedY = 0;

    predictions.forEach((prediction) => {
      positionXLeftIris = prediction.annotations.leftEyeIris[0][0];
      positionYLeftIris = prediction.annotations.leftEyeIris[0][1];

      const faceBottomLeftX =
        video.width - prediction.boundingBox.bottomRight[0]; // face is flipped horizontally so bottom right is actually bottom left.
      const faceBottomLeftY = prediction.boundingBox.bottomRight[1];

      const faceTopRightX = video.width - prediction.boundingBox.topLeft[0]; // face is flipped horizontally so top left is actually top right.
      const faceTopRightY = prediction.boundingBox.topLeft[1];

      if (faceBottomLeftX > 0 && !isFaceRotated(prediction.annotations)) {
        const positionLeftIrisX = video.width - positionXLeftIris;
        const normalizedXIrisPosition = normalize(
          positionLeftIrisX,
          faceTopRightX,
          faceBottomLeftX
        );

        const normalizedYIrisPosition = normalize(
          positionYLeftIris,
          faceTopRightY,
          faceBottomLeftY
        );

        // Convert normalized positions (0-1) to range (-1 to 1) for eye movement
        // Center is 0.335 for X, 0.5 for Y
        normalizedX = (normalizedXIrisPosition - 0.335) * 3.0; // Scale for visibility
        normalizedY = (normalizedYIrisPosition - 0.5) * 2.0;

        // Clamp to -1 to 1 range
        normalizedX = Math.max(-1, Math.min(1, normalizedX));
        normalizedY = Math.max(-1, Math.min(1, normalizedY));

        // Return continuous coordinates instead of discrete directions
        event = { x: normalizedX, y: normalizedY };
      }
    });

    return event;
  }
  return null;
}

const loadModel = async () => {
  await tf.setBackend("webgl");

  model = await faceLandmarksDetection.load(
    faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
    { maxFaces: 1 }
  );
};

const setUpCamera = async (videoElement, webcamId = undefined) => {
  video = videoElement;
  const mediaDevices = await navigator.mediaDevices.enumerateDevices();

  const defaultWebcam = mediaDevices.find(
    (device) =>
      device.kind === "videoinput" && device.label.includes("Built-in")
  );

  const cameraId = defaultWebcam ? defaultWebcam.deviceId : webcamId;

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      deviceId: cameraId,
      width: VIDEO_SIZE,
      height: VIDEO_SIZE,
    },
  });

  video.srcObject = stream;
  video.play();
  video.width = 500;
  video.height = 500;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
};

const gaze = {
  loadModel: loadModel,
  setUpCamera: setUpCamera,
  getGazePrediction: renderPrediction,
};

export default gaze;
