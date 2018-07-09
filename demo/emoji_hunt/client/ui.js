function sel(str) {
  return document.querySelector(String.raw(...arguments));
}

const modelVersionElt = sel`#modelversion`
const statusElt = sel`#status`
const findMeElt = sel`#findme`
const overrideButtonElt = sel`#override`

export function modelVersion(str) {
  modelVersionElt.innerText = str;
}

export function status(str) {
  statusElt.innerText = str;
}

export function findMe(str) {
  findMeElt.innerText = str;
}

export async function webcam() {
  const video = sel`#webcamvideo`;
  try {
    const stream =
      await navigator.mediaDevices.getUserMedia({audio: false, video: true});
    video.srcObject = stream;
  } catch (exn) {
    status(`Error in accessing webcam: ${exn.toString()}`);
    throw exn;
  }


  return video;
}

export function overrideButton() {
  return overrideButtonElt;
}
