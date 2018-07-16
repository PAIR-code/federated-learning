/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

function sel(str) {
  return document.querySelector(String.raw(...arguments));
}

const modelVersionElt = sel`#modelversion`
const statusElt = sel`#status`
const findMeElt = sel`#findme`
const overrideButtonElt = sel`#override`
const uploadAllowedElt = sel`#uploaddata`

let webcamSetup = false;

export function modelVersion(str) {
  modelVersionElt.innerText = str;
}

export function status(str) {
  statusElt.innerText = str;
}

export function findMe(str) {
  findMeElt.innerText = str;
}

export function uploadAllowed() {
  return uploadAllowedElt.checked;
}

export async function webcam() {
  const video = sel`#webcamvideo`;
  if(webcamSetup) {
    return video;
  }
  try {
    const stream =
      await navigator.mediaDevices.getUserMedia({audio: false, video: true});
    video.srcObject = stream;
    webcamSetup = true;
  } catch (exn) {
    status(`Error in accessing webcam: ${exn.toString()}`);
    throw exn;
  }

  return video;
}

export function overrideButton(handler) {
  return overrideButtonElt.addEventListener('click', handler);
}
