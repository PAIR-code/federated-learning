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

const canvas = document.createElement('canvas');
const drawCtx = canvas.getContext('2d');

async function captureFrame(video) {
  canvas.width = video.videoWidth || video.width;
  canvas.height = video.videoHeight || video.height;
  drawCtx.drawImage(video, 0, 0);
  return new Promise((res, _) => canvas.toBlob(res, 'image/png'))
}

export async function upload(url, target, webcam) {
  const blob = await captureFrame(webcam);
  const file = new File([blob], `data_${target}_${Date.now()}.png`, {type: 'image/png'});

  const req = new XMLHttpRequest();
  req.open('POST', url, true);

  const formData = new FormData();
  formData.append('file', file)

  req.send(formData);

  return new Promise((res, rej) => {
    req.onreadystatechange = () => {
      if(req.readyState === XMLHttpRequest.DONE) {
        if(req.status === 200) {
          res();
        } else {
          rej(req.status);
        }
      }
    }
  });
}
