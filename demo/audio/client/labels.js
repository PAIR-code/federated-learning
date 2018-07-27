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

import * as tf from '@tensorflow/tfjs';

const firstPrompt = 'Would you be willing to help me?' +
    ' I\'d love it if you could show me how to pronounce the word:';
const laterPrompt = 'If you\'re up for another, ' +
    'could you show me how to pronounce:';

let thanksVariants = [];
const basicThanks = ['Thanks!', 'Gracias!', 'Much obliged!', 'Bravo!'];
thanksVariants = thanksVariants.concat(basicThanks);
thanksVariants = thanksVariants.concat(basicThanks);
thanksVariants.push('Not bad!');
thanksVariants = thanksVariants.concat(basicThanks);
thanksVariants.push('You\'re getting good at this!');
thanksVariants = thanksVariants.concat(basicThanks);
thanksVariants = thanksVariants.concat(basicThanks);
thanksVariants.push('That was a lot, maybe you should take a break!');

const suggestedLabel = document.getElementById('suggested-label');
const introText = document.getElementById('intro-text');

// Ask user to provide audio for all labels, but in a random order
/// const randomLabels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
const randomLabels = [0, 1, 2, 3];//, 3, 4];//, 5, 6, 7, 8, 9, 10, 11, 12, 13];
tf.util.shuffle(randomLabels);

let labelIdx = 0;
let numAsked = 0;

//export const labelNames = [
//  'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
//  'zero', 'left', 'right', 'go', 'stop'
//];
export const labelNames = [
  'accio', 'expelliarmus', 'lumos', 'nox'
];

export function getNextLabel() {
  const label = randomLabels[labelIdx];
  suggestedLabel.innerText = '"' + labelNames[label] + '"';
  if (numAsked == 0) {
    introText.innerText = firstPrompt;
  } else {
    const thanks = thanksVariants[numAsked % thanksVariants.length];
    introText.innerText = thanks + ' ' + laterPrompt;
  }
  numAsked++;
  labelIdx += 1;
  if (labelIdx >= labelNames.length) {
    labelIdx = 0;
    tf.util.shuffle(randomLabels);  // reshuffle each iteration
  }
  return label;
}
