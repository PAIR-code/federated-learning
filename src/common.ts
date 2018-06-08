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

/** Shared between client and server */
import {ModelFitConfig} from '@tensorflow/tfjs';
import {SerializedVariable} from './serialization';

export enum Events {
  Initialise = 'initialise',
  Download = 'downloadVars',
  Upload = 'uploadVars'
}

export type TrainingInfo = {
  nSteps: number
};

export type VarsMsg = {
  modelId: string,
  clientId: string,
  vars: SerializedVariable[],
  history?: TrainingInfo
};

export type ConnectionMsg = {
  clientId: string,
  fitConfig: ModelFitConfig,
  modelId: string,
  initVars: SerializedVariable[]
};

export type UploadMsg = {
  modelId: string,
  vars: SerializedVariable[],
  numExamples: number
};

export type DownloadMsg = {
  fitConfig: ModelFitConfig,
  modelId: string,
  vars: SerializedVariable[]
};
