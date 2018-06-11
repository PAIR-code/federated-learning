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

import {Scalar, Tensor, Variable} from '@tensorflow/tfjs';
import {LayerVariable} from '@tensorflow/tfjs-layers/dist/variables';

export type LossFun = (inputs: Tensor, labels: Tensor) => Scalar;
export type PredFun = (inputs: Tensor) => Tensor|Tensor[];
export type VarList = Array<Variable|LayerVariable>;
export type ModelDict = {
  vars: VarList,
  loss: LossFun,
  predict: PredFun
};

export interface FederatedModel {
  setup(): Promise<ModelDict>;
}
