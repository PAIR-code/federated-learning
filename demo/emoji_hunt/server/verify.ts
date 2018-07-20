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

import {OAuth2Client} from 'google-auth-library';

const client = new OAuth2Client('834911136599-o3feieivbdf7kff50hjn1tnfmkv4noqo.apps.googleusercontent.com');

// from https://developers.google.com/identity/sign-in/web/backend-auth
export async function verify(token: string) {
  const ticket = await client.verifyIdToken({
      idToken: token,
      audience: '834911136599-o3feieivbdf7kff50hjn1tnfmkv4noqo.apps.googleusercontent.com',  // Specify the CLIENT_ID of the app that accesses the backend
  });
  const payload = ticket.getPayload();
  const userid = payload['sub'];
  return userid;
}
