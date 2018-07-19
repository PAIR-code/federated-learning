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
