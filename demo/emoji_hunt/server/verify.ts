import {OAuth2Client} from 'google-auth-library';
const client = new OAuth2Client('834911136599-o3feieivbdf7kff50hjn1tnfmkv4noqo.apps.googleusercontent.com');

export async function verify(token: string) {
  const ticket = await client.verifyIdToken({
      idToken: token,
      audience: '834911136599-o3feieivbdf7kff50hjn1tnfmkv4noqo.apps.googleusercontent.com',  // Specify the CLIENT_ID of the app that accesses the backend
      // Or, if multiple clients access the backend:
      //[CLIENT_ID_1, CLIENT_ID_2, CLIENT_ID_3]
  });
  const payload = ticket.getPayload();
  const userid = payload['sub'];
  return userid;
}
