import * as es6Promise from 'es6-promise';
es6Promise.polyfill();

eval(`
var realFetch = require('node-fetch');

if (!global.fetch) {
	global.fetch = realFetch;
	global.Response = realFetch.Response;
	global.Headers = realFetch.Headers;
	global.Request = realFetch.Request;
}
`);
