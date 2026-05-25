// On Vercel (HTTPS): use same-origin proxy (see frontend/vercel.json rewrites).
// On localhost: call the droplet API directly.
(function () {
  var host = typeof location !== 'undefined' ? location.hostname : '';
  if (host === 'localhost' || host === '127.0.0.1') {
    window.BACKEND_URL = 'http://147.182.173.184:3004';
  } else {
    window.BACKEND_URL = '';
  }
})();
