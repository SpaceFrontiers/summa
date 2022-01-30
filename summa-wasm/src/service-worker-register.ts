(() => {
  // You can customize the behavior of this script through a global `coi` variable.
  const coi = {
    shouldRegister: () => true,
    shouldDeregister: () => false,
    doReload: () => window.location.reload(),
  };


  const n = navigator;
  if (coi.shouldDeregister() && n.serviceWorker && n.serviceWorker.controller) {
    n.serviceWorker.controller.postMessage({ type: "deregister" });
  }

  // If we're already coi: do nothing. Perhaps it's due to this script doing its job, or COOP/COEP are
  // already set from the origin server. Also if the browser has no notion of crossOriginIsolated, just give up here.
  if (window.crossOriginIsolated || !coi.shouldRegister()) return;

  if (!window.isSecureContext) {
    return;
  }

  // In some environments (e.g. Chrome incognito mode) this won't be available
  if (n.serviceWorker) {
    n.serviceWorker
      .register("/service-worker.js")
      .then(
        (registration) => {
          registration.addEventListener("updatefound", () => {
            coi.doReload();
          });
          // If the registration is active, but it's not controlling the page
          if (registration.active && !n.serviceWorker.controller) {
            coi.doReload();
          }
          n.serviceWorker.controller.postMessage({type: "ipfs"})
        },
        (err) => {
          console.error("COOP/COEP Service Worker failed to register:", err);
        }
      );
  }
})();
