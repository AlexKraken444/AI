"use strict";
(function (root) {
  const names = Object.freeze({local: "Kraken Qwen3 4B · GPU", context4: "Kraken Context 4", context3: "Kraken Context 3", context: "Kraken Context 2", reference: "Kraken Mini"});
  const revision = 5;
  function restore(state) {
    if (state?.modelRevision === revision && Object.hasOwn(names, state.modelMode)) return state.modelMode;
    return "local";
  }
  const api = Object.freeze({names, revision, restore, defaultMode: "local"});
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.KrakenModels = api;
})(globalThis);
