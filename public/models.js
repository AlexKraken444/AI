"use strict";
(function (root) {
  const names = Object.freeze({context4: "Kraken Context 4", context3: "Kraken Context 3", context: "Kraken Context 2", reference: "Kraken Mini"});
  const revision = 4;
  function restore(state) {
    if (state?.modelMode === "reference") return "reference";
    if (state?.modelRevision === revision && Object.hasOwn(names, state.modelMode)) return state.modelMode;
    return "context4";
  }
  const api = Object.freeze({names, revision, restore, defaultMode: "context4"});
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.KrakenModels = api;
})(globalThis);
