"use strict";
(function(root) {
 const names=Object.freeze({context4:"Kraken Context 4 · эксперимент",context3:"Kraken Context 3 · эксперимент",context:"Kraken Context 2 · эксперимент",reference:"Kraken Mini · справочник"});
 const revision=7;
 function restore(state) {return state?.modelRevision===revision && Object.hasOwn(names,state.modelMode) ? state.modelMode : "context4";}
 const api=Object.freeze({names,revision,restore,defaultMode:"context4"});
 if(typeof module!=="undefined" && module.exports) module.exports=api; else root.KrakenModels=api;
})(globalThis);
