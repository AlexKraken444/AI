const test = require("node:test");
const assert = require("node:assert/strict");
const models = require("../public/models.js");

test("existing generative mode upgrades without mutating chats or memory", () => {
  const state = {modelMode: "context3", chats: [{id: "old", messages: []}], memory: {facts: [{text: "Имя: Алекс"}]}};
  const before = JSON.stringify(state);
  assert.equal(models.restore(state), "local");
  assert.equal(JSON.stringify(state), before);
});
test("new mode choices persist after the one-time upgrade", () => {
  for (const modelMode of Object.keys(models.names)) assert.equal(models.restore({modelMode, modelRevision: 5}), modelMode);
  assert.equal(models.restore({modelMode: "reference"}), "local");
});
test("fresh or invalid settings use the new model", () => {
  assert.equal(models.restore(null), "local");
  assert.equal(models.restore({modelMode: "unknown", modelRevision: 5}), "local");
  assert.equal(models.restore({modelMode: "toString", modelRevision: 5}), "local");
});
