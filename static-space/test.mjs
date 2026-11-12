import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { example, parseAccount, predict } from "./predict.mjs";

const model = JSON.parse(readFileSync(new URL("./model.json", import.meta.url), "utf8"));
const result = predict(parseAccount(example), model);
assert.ok(Math.abs(result.score - 0.28784664907587476) < 1e-10);
assert.equal(result.prediction, 0);
assert.throws(() => parseAccount("LIMIT_BAL\n123"));
console.log("Sample prediction passed:", result);
