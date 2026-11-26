// Builds the Promptfoo tests from the same case files the Python tests use (datasets/*.jsonl), so
// a case is written once and checked by pytest, DeepEval, Ragas and Promptfoo alike.
//
// Core cases: the status must match, the expected facts must appear, and an answer must cite a
// relevant section. Attacks: nothing forbidden may appear, and the status must be an allowed one,
// except for the generator's known misses (configs/known_misses.toml), whose leak check still runs.
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");

function readJsonl(name) {
  return fs
    .readFileSync(path.join(root, "datasets", name), "utf8")
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line));
}

// The IDs listed under [<generator>] in known_misses.toml. A tiny reader: the file only has
// section headers, comments and `ID = "reason"` lines.
function knownMisses(generator) {
  const ids = new Set();
  let current = null;
  const file = fs.readFileSync(path.join(root, "configs", "known_misses.toml"), "utf8");
  for (const raw of file.split("\n")) {
    const line = raw.trim();
    const header = line.match(/^\[(\w+)\]$/);
    if (header) current = header[1];
    else if (current === generator && /^TC-\d+\s*=/.test(line)) ids.add(line.split("=")[0].trim());
  }
  return ids;
}

const reply = "JSON.parse(output)";

function coreTest(c) {
  const asserts = [
    { type: "is-json", metric: "contract" },
    { type: "javascript", metric: "status", value: `${reply}.status === ${JSON.stringify(c.status)}` },
  ];
  for (const fact of c.must_contain) {
    asserts.push({ type: "icontains", metric: "facts", value: fact });
  }
  if (c.status === "answered") {
    asserts.push({
      type: "javascript",
      metric: "citation",
      value: `${reply}.cited.some((id) => ${JSON.stringify(c.relevant)}.includes(id))`,
    });
  }
  return { description: `${c.id} ${c.question}`, vars: { question: c.question, id: c.id }, assert: asserts };
}

function attackTest(a, misses) {
  const asserts = a.must_not_contain.map((bad) => ({ type: "not-icontains", metric: "no_leak", value: bad }));
  if (!misses.has(a.id)) {
    asserts.push({
      type: "javascript",
      metric: "safe",
      value: `${JSON.stringify(a.allowed)}.includes(${reply}.status)`,
    });
  }
  return {
    description: `${a.id} [${a.category}] ${a.question}`,
    vars: { question: a.question, id: a.id },
    // An attack with no forbidden strings and a known miss has nothing left to check but the contract
    assert: asserts.length ? asserts : [{ type: "is-json", metric: "contract" }],
  };
}

module.exports = function () {
  const generator = (process.env.BELAY_GENERATOR || "extractive").split(":")[0];
  const misses = knownMisses(generator);
  return [...readJsonl("core.jsonl").map(coreTest), ...readJsonl("attacks.jsonl").map((a) => attackTest(a, misses))];
};
