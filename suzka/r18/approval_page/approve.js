// SUZKA R18 approval page (#411 D1 r7 §5).  Static, code-generated, with no
// HTML-parsing sink at all: every value is rendered with textContent after a
// length cap.  The digests and the code are
// recomputed here (WebCrypto SHA-256) from the canonical bytes the server
// returns; a mismatch aborts and sends nothing.
(function () {
  "use strict";

  var CHALLENGE_DOMAIN = "PROJECT-SUZKA:R18:APPROVAL-CHALLENGE:V1\0";
  var CODE_DOMAIN = "PROJECT-SUZKA:R18:CONFIRMATION-CODE:V1\0";
  var INTENT_DOMAIN = "PROJECT-SUZKA:R18:INTENT:V1\0";
  var ARTIFACT_DOMAIN = "PROJECT-SUZKA:R18:REVIEW-ARTIFACT:V1\0";
  var CODE_LENGTH = 8;
  var MAX_TEXT = 200;
  var HEADERS = { "X-SUZKA-R18": "1" };
  var B32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

  function utf8(text) { return new TextEncoder().encode(text); }

  function concat(a, b) {
    var out = new Uint8Array(a.length + b.length);
    out.set(a, 0);
    out.set(b, a.length);
    return out;
  }

  function hex(buffer) {
    var bytes = new Uint8Array(buffer);
    var text = "";
    for (var i = 0; i < bytes.length; i++) {
      text += (bytes[i] < 16 ? "0" : "") + bytes[i].toString(16);
    }
    return text;
  }

  function sha256(bytes) { return crypto.subtle.digest("SHA-256", bytes); }

  // The same canonical JSON as the server: sorted keys, no spaces, integers only.
  function canon(value) {
    if (value === null) { return "null"; }
    if (typeof value === "string") { return JSON.stringify(value); }
    if (typeof value === "number") {
      if (!Number.isSafeInteger(value)) { throw new Error("not canonical"); }
      return String(value);
    }
    if (typeof value === "boolean") { return value ? "true" : "false"; }
    if (Array.isArray(value)) { return "[" + value.map(canon).join(",") + "]"; }
    if (typeof value === "object") {
      return "{" + Object.keys(value).sort().map(function (key) {
        return JSON.stringify(key) + ":" + canon(value[key]);
      }).join(",") + "}";
    }
    throw new Error("not canonical");
  }

  // The first CODE_LENGTH characters of RFC 4648 base32 = the first 40 bits.
  function codeFrom(buffer) {
    var bytes = new Uint8Array(buffer);
    var bits = 0, acc = 0, out = "";
    for (var i = 0; i < 5; i++) {
      acc = (acc << 8) | bytes[i];
      bits += 8;
      while (bits >= 5 && out.length < CODE_LENGTH) {
        out += B32.charAt((acc >> (bits - 5)) & 31);
        bits -= 5;
      }
      acc &= (1 << bits) - 1;
    }
    return out;
  }

  async function digestOf(domain, text) {
    return hex(await sha256(concat(utf8(domain), utf8(text))));
  }

  // Verify challenge <-> intent <-> forecast and recompute the digests and
  // the code.  Returns {ok, problems, challenge, intent, code, ...}.
  async function verify(reply) {
    var problems = [];
    var challenge = JSON.parse(reply.challenge);
    var intent = JSON.parse(reply.intent);
    if (canon(challenge) !== reply.challenge) { problems.push("challenge bytes are not canonical"); }
    if (canon(intent) !== reply.intent) { problems.push("intent bytes are not canonical"); }
    var challengeDigest = await digestOf(CHALLENGE_DOMAIN, reply.challenge);
    var body = Object.assign({}, intent);
    delete body.intent_digest;
    delete body.intent_id;
    var intentDigest = await digestOf(INTENT_DOMAIN, canon(body));
    var payloadDigest = await digestOf(ARTIFACT_DOMAIN, canon(intent.artifact));
    var code = codeFrom(await sha256(concat(utf8(CODE_DOMAIN), utf8(challengeDigest))));
    function same(name, a, b) { if (a !== b) { problems.push(name + " differs"); } }
    same("intent_digest", intent.intent_digest, intentDigest);
    same("challenge intent_digest", challenge.intent_digest, intentDigest);
    same("challenge_digest", reply.challenge_digest, challengeDigest);
    same("intent_id", challenge.intent_id, intent.intent_id);
    same("operation_key", challenge.operation_key, intent.forward_key);
    same("payload_digest", challenge.payload_digest, payloadDigest);
    same("target_instance_id", challenge.target_instance_id, intent.target_instance_id);
    same("target_expected_revision", challenge.target_expected_revision, intent.target_expected_revision);
    same("expectation_digest", challenge.expectation_digest, reply.forecast_digest);
    same("direction", challenge.direction, "FORWARD");
    same("window_end", reply.window_end_utc, intent.approval_window_end_utc);
    if (challenge.expires_utc > intent.approval_window_end_utc) { problems.push("challenge outlives the window"); }
    if (challenge.commitment_deadline_utc !== null && challenge.expires_utc > challenge.commitment_deadline_utc) {
      problems.push("challenge outlives the Commitment deadline");
    }
    return {
      ok: problems.length === 0,
      problems: problems,
      challenge: challenge,
      intent: intent,
      code: code,
      challengeDigest: challengeDigest
    };
  }

  function clip(value) {
    var text = String(value);
    return text.length > MAX_TEXT ? text.slice(0, MAX_TEXT) + "..." : text;
  }

  function el(tag, text, className) {
    var node = document.createElement(tag);
    if (text !== undefined) { node.textContent = clip(text); }
    if (className) { node.className = className; }
    return node;
  }

  function row(table, label, value) {
    var tr = document.createElement("tr");
    tr.appendChild(el("th", label));
    // Digests and ids are shown whole: never truncated (only status text is).
    var cell = document.createElement("td");
    cell.textContent = value === null || value === undefined ? "none" : String(value);
    tr.appendChild(cell);
    table.appendChild(tr);
  }

  function setStatus(text, className) {
    var node = document.getElementById("status");
    node.textContent = clip(text);
    node.className = className || "note";
  }

  async function request(method, path, body) {
    var options = { method: method, headers: Object.assign({}, HEADERS), credentials: "same-origin" };
    if (body !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    var response = await fetch(path, options);
    var data = null;
    try { data = await response.json(); } catch (error) { data = null; }
    return { status: response.status, data: data };
  }

  function path(intentId, action) {
    return "/api/r18/approvals/" + encodeURIComponent(intentId) + "/" + action;
  }

  async function review(intentId) {
    var card = document.getElementById("card");
    card.hidden = true;
    card.textContent = "";
    var result = await request("POST", path(intentId, "review"), {});
    if (result.status !== 200 || !result.data || result.data.status === "hold") {
      setStatus("Not reviewable now: " + (result.data && result.data.reason ? result.data.reason : "http " + result.status), "abort");
      return;
    }
    var checked;
    try { checked = await verify(result.data); } catch (error) {
      setStatus("ABORT: the server's records could not be checked. Nothing was sent.", "abort");
      return;
    }
    var challenge = checked.challenge;
    var intent = checked.intent;
    var table = document.createElement("table");
    row(table, "Operation", "write one review artifact (forward)");
    row(table, "Intent id", intent.intent_id);
    row(table, "Intent digest", intent.intent_digest);
    row(table, "Decision id", intent.decision_id);
    row(table, "Decision digest", intent.decision_committed_digest);
    var sources = intent.artifact.sources;
    for (var i = 0; i < sources.length; i++) {
      row(table, "Source " + sources[i].role, sources[i].ref_id + " r" + sources[i].revision + " " + sources[i].digest);
    }
    var gates = intent.artifact.gates;
    for (var j = 0; j < gates.length; j++) {
      row(table, "Gate " + gates[j].gate, gates[j].status);
    }
    row(table, "Target", intent.target_namespace + " " + intent.target_instance_id + " (revision " + intent.target_expected_revision + ")");
    row(table, "Payload digest", challenge.payload_digest);
    row(table, "Forecast digest", challenge.expectation_digest);
    row(table, "Commitment deadline", challenge.commitment_deadline_utc);
    row(table, "Approval window ends", intent.approval_window_end_utc);
    row(table, "This code expires", challenge.expires_utc);
    row(table, "Attempt", "attempt " + result.data.attempt + "/" + result.data.max_attempts);
    row(table, "Challenge digest", checked.challengeDigest);
    card.appendChild(table);
    if (!checked.ok) {
      card.appendChild(el("p", "ABORT: " + checked.problems.join("; ") + ". Nothing was sent.", "abort"));
      card.hidden = false;
      return;
    }
    card.appendChild(el("p", "Checks passed (digests recomputed in this browser). Type this code to approve:"));
    card.appendChild(el("p", checked.code, "code"));
    var input = document.createElement("input");
    input.type = "text";
    input.autocomplete = "off";
    input.maxLength = CODE_LENGTH;
    input.setAttribute("aria-label", "confirmation code");
    var button = el("button", "Approve");
    button.type = "button";
    button.addEventListener("click", async function () {
      var typed = input.value.trim().toUpperCase();
      if (typed !== checked.code) {
        setStatus("The typed code does not match; nothing was sent.", "abort");
        return;
      }
      button.disabled = true;
      var done = await request("POST", path(intentId, "confirm"), { nonce: challenge.nonce, code: typed });
      if (done.status === 200 && done.data && done.data.status === "claimed") {
        setStatus("Approved: claim " + done.data.claim_id + " recorded. No effect has been executed.", "ok");
        card.hidden = true;
        load();
      } else {
        setStatus("Not approved: " + (done.data && done.data.reason ? done.data.reason : "http " + done.status), "abort");
        button.disabled = false;
      }
    });
    card.appendChild(input);
    card.appendChild(button);
    card.hidden = false;
    setStatus("Review the fields above, then type the code.");
  }

  async function load() {
    var list = document.getElementById("list");
    list.textContent = "";
    var result = await request("GET", "/api/r18/approvals");
    if (result.status !== 200 || !result.data || !Array.isArray(result.data.approvals)) {
      setStatus("Could not list approvals: " + (result.data && result.data.reason ? result.data.reason : "http " + result.status), "abort");
      return;
    }
    if (result.data.approvals.length === 0) {
      setStatus("Nothing is waiting for approval.");
      return;
    }
    setStatus("Waiting for approval:");
    result.data.approvals.forEach(function (item) {
      var table = document.createElement("table");
      row(table, "Intent", item.intent_id);
      row(table, "Window ends", item.window_end_utc);
      row(table, "Attempts used", item.attempts_used + "/3");
      row(table, "Live challenge", item.live_challenge ? "yes, until " + item.challenge_expires_utc : "no");
      row(table, "Commitment", item.commitment_id + (item.commitment_deadline_utc ? " (deadline " + item.commitment_deadline_utc + ")" : ""));
      var button = el("button", "Review");
      button.type = "button";
      button.addEventListener("click", function () { review(item.intent_id); });
      list.appendChild(table);
      list.appendChild(button);
    });
  }

  if (typeof document !== "undefined" && document.getElementById("list")) {
    load();
  }
  if (typeof globalThis !== "undefined") {
    globalThis.__suzkaApproval = { verify: verify, canon: canon, codeFrom: codeFrom };
  }
})();
