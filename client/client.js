console.log("client.js loaded OK");

const API_BASE = "http://127.0.0.1:8000";

const micBtn = document.getElementById("micBtn");
const btnSend = document.getElementById("btnSend");
const btnEnd = document.getElementById("btnEnd");
const textInput = document.getElementById("textInput");
const logEl = document.getElementById("log");
const player = document.getElementById("player");

let sessionId = null;

// ---------- UI helpers ----------
function log(msg) {
  logEl.textContent += "\n" + msg;
  logEl.scrollTop = logEl.scrollHeight;
}

async function postJSON(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res.json();
}

async function fetchTTSBlob(text) {
  const res = await fetch(`${API_BASE}/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (res.status === 204) return null;

  if (!res.ok) {
    const t = await res.text();
    throw new Error(`TTS failed: ${t}`);
  }
  return res.blob();
}

async function speak(text) {
  if (!text) return;
  const blob = await fetchTTSBlob(text);
  if (!blob) return;
  const url = URL.createObjectURL(blob);
  player.src = url;
  await player.play().catch(() => {});
}

function renderSummary(summaryObj) {
  if (!summaryObj) return;
  log("\n=== SESSION SUMMARY ===");
  log(`TITLE: ${summaryObj.title || ""}`);
  log(`${summaryObj.summary_text || ""}`);
  log("=== END SUMMARY ===\n");
}

// ---------- chat ----------
async function sendText(text) {
  const cleaned = (text || "").trim();
  if (!cleaned) return;

  log(`\nYou: ${cleaned}`);

  const data = await postJSON("/chat", { session_id: sessionId, text: cleaned });
  sessionId = data.session_id || sessionId;

  // end session response includes summary
  if (data.summary) {
    log(`\nAssistant: ${data.text || ""}`);
    renderSummary(data.summary);
    await speak(data.text || "Session ended.");
    return;
  }

  const reply = data.text || "";
  log(`\nAssistant: ${reply}`);
  await speak(reply);
}

// Send button
btnSend.addEventListener("click", async () => {
  const t = textInput.value;
  textInput.value = "";
  await sendText(t);
  textInput.focus();
});

// Enter to send
textInput.addEventListener("keydown", async (e) => {
  if (e.key === "Enter") {
    const t = textInput.value;
    textInput.value = "";
    await sendText(t);
    textInput.focus();
  }
});

// End session
btnEnd.addEventListener("click", async () => {
  await sendText("end session");
  textInput.focus();
});

// ---------- voice (hold-to-talk on mic button) ----------
let mediaRecorder = null;
let chunks = [];
let stream = null;

async function initMic() {
  if (stream) return;
  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  log("[Mic] Permission granted");
}

async function startRecording() {
  await initMic();
  chunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) chunks.push(e.data);
  };
  mediaRecorder.start();
  log("[Mic] Recording...");
}

async function stopRecording() {
  if (!mediaRecorder) return;

  // Stop recording
  mediaRecorder.stop();

  const stopped = new Promise((resolve) => {
    mediaRecorder.onstop = resolve;
  });
  await stopped;

  log("[Mic] Uploading audio...");
  const blob = new Blob(chunks, { type: "audio/webm" });
  const form = new FormData();
  form.append("file", blob, "audio.webm");

  const res = await fetch(`${API_BASE}/asr`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const t = await res.text();
    log(`[ASR] Failed: ${t}`);
    return;
  }

  const data = await res.json();
  const text = (data.text || "").trim();
  log(`[ASR] ${text || "(empty)"}`);

  if (text) {
    // Optional: show recognized text in input before sending
    textInput.value = text;
    await sendText(text);
    textInput.value = "";
  }
}

// Mouse: hold to talk
micBtn.addEventListener("mousedown", async () => {
  micBtn.classList.add("recording");
  await startRecording();
});

micBtn.addEventListener("mouseup", async () => {
  micBtn.classList.remove("recording");
  await stopRecording();
});

micBtn.addEventListener("mouseleave", () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    micBtn.classList.remove("recording");
    stopRecording();
  }
});

// Touch: hold to talk (mobile)
micBtn.addEventListener("touchstart", (e) => {
  e.preventDefault();
  micBtn.classList.add("recording");
  startRecording();
});

micBtn.addEventListener("touchend", (e) => {
  e.preventDefault();
  micBtn.classList.remove("recording");
  stopRecording();
});

log("[Ready] Type and press Enter/Send, or hold the mic to talk.");
textInput.focus();
