const $ = (s) => document.querySelector(s);
const docsEl = $("#docs"), thread = $("#thread"), statusEl = $("#status");

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return res.status === 204 ? null : res.json();
}

function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => (k === "class" ? (n.className = v) : n.setAttribute(k, v)));
  children.forEach((c) => n.append(c));
  return n;
}

async function refreshDocs() {
  const docs = await api("/api/documents");
  docsEl.replaceChildren(...docs.map((d) => {
    const del = el("button", { title: "Remove" }, "×");
    del.onclick = async () => { await api(`/api/documents/${d.id}`, { method: "DELETE" }); refreshDocs(); };
    return el("li", {}, el("div", {}, el("div", { class: "name" }, d.filename),
      el("div", { class: "sub" }, `${d.pages} pages · ${d.chunks} chunks`)), del);
  }));
}

async function upload(files) {
  for (const f of files) {
    const row = el("li", { class: "pending" }, el("div", { class: "name" }, `Indexing ${f.name}…`));
    docsEl.append(row);
    const body = new FormData();
    body.append("file", f);
    try { await api("/api/documents", { method: "POST", body }); }
    catch (e) { alert(`${f.name}: ${e.message}`); }
    row.remove();
  }
  refreshDocs();
}

const drop = $("#drop"), fileInput = $("#file");
fileInput.onchange = () => upload([...fileInput.files]);
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => upload([...e.dataTransfer.files]));

function renderAnswer(node, data) {
  const a = node.querySelector(".a"), cites = node.querySelector(".cites");
  a.classList.remove("loading");
  a.replaceChildren();
  // Turn [n] markers into clickable superscripts.
  data.answer.split(/(\[\d+\])/).forEach((part) => {
    const m = part.match(/^\[(\d+)\]$/);
    if (!m) return a.append(part);
    const s = el("span", { class: "cite" }, part);
    s.onclick = () => cites.querySelector(`[data-n="${m[1]}"]`)?.click();
    a.append(s);
  });
  let open = null;
  data.citations.forEach((c) => {
    const chip = el("button", { class: "chip", "data-n": c.n });
    chip.append(el("b", {}, `[${c.n}] `), `${c.filename} · p. ${c.page}`);
    chip.onclick = () => {
      cites.querySelectorAll(".chip").forEach((x) => x.classList.remove("active"));
      open?.remove();
      if (open?.dataset.n === String(c.n)) { open = null; return; }
      chip.classList.add("active");
      open = el("div", { class: "passage" }, c.snippet);
      open.dataset.n = c.n;
      chip.after(open);
    };
    cites.append(chip);
  });
  const t = data.timing_ms;
  node.querySelector(".meta").textContent =
    `${data.mode === "extractive" ? "extractive mode · " : ""}retrieval ${t.retrieval} ms · generation ${t.generation} ms`;
}

$("#ask").onsubmit = async (e) => {
  e.preventDefault();
  const q = $("#question").value.trim();
  if (!q) return;
  $("#empty")?.remove();
  const node = $("#tpl-answer").content.firstElementChild.cloneNode(true);
  node.querySelector(".q").textContent = q;
  const a = node.querySelector(".a");
  a.textContent = "Thinking…";
  a.classList.add("loading");
  thread.append(node);
  node.scrollIntoView({ behavior: "smooth" });
  $("#question").value = "";
  $("#send").disabled = true;
  try {
    renderAnswer(node, await api("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q }),
    }));
  } catch (err) {
    a.textContent = `Error: ${err.message}`;
  }
  $("#send").disabled = false;
};

api("/api/health").then((h) => {
  statusEl.textContent = h.llm ? "● LLM connected" : "● extractive mode (no API key)";
}).catch(() => (statusEl.textContent = "● backend unreachable"));
refreshDocs();
