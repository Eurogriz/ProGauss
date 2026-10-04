// Веб-интерфейс QuantumLab. Научной логики здесь нет: страница вызывает те же
// эндпоинты, что и любой клиент API, а все тексты берёт из /i18n/{locale}.
"use strict";

const API = "/api/v1";
const TASKS = ["single_point", "optimization", "frequencies"];
const PROFILES = ["screening", "standard", "high_accuracy", "research"];
const FINAL = new Set(["completed", "completed_with_warnings", "failed", "cancelled"]);
const COVALENT = { H: 0.31, B: 0.84, C: 0.76, N: 0.71, O: 0.66, F: 0.57, Si: 1.11, P: 1.07, S: 1.05, Cl: 1.02, Br: 1.2, I: 1.39 };
const COLORS = { H: "#9aa4b2", C: "#3a3f47", N: "#2f5bd8", O: "#d92d20", F: "#12a150", S: "#d9a400", P: "#f08c00", Cl: "#12a150" };
const WATER = "3\nwater\nO 0.000 0.000 0.117\nH 0.000 0.757 -0.469\nH 0.000 -0.757 -0.469\n";

const state = {
  locale: localStorage.getItem("ql.locale") || "ru",
  messages: {},
  projectId: null,
  moleculeId: null,
  molecules: [],
  plan: null,
  jobId: null,
};

// -- утилиты ---------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const camel = (key) => key.replace(/_([a-z0-9])/g, (_, c) => c.toUpperCase());

function h(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs || {})) {
    if (name === "class") node.className = value;
    else if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
    else node.setAttribute(name, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

// Каталог строк приходит с ключами в camelCase (граница API), поэтому ключ
// приводится так же; параметры подстановки — по той же схеме.
function t(key, params) {
  const text = state.messages[camel(key)] ?? key;
  if (!params) return text;
  return text.replace(/\{([a-z0-9_]+)\}/gi, (_, name) => {
    const value = params[camel(name)] ?? params[name];
    return value === undefined ? `{${name}}` : String(value);
  });
}

async function api(path, options) {
  const response = await fetch(API + path, {
    headers: { "Content-Type": "application/json", "Accept-Language": state.locale },
    ...options,
  });
  const text = await response.text();
  const body = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const error = new Error(body?.detail || body?.title || response.statusText);
    error.title = body?.title;
    throw error;
  }
  return body;
}

function banner(message, info = false) {
  const node = $("banner");
  if (!message) { node.hidden = true; return; }
  node.textContent = message;
  node.className = info ? "banner info" : "banner";
  node.hidden = false;
}

async function guarded(action) {
  try { banner(""); await action(); }
  catch (error) { banner(error.title && error.title !== error.message ? `${error.title}: ${error.message}` : error.message); }
}

const number = (value, digits = 8) => (typeof value === "number" ? value.toFixed(digits) : "—");

// -- локализация -------------------------------------------------------------
async function loadMessages() {
  state.messages = await api(`/i18n/${state.locale}`);
  document.documentElement.lang = state.locale;
  $("title").textContent = t("gui.title");
  $("subtitle").textContent = t("app.tagline");
  document.title = t("gui.title");
  for (const node of document.querySelectorAll("[data-i18n]")) node.textContent = t(node.dataset.i18n);
  for (const node of document.querySelectorAll("[data-i18n-placeholder]")) node.placeholder = t(node.dataset.i18nPlaceholder);
  fillSelect($("task"), TASKS, (value) => t(`task.${value}.title`));
  fillSelect($("profile"), PROFILES, (value) => t(`profile.${value}.name`));
  $("project-list").dataset.empty = t("gui.noProjects");
  $("molecule-list").dataset.empty = t("gui.noMolecules");
}

function fillSelect(select, values, label) {
  const current = select.value;
  select.replaceChildren(...values.map((value) => h("option", { value }, label(value))));
  if (values.includes(current)) select.value = current;
}

// -- проекты и структуры -----------------------------------------------------
async function refreshProjects() {
  const { items } = await api("/projects");
  if (!items.some((item) => item.id === state.projectId)) state.projectId = items[0]?.id ?? null;
  $("project-list").replaceChildren(...items.map((item) =>
    h("li", { class: item.id === state.projectId ? "selected" : "", onclick: () => guarded(async () => { state.projectId = item.id; state.moleculeId = null; await refreshProjects(); }) }, item.name)));
  await refreshMolecules();
  await refreshJobs();
}

async function refreshMolecules() {
  state.molecules = state.projectId ? (await api(`/projects/${state.projectId}/molecules`)).items : [];
  if (!state.molecules.some((item) => item.id === state.moleculeId)) state.moleculeId = state.molecules[0]?.id ?? null;
  $("molecule-list").replaceChildren(...state.molecules.map((item) =>
    h("li", { class: item.id === state.moleculeId ? "selected" : "", onclick: () => { state.moleculeId = item.id; state.plan = null; renderMolecules(); renderPlan(); } },
      `${item.name} · ${item.atoms.length} ${t("gui.atoms")}`)));
  renderMolecules();
}

function renderMolecules() {
  for (const node of $("molecule-list").children) node.classList.remove("selected");
  const index = state.molecules.findIndex((item) => item.id === state.moleculeId);
  if (index >= 0) $("molecule-list").children[index].classList.add("selected");
  drawMolecule(state.molecules[index]);
  $("submit-button").disabled = !state.plan;
}

function drawMolecule(molecule) {
  const svg = $("preview");
  svg.replaceChildren();
  if (!molecule) return;
  const atoms = molecule.atoms.map((atom) => ({ symbol: atom.symbol, x: atom.position[0], y: atom.position[1], z: atom.position[2] }));
  const centre = ["x", "y", "z"].map((axis) => atoms.reduce((sum, atom) => sum + atom[axis], 0) / atoms.length);
  for (const atom of atoms) { atom.x -= centre[0]; atom.y -= centre[1]; atom.z -= centre[2]; }
  // Проекция на плоскость наибольшего разброса (главные оси не считаем: достаточно выбрать две оси с большим размахом).
  const spread = ["x", "y", "z"].map((axis) => Math.max(...atoms.map((a) => a[axis])) - Math.min(...atoms.map((a) => a[axis])));
  const order = [0, 1, 2].sort((a, b) => spread[b] - spread[a]);
  const [u, v] = [["x", "y", "z"][order[0]], ["x", "y", "z"][order[1]]];
  const extent = Math.max(spread[order[0]], spread[order[1]], 1);
  const scale = 80 / extent;
  const point = (atom) => [atom[u] * scale, -atom[v] * scale];
  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      const limit = 1.3 * ((COVALENT[atoms[i].symbol] ?? 0.8) + (COVALENT[atoms[j].symbol] ?? 0.8));
      const distance = Math.hypot(atoms[i].x - atoms[j].x, atoms[i].y - atoms[j].y, atoms[i].z - atoms[j].z);
      if (distance < limit) {
        const [x1, y1] = point(atoms[i]);
        const [x2, y2] = point(atoms[j]);
        svg.append(svgNode("line", { x1, y1, x2, y2, stroke: "#98a2b3", "stroke-width": 2 }));
      }
    }
  }
  for (const atom of atoms) {
    const [cx, cy] = point(atom);
    const radius = 4 + 8 * (COVALENT[atom.symbol] ?? 0.8);
    svg.append(svgNode("circle", { cx, cy, r: radius, fill: COLORS[atom.symbol] ?? "#7a5af8", opacity: 0.9 }));
    const label = svgNode("text", { x: cx, y: cy + 3, "text-anchor": "middle", "font-size": 8, fill: "#fff" });
    label.textContent = atom.symbol;
    svg.append(label);
  }
}

function svgNode(tag, attrs) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
  return node;
}

// -- план и постановка в очередь --------------------------------------------
function renderPlan() {
  const box = $("plan-result");
  box.replaceChildren();
  $("submit-button").disabled = !state.plan;
  if (!state.plan) return;
  box.append(
    h("h3", {}, t("gui.planTitle")),
    h("p", { class: "muted" }, state.plan.rationale),
    h("ul", { class: "plain" }, state.plan.decisions.map((decision) => h("li", {}, decision.text))),
  );
}

async function makePlan() {
  if (!state.moleculeId) throw new Error(t("gui.selectMoleculeFirst"));
  state.plan = await api("/calculations/plan", {
    method: "POST",
    body: JSON.stringify({ task: $("task").value, profile: $("profile").value, moleculeId: state.moleculeId }),
  });
  renderPlan();
}

async function submitJob() {
  if (!state.plan || !state.moleculeId) throw new Error(t("gui.planFirst"));
  const job = await api("/jobs", { method: "POST", body: JSON.stringify({ moleculeId: state.moleculeId, spec: state.plan.spec }) });
  state.jobId = job.id;
  banner(t("gui.submitted", { id: job.id.slice(0, 8) }), true);
  await refreshJobs();
}

// -- задания -------------------------------------------------------------------
async function refreshJobs() {
  const query = state.projectId ? `?projectId=${encodeURIComponent(state.projectId)}` : "";
  const { items } = await api(`/jobs${query}`);
  $("no-jobs").hidden = items.length > 0;
  $("job-rows").replaceChildren(...items.map((job) =>
    h("tr", { class: job.id === state.jobId ? "selected" : "", onclick: () => guarded(async () => { state.jobId = job.id; await refreshJobs(); }) },
      h("td", {}, job.name),
      h("td", {}, h("span", { class: `badge ${job.status}` }, t(`status.${job.status}`))),
      h("td", {}, job.priority),
      h("td", {}, new Date(job.createdAt).toLocaleString(state.locale)))));
  const selected = items.find((job) => job.id === state.jobId);
  await renderJob(selected);
}

async function renderJob(job) {
  const box = $("job-details");
  if (!job) { box.replaceChildren(); return; }
  const parts = [h("h3", {}, `${job.name} · ${t(`status.${job.status}`)}`)];
  if (job.status === "completed" || job.status === "completed_with_warnings") {
    const r = await api(`/jobs/${job.id}/result`);
    parts.push(
      h("dl", { class: "kv" },
        h("dt", {}, t("gui.energy")), h("dd", {}, `${number(r.energyHartree, 10)} Eh`),
        h("dt", {}, t("gui.converged")), h("dd", {}, r.converged ? t("gui.yes") : t("gui.no")),
        h("dt", {}, t("gui.iterations")), h("dd", {}, r.scfIterations),
        h("dt", {}, t("gui.homo")), h("dd", {}, `${number(r.homoEnergyHartree, 6)} Eh`),
        h("dt", {}, t("gui.lumo")), h("dd", {}, `${number(r.lumoEnergyHartree, 6)} Eh`),
        h("dt", {}, t("gui.dipole")), h("dd", {}, `${number(r.dipoleDebye, 4)} D`),
        r.optimizationSteps != null ? [h("dt", {}, t("gui.steps")), h("dd", {}, r.optimizationSteps)] : null,
        r.frequenciesCm1?.length ? [h("dt", {}, t("gui.frequencies")), h("dd", {}, r.frequenciesCm1.map((f) => f.toFixed(1)).join(", "))] : null),
    );
    if (r.warnings?.length) {
      parts.push(h("h4", {}, t("gui.warnings")), h("ul", { class: "plain" }, r.warnings.map((w) => h("li", {}, t(w.key, w.params)))));
    }
    parts.push(h("details", {}, h("summary", {}, t("gui.raw")), h("pre", {}, JSON.stringify(r, null, 2))));
  } else if (job.status === "failed") {
    parts.push(h("p", { class: "muted" }, t("gui.failedHint")));
  }
  box.replaceChildren(...parts);
}

// -- возможности -------------------------------------------------------------
async function renderCapabilities() {
  const snapshot = await api("/capabilities");
  const groups = Object.entries(snapshot).map(([kind, items]) =>
    h("div", { class: "cap-group" }, h("h3", {}, kind),
      items.map((item) => h("div", { class: "cap", title: (item.limitations || []).join("\n") },
        h("code", {}, item.id), h("span", { class: `badge ${item.availability}` }, item.availability),
        item.limitations?.length ? h("span", { class: "muted" }, item.limitations[0]) : null))));
  $("capability-list").replaceChildren(...groups);
}

// -- запуск ---------------------------------------------------------------------
async function checkHealth() {
  const badge = $("health");
  try {
    const body = await api("/ready");
    badge.textContent = body.status === "ready" ? t("gui.ready") : t("gui.notReady");
    badge.className = `badge ${body.status === "ready" ? "ok" : "bad"}`;
  } catch {
    badge.textContent = t("gui.notReady");
    badge.className = "badge bad";
  }
}

function bind() {
  $("locale").value = state.locale;
  $("locale").addEventListener("change", () => guarded(async () => {
    state.locale = $("locale").value;
    localStorage.setItem("ql.locale", state.locale);
    await loadMessages(); await refreshProjects(); await renderCapabilities(); await checkHealth();
  }));
  $("tabs").addEventListener("click", (event) => {
    const tab = event.target.dataset?.tab;
    if (!tab) return;
    for (const button of $("tabs").children) button.classList.toggle("active", button.dataset.tab === tab);
    for (const node of document.querySelectorAll(".tab")) node.hidden = node.id !== `tab-${tab}`;
  });
  $("project-form").addEventListener("submit", (event) => { event.preventDefault(); guarded(async () => {
    const created = await api("/projects", { method: "POST", body: JSON.stringify({ name: $("project-name").value }) });
    $("project-name").value = ""; state.projectId = created.id; await refreshProjects(); }); });
  $("molecule-form").addEventListener("submit", (event) => { event.preventDefault(); guarded(async () => {
    if (!state.projectId) throw new Error(t("gui.selectProject"));
    const created = await api(`/projects/${state.projectId}/molecules`, { method: "POST", body: JSON.stringify({
      name: $("molecule-name").value || null, content: $("molecule-xyz").value,
      charge: Number($("molecule-charge").value), multiplicity: Number($("molecule-multiplicity").value) }) });
    state.moleculeId = created.id; state.plan = null; await refreshMolecules(); renderPlan(); }); });
  $("example-water").addEventListener("click", () => { $("molecule-name").value = "water"; $("molecule-xyz").value = WATER; });
  $("plan-button").addEventListener("click", () => guarded(makePlan));
  $("submit-button").addEventListener("click", () => guarded(submitJob));
  for (const id of ["task", "profile"]) $(id).addEventListener("change", () => { state.plan = null; renderPlan(); });
}

async function start() {
  bind();
  await guarded(async () => {
    await loadMessages();
    await renderCapabilities();
    await refreshProjects();
    await checkHealth();
  });
  setInterval(() => { refreshJobs().catch(() => {}); checkHealth(); }, 2000);
}

start();
