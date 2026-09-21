"use strict";
const $ = (selector) => document.querySelector(selector);
const storeKey = "kraken.chats.v1";
const makeId = () => crypto.randomUUID();
let chats = [], currentId = null, pending = null, toastTimer;
let personality = true;

function toast(text) {
  $("#toast").textContent = text;
  $("#toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("#toast").hidden = true; }, 4000);
}

function load() {
  try {
    const state = JSON.parse(localStorage.getItem(storeKey) || "null");
    if (!state) return;
    if (!Array.isArray(state.chats)) throw new Error("Invalid history");
    chats = state.chats.slice(0, 50).filter(c => typeof c.id === "string" && typeof c.title === "string" && Array.isArray(c.messages)).map(c => ({
      id: c.id, title: c.title.slice(0, 80), messages: c.messages.slice(-200).filter(m => m && ["user", "assistant"].includes(m.role) && typeof m.content === "string").map(m => ({
        id: typeof m.id === "string" ? m.id : makeId(), role: m.role, content: m.content.slice(0, 12000),
        statuses: Array.isArray(m.statuses) ? m.statuses.filter(s => typeof s === "string").slice(0, 5) : [],
        state: m.state === "pending" ? "stopped" : m.state, error: typeof m.error === "string" ? m.error : ""
      }))
    }));
    currentId = chats.some(c => c.id === state.currentId) ? state.currentId : null;
    personality = state.personality !== false;
  } catch { toast("Не удалось прочитать историю. Можно начать новый диалог."); }
}

function save() {
  try { localStorage.setItem(storeKey, JSON.stringify({chats, currentId, personality})); }
  catch { toast("Браузер не смог сохранить историю. Текущий диалог остаётся доступен до закрытия страницы."); }
}

function current() { return chats.find(c => c.id === currentId); }
function element(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function setMenu(open) {
  $("#sidebar").classList.toggle("open", open);
  $("#backdrop").hidden = !open;
  $("#menu-button").setAttribute("aria-expanded", String(open));
  $("#sidebar").inert = window.matchMedia("(max-width: 760px)").matches && !open;
}

function renderHistory() {
  const list = $("#history"); list.replaceChildren();
  $("#chat-count").textContent = chats.length;
  if (!chats.length) list.append(element("p", "history-empty", "Пока чистый лист.\nВаш первый диалог появится здесь."));
  for (const chat of chats) {
    const row = element("div", `history-item${chat.id === currentId ? " active" : ""}`);
    const open = element("button", "history-open", chat.title);
    open.title = chat.title;
    if (chat.id === currentId) open.setAttribute("aria-current", "page");
    open.onclick = () => { if (pending) stop(); currentId = chat.id; save(); render(); setMenu(false); };
    const del = element("button", "history-delete", "×");
    del.setAttribute("aria-label", `Удалить диалог: ${chat.title}`);
    del.onclick = () => {
      if (!confirm(`Удалить диалог «${chat.title}» из этого браузера?`)) return;
      if (pending?.chatId === chat.id) stop();
      chats = chats.filter(c => c.id !== chat.id);
      if (currentId === chat.id) currentId = null;
      save(); render();
    };
    row.append(open, del); list.append(row);
  }
}

// No innerHTML: user text, model replies and saved history remain inert text.
function renderText(target, text) {
  target.replaceChildren();
  const chunks = text.split("```");
  chunks.forEach((chunk, i) => {
    if (!chunk) return;
    if (i % 2) {
      const pre = element("pre");
      const line = chunk.indexOf("\n");
      const language = line >= 0 ? chunk.slice(0, line).trim() : "";
      if (language && /^[a-z0-9+#-]{1,20}$/i.test(language)) pre.append(element("span", "code-label", language));
      pre.append(element("code", "", line >= 0 ? chunk.slice(line + 1).trimEnd() : chunk));
      target.append(pre);
    } else {
      for (const paragraph of chunk.split("\n\n")) if (paragraph.trim()) target.append(element("p", "", paragraph.trim()));
    }
  });
}

function renderMessage(message, chat) {
  const article = element("article", `message ${message.role}`);
  article.dataset.id = message.id;
  if (message.role === "user") { article.append(element("div", "message-body", message.content)); return article; }
  const label = element("div", "message-label");
  const logo = element("img"); logo.src = "/favicon.svg"; logo.alt = "";
  label.append(logo, document.createTextNode("Kraken"), element("span", "", "MINI")); article.append(label);
  const details = element("details", `processing${message.state === "pending" ? " pending" : ""}`);
  details.open = message.state === "pending";
  const summary = element("summary", "", message.state === "pending" ? "Обрабатываю запрос…" : "Обработка и комментарии");
  details.append(summary);
  for (const status of message.statuses || []) details.append(element("p", "", status));
  if ((message.statuses || []).length || message.state === "pending") article.append(details);
  const body = element("div", "message-body"); renderText(body, message.content); article.append(body);
  if (message.state === "stopped") article.append(element("p", "message-error", "Ответ остановлен."));
  if (message.error) article.append(element("p", "message-error", message.error));
  const actions = element("div", "message-actions");
  if (message.content && message.state !== "pending") {
    const copy = element("button", "", "Копировать");
    copy.onclick = async () => {
      try { await navigator.clipboard.writeText(message.content); toast("Ответ скопирован"); }
      catch { toast("Не удалось скопировать. Выделите текст и нажмите Ctrl+C."); }
    };
    actions.append(copy);
  }
  if (message.state !== "pending" && chat.messages.at(-1) === message) {
    const retry = element("button", "", "↻ Повторить");
    retry.disabled = Boolean(pending);
    retry.onclick = () => { if (pending) return; chat.messages.pop(); save(); requestReply(chat); };
    actions.append(retry);
  }
  article.append(actions);
  return article;
}

function render() {
  renderHistory();
  const chat = current();
  const hasMessages = Boolean(chat?.messages.length);
  $("#welcome").hidden = hasMessages;
  $("#messages").hidden = !hasMessages;
  $("#messages").replaceChildren(...(chat?.messages || []).map(m => renderMessage(m, chat)));
  updateControls();
  if (hasMessages) scrollBottom();
  else $("#scroll-area").scrollTop = 0;
}

function updateMessage(chat, message) {
  if (currentId !== chat.id) return;
  const area = $("#scroll-area");
  const nearBottom = area.scrollHeight - area.scrollTop - area.clientHeight < 100;
  const previous = [...$("#messages").children].find(el => el.dataset.id === message.id);
  if (previous) {
    const wasOpen = previous.querySelector("details")?.open;
    const replacement = renderMessage(message, chat);
    if (wasOpen && replacement.querySelector("details")) replacement.querySelector("details").open = true;
    previous.replaceWith(replacement);
  }
  if (nearBottom) scrollBottom();
}

function scrollBottom() { const a = $("#scroll-area"); a.scrollTop = a.scrollHeight; }
function updateControls() {
  const button = $("#send-button");
  button.disabled = !pending && !$("#prompt").value.trim();
  button.textContent = pending ? "■" : "↑";
  button.classList.toggle("stop", Boolean(pending));
  button.setAttribute("aria-label", pending ? "Остановить ответ" : "Отправить сообщение");
  $("#personality").setAttribute("aria-pressed", String(personality));
  $("#connection-status").textContent = pending ? "Kraken обрабатывает запрос…" : "Без внешних AI API";
}

function stop() {
  if (!pending) return;
  const job = pending;
  job.message.state = "stopped";
  job.controller.abort();
  pending = null;
  save(); updateMessage(job.chat, job.message); updateControls();
}

async function requestReply(chat) {
  const message = {id: makeId(), role: "assistant", content: "", statuses: [], state: "pending"};
  const controller = new AbortController();
  const context = chat.messages.filter(m => m.content && (m.role === "user" || m.state === "done")).slice(-24).map(({role, content}) => ({role, content}));
  // Keep the serialized UTF-8 request comfortably below the API body limit.
  while (context.length > 1 && new TextEncoder().encode(JSON.stringify(context)).length > 55000) context.shift();
  const job = {controller, chatId: chat.id, chat, message};
  chat.messages.push(message); pending = job;
  render(); save();
  let timedOut = false, completed = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, 30000);
  try {
    const response = await fetch("/api/chat", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({messages: context, personality}), signal: controller.signal});
    if (!response.ok) {
      let reason = "Сервер не смог обработать запрос.";
      try { reason = (await response.json()).error || reason; } catch {}
      throw new Error(reason);
    }
    if (!response.body) throw new Error("Браузер не поддерживает потоковый ответ.");
    const reader = response.body.getReader();
    const decoder = new TextDecoder(); let buffer = "";
    function event(line) {
      if (!line.trim()) return;
      const data = JSON.parse(line);
      if (data.type === "status" && typeof data.text === "string") message.statuses.push(data.text);
      if (data.type === "token" && typeof data.text === "string") message.content += data.text;
      if (data.type === "error") throw new Error(data.text || "Ошибка модели.");
      if (data.type === "done") { completed = true; message.state = "done"; }
      updateMessage(chat, message);
    }
    while (true) {
      const {value, done} = await reader.read();
      if (controller.signal.aborted) throw new DOMException("Aborted", "AbortError");
      buffer += decoder.decode(value, {stream: !done});
      const lines = buffer.split("\n"); buffer = lines.pop();
      for (const line of lines) event(line);
      if (done) { if (buffer.trim()) event(buffer); break; }
    }
    if (!completed) throw new Error("Соединение прервалось. Повторите запрос.");
    $("#announcer").textContent = "Kraken ответил: " + message.content;
  } catch (error) {
    if (error.name === "AbortError" && !timedOut) message.state = "stopped";
    else { message.state = "error"; message.error = timedOut ? "Сервер не ответил за 30 секунд. Попробуйте ещё раз." : error.message === "Failed to fetch" ? "Не удалось связаться с сервером. Проверьте интернет и повторите запрос." : error.message; }
  } finally {
    clearTimeout(timer);
    if (pending === job) pending = null;
    save(); updateMessage(chat, message); updateControls();
  }
}

function send(text) {
  if (pending || !text.trim()) return;
  text = text.trim();
  if (text.length > 4000) { toast("Максимум 4000 символов в одном сообщении."); return; }
  let chat = current();
  if (!chat) {
    if (chats.length >= 50) { toast("Сохранено 50 диалогов. Удалите один, чтобы начать новый."); return; }
    chat = {id: makeId(), title: text.slice(0, 52), messages: []};
    chats.unshift(chat); currentId = chat.id;
  }
  if (chat.messages.length >= 200) { toast("В этом диалоге уже 100 пар сообщений. Начните новый."); return; }
  chat.messages.push({id: makeId(), role: "user", content: text});
  $("#prompt").value = ""; $("#prompt").style.height = "auto";
  requestReply(chat);
}

$("#chat-form").addEventListener("submit", event => { event.preventDefault(); if (pending) stop(); else send($("#prompt").value); });
// Stop remains usable when the required textarea is empty.
$("#send-button").addEventListener("click", event => { if (pending) { event.preventDefault(); stop(); } });
$("#prompt").addEventListener("input", () => { const field = $("#prompt"); field.style.height = "auto"; field.style.height = Math.min(field.scrollHeight, 160) + "px"; updateControls(); });
$("#prompt").addEventListener("keydown", event => { if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!pending) send(event.target.value); } });
$("#new-chat").onclick = () => { if (pending) stop(); currentId = null; save(); render(); setMenu(false); $("#prompt").focus(); };
$("#personality").onclick = () => { personality = !personality; save(); updateControls(); };
document.querySelectorAll("[data-prompt]").forEach(button => { button.onclick = () => send(button.dataset.prompt); });
$("#menu-button").onclick = () => setMenu(!$("#sidebar").classList.contains("open"));
$("#backdrop").onclick = () => setMenu(false);
const about = $("#about-dialog");
$("#about-button").onclick = $("#model-button").onclick = () => { setMenu(false); about.showModal(); };
$("#close-about").onclick = () => about.close();
about.addEventListener("click", event => { if (event.target === about) { const r = about.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) about.close(); } });
document.addEventListener("keydown", event => { if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") { event.preventDefault(); $("#new-chat").click(); } if (event.key === "Escape") setMenu(false); });
window.matchMedia("(max-width: 760px)").addEventListener("change", () => setMenu(false));
load(); render(); setMenu(false);
