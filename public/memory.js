/* Browser-owned memory: explicit facts, provenance and bounded history retrieval.
 * No remote storage, background requests, or data shared between visitors.
 * The CommonJS export lets Node exercise the exact browser implementation.
 */
(function (root) {
  "use strict";
  const MAX_FACTS = 60;
  const STOP = new Set("что как где когда почему зачем это меня мне мой моя мои мы ты вы он она оно есть был была было про для или кто чем еще уже помнишь расскажи пожалуйста такое такой этот об из на по да нет а и в с у о к я не но за от до".split(" "));
  const words = text => (text.toLowerCase().replaceAll("ё", "е").match(/[\p{L}\p{N}]+/gu) || []).filter(w => w.length > 2 && !STOP.has(w)).map(w => w.length > 5 ? w.slice(0, -2) : w);
  const blank = () => ({version: 1, enabled: true, facts: [], ignored: [], blockedChats: []});
  function normalize(value) {
    const result = blank();
    if (!value || typeof value !== "object") return result;
    result.enabled = value.enabled !== false;
    result.facts = (Array.isArray(value.facts) ? value.facts : []).filter(f => f && typeof f.id === "string" && typeof f.key === "string" && typeof f.text === "string" && typeof f.value === "string" && typeof f.chatId === "string" && typeof f.messageId === "string").slice(-MAX_FACTS).map(f => ({id: f.id, key: f.key.slice(0, 80), text: f.text.slice(0, 600), value: f.value.slice(0, 400), chatId: f.chatId, messageId: f.messageId, createdAt: Number(f.createdAt) || 0}));
    result.ignored = (Array.isArray(value.ignored) ? value.ignored : []).filter(x => typeof x === "string").slice(-10000);
    result.blockedChats = (Array.isArray(value.blockedChats) ? value.blockedChats : []).filter(x => typeof x === "string").slice(-100);
    return result;
  }
  function extract(text) {
    if (typeof text !== "string" || text.length > 4000) return [];
    // Avoid automatically recording credential-shaped content.
    if (/(пароль|секретный ключ|api.?key|access.?token|номер карты)/i.test(text)) return [];
    const patterns = [
      ["name", /(?:^|[.!?]\s*|\n)(?:кстати,?\s*)?меня зовут\s+([\p{L}][\p{L}\-]{1,35})(?=[\s.!?,]|$)/iu, "Имя"],
      ["city", /(?:^|[.!?]\s*|\n)я живу в\s+([^.!?\n]{2,100})/iu, "Город"],
      ["study", /(?:^|[.!?]\s*|\n)я (?:изучаю|учу)\s+([^.!?\n]{2,150})/iu, "Изучает"],
      ["likes", /(?:^|[.!?]\s*|\n)(?:мне нравится|я люблю)\s+([^.!?\n]{2,150})/iu, "Нравится"],
      ["dislikes", /(?:^|[.!?]\s*|\n)я не люблю\s+([^.!?\n]{2,150})/iu, "Не нравится"],
      ["project", /(?:^|[.!?]\s*|\n)мой проект называется\s+([^.!?\n]{2,150})/iu, "Проект"],
      ["goal", /(?:^|[.!?]\s*|\n)моя цель\s*[:—-]?\s+([^.!?\n]{2,200})/iu, "Цель"],
      ["style", /(?:^|[.!?]\s*|\n)(отвечай (?:кратко|подробно|без сарказма|простыми словами))/iu, "Стиль ответа"]
    ];
    const facts = [];
    for (const [key, regex, title] of patterns) {
      const match = text.match(regex);
      if (match) { const value = match[1].trim(); facts.push({key, value, text: `${title}: ${value}`}); }
    }
    const explicit = text.match(/^запомни\s*[:—-]?\s+([\s\S]{2,400})$/iu);
    if (explicit && !facts.length) {
      const nested = extract(explicit[1]);
      if (nested.length) return nested;
      const value = explicit[1].trim();
      facts.push({key: "note:" + value.toLowerCase().slice(0, 70), value, text: value});
    }
    return facts;
  }
  function learn(memory, message, chatId, now = Date.now()) {
    if (!memory.enabled || memory.blockedChats.includes(chatId) || message.role !== "user" || memory.ignored.includes(message.id)) return [];
    const updated = [];
    for (const fact of extract(message.content)) {
      const previous = memory.facts.find(f => f.key === fact.key);
      if (previous?.value.toLowerCase() === fact.value.toLowerCase()) continue;
      // The replaced source must never reappear via history retrieval.
      if (previous && !memory.ignored.includes(previous.messageId)) memory.ignored.push(previous.messageId);
      memory.facts = memory.facts.filter(f => f.key !== fact.key);
      const record = {...fact, id: `${message.id}:${fact.key}`, messageId: message.id, chatId, createdAt: now};
      memory.facts.push(record); updated.push(record);
    }
    memory.facts = memory.facts.slice(-MAX_FACTS);
    return updated;
  }
  function forget(memory, id, chats) {
    const fact = memory.facts.find(f => f.id === id);
    if (!fact) return;
    // Exclude every existing repetition of the forgotten fact, not only its first source.
    for (const chat of chats) for (const message of chat.messages) {
      if (message.role === "user" && extract(message.content).some(f => f.key === fact.key && f.value.toLowerCase() === fact.value.toLowerCase())) memory.ignored.push(message.id);
    }
    memory.ignored = [...new Set(memory.ignored)];
    memory.facts = memory.facts.filter(f => f.id !== id);
  }
  function clear(memory, chats) {
    memory.facts = [];
    memory.ignored = [...new Set([...memory.ignored, ...chats.flatMap(c => c.messages.map(m => m.id))])];
  }
  function removeChat(memory, chatId) {
    memory.facts = memory.facts.filter(f => f.chatId !== chatId);
    memory.blockedChats = memory.blockedChats.filter(id => id !== chatId);
  }
  function retrieve(memory, chats, currentId, query) {
    if (!memory.enabled) return {enabled: false, facts: [], excerpts: []};
    const queryWords = new Set(words(query));
    const score = text => words(text).reduce((sum, word) => sum + (queryWords.has(word) ? 1 : 0), 0);
    const requested = /как меня зовут|мо[её] имя/iu.test(query) ? "name" : /где я живу|мо[йеё] город/iu.test(query) ? "city" : /что я (?:учу|изучаю)/iu.test(query) ? "study" : /мо[йяю] проект/iu.test(query) ? "project" : /мо[яю] цель/iu.test(query) ? "goal" : null;
    const stated = extract(query);
    const saved = memory.facts.filter(f => stated.some(s => s.key === f.key && s.value.toLowerCase() === f.value.toLowerCase())).map(f => f.id);
    const facts = memory.facts.filter(f => chats.some(c => c.id === f.chatId) && !memory.blockedChats.includes(f.chatId)).map(f => ({...f, score: score(f.text) + (f.key === requested || saved.includes(f.id) ? 50 : 0) + (["name", "study", "style"].includes(f.key) ? 1 : 0)})).sort((a, b) => b.score - a.score || b.createdAt - a.createdAt).slice(0, 8).map(({score, ...f}) => f);
    const recall = /(?:прошл|предыдущ|раньше|обсуждали|о ч[её]м говорили|последнем чате)/iu.test(query);
    const snippets = [];
    for (let chatIndex = 0; chatIndex < chats.length; chatIndex++) {
      const chat = chats[chatIndex];
      if (memory.blockedChats.includes(chat.id)) continue;
      for (let index = 0; index < chat.messages.length; index++) {
        if (chat.id === currentId && index >= chat.messages.length - 24) continue;
        const message = chat.messages[index];
        if (message.role !== "user" || memory.ignored.includes(message.id)) continue;
        const relevance = score(message.content);
        if (!relevance && !recall) continue;
        snippets.push({id: message.id, chatId: chat.id, title: chat.title.slice(0, 80), text: message.content.slice(0, 600), score: relevance, order: (chats.length - chatIndex) * 1000 + index});
      }
    }
    const excerpts = snippets.sort((a, b) => b.score - a.score || b.order - a.order).slice(0, 4).map(({score, order, ...item}) => item);
    return {enabled: true, facts, excerpts, saved: saved.filter(id => facts.some(f => f.id === id))};
  }
  const api = {blank, normalize, extract, learn, forget, clear, removeChat, retrieve};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.KrakenMemory = api;
})(typeof window !== "undefined" ? window : globalThis);
