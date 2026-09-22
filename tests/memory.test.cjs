const {test} = require('node:test');
const assert = require('node:assert/strict');
const M = require('../public/memory.js');
const message = (id, content) => ({id, role:'user', content});

test('facts survive serialization and are recalled in a different chat', () => {
  let state = M.blank();
  const first = {id:'a', title:'Знакомство', messages:[message('1','Меня зовут Алекс. Я изучаю Python.')]};
  M.learn(state, first.messages[0], first.id, 1);
  state = M.normalize(JSON.parse(JSON.stringify(state)));
  const result = M.retrieve(state, [first, {id:'b', title:'Новый', messages:[]}], 'b', 'Как меня зовут?');
  assert.equal(result.facts[0].value, 'Алекс');
  assert.equal(result.facts.find(f => f.key === 'study').value, 'Python');
});
test('updated name supersedes the old fact and old history source', () => {
  const state = M.blank();
  const chat = {id:'a', title:'Имена', messages:[message('1','Меня зовут Алекс'), message('2','Меня зовут Борис')]};
  chat.messages.forEach((m,i) => M.learn(state,m,'a',i));
  assert.equal(state.facts.length,1);
  assert.equal(state.facts[0].value,'Борис');
  assert.ok(state.ignored.includes('1'));
});
test('forget removes repeated sources without silently relearning', () => {
  const state = M.blank();
  const chat = {id:'a', title:'Знакомство', messages:[message('1','Меня зовут Алекс'),message('2','Меня зовут Алекс')]};
  chat.messages.forEach(m => M.learn(state,m,'a'));
  M.forget(state,state.facts[0].id,[chat]);
  assert.equal(state.facts.length,0);
  chat.messages.forEach(m => M.learn(state,m,'a'));
  assert.equal(state.facts.length,0);
  assert.equal(M.retrieve(state,[chat],'b','Что мы обсуждали раньше?').excerpts.length,0);
});
test('clear excludes old messages while new memories can be learned', () => {
  const state = M.blank();
  const chat = {id:'a',title:'Python',messages:[message('1','Я изучаю Python')]};
  M.learn(state,chat.messages[0],'a');
  M.clear(state,[chat]);
  assert.deepEqual(M.retrieve(state,[chat],'b','Что я изучаю?').facts,[]);
  assert.deepEqual(M.retrieve(state,[chat],'b','Прошлый чат о Python').excerpts,[]);
  chat.messages.push(message('2','Я изучаю Rust'));
  M.learn(state,chat.messages[1],'a');
  assert.equal(state.facts[0].value,'Rust');
});
test('disabled memory sends no facts and learns nothing', () => {
  const state = M.blank(); state.enabled = false;
  M.learn(state,message('1','Меня зовут Алекс'),'a');
  assert.equal(state.facts.length,0);
  assert.deepEqual(M.retrieve(state,[],'a','Привет'),{enabled:false,facts:[],excerpts:[]});
});
test('related old messages include source provenance and bounds', () => {
  const state = M.blank();
  const chats = [{id:'a',title:'Код',messages:[message('1','Напиши цикл на Python')]},{id:'b',title:'Еда',messages:[message('2','Я люблю рис')]}];
  const result = M.retrieve(state,chats,'c','Давай продолжим про Python');
  assert.equal(result.excerpts.length,1);
  assert.equal(result.excerpts[0].chatId,'a');
  assert.equal(result.excerpts[0].text,'Напиши цикл на Python');
  state.blockedChats.push('a');
  assert.equal(M.retrieve(state,chats,'c','Давай продолжим про Python').excerpts.length,0);
});
test('old messages from the active chat beyond the recent context can be retrieved', () => {
  const state=M.blank();
  const chat={id:'a',title:'Длинный чат',messages:[message('old','Обсудим проект Kraken'),...Array.from({length:25},(_,i)=>message(String(i),'другое'))]};
  assert.equal(M.retrieve(state,[chat],'a','проект Kraken').excerpts[0].id,'old');
});
test('deleting a chat removes its facts', () => {
  const state=M.blank(); M.learn(state,message('1','Меня зовут Алекс'),'a');
  M.removeChat(state,'a'); assert.equal(state.facts.length,0);
});
test('explicit notes work, questions and credential-like text are not facts', () => {
  assert.equal(M.extract('Запомни: мой проект называется Kraken')[0].key,'project');
  assert.equal(M.extract('Запомни: дедлайн в пятницу')[0].value,'дедлайн в пятницу');
  assert.equal(M.extract('Как меня зовут?').length,0);
  assert.equal(M.extract('Запомни: пароль abc123').length,0);
});
