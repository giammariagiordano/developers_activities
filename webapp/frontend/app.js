// ─── API Client ───────────────────────────────────────────────────────────────

const API = {
  async request(method, url, body) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const res = await fetch(url, opts);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    if (res.status === 204) return null;
    return res.json();
  },
  get:    (url)        => API.request('GET', url),
  post:   (url, body)  => API.request('POST', url, body),
  put:    (url, body)  => API.request('PUT', url, body),
  delete: (url)        => API.request('DELETE', url),

  async upload(url, file) {
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch(url, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
};

// ─── State ────────────────────────────────────────────────────────────────────

const state = {
  sessions: [],
  session: null,
  repos: [],
  patterns: [],
  smells: { smells: [], total: 0, page: 1 },
  results: null,
  tab: 'config',
  log: [],
  es: null,          // EventSource
  charts: {},        // Chart.js instances
  presets: [],
  loading: false,
  smellPage: 1,
  smellFilter: '',
};

// ─── Utility ──────────────────────────────────────────────────────────────────

function el(id) { return document.getElementById(id); }

function badge(status) {
  const map = {
    idle: 'badge-idle', running: 'badge-running', pausing: 'badge-running',
    paused: 'badge-paused', completed: 'badge-completed', error: 'badge-error',
    pending: 'badge-pending', scanned: 'badge-scanned', cloning: 'badge-running',
    scanning: 'badge-running', failed: 'badge-error',
  };
  const cls = map[status] || 'badge-idle';
  return `<span class="px-2 py-0.5 rounded text-xs font-medium ${cls}">${status}</span>`;
}

function pct(done, total) {
  if (!total) return 0;
  return Math.round((done / total) * 100);
}

function progressBar(done, total, color = 'blue') {
  const p = pct(done, total);
  return `<div class="w-full bg-gray-200 rounded-full h-2 mt-1">
    <div class="bg-${color}-500 h-2 rounded-full transition-all" style="width:${p}%"></div>
  </div><div class="text-xs text-gray-500 mt-0.5">${done}/${total} (${p}%)</div>`;
}

function addLog(msg, type = 'info') {
  const colors = { info: 'text-gray-600', success: 'text-green-600', error: 'text-red-500', warn: 'text-yellow-600' };
  const ts = new Date().toLocaleTimeString();
  state.log.unshift({ ts, msg, cls: colors[type] || 'text-gray-600' });
  if (state.log.length > 200) state.log.pop();
  renderLog();
}

function renderLog() {
  const container = el('log-container');
  if (!container) return;
  container.innerHTML = state.log.slice(0, 50).map(e =>
    `<div class="log-entry ${e.cls}">[${e.ts}] ${e.msg}</div>`
  ).join('');
}

// ─── Router ───────────────────────────────────────────────────────────────────

async function route() {
  const hash = location.hash || '#/';
  if (hash === '#/' || hash === '') {
    await renderSessions();
  } else if (hash.startsWith('#/session/')) {
    const id = parseInt(hash.split('/')[2]);
    await loadSession(id);
  }
}

window.addEventListener('hashchange', route);
window.addEventListener('load', route);

// ─── Sessions List ────────────────────────────────────────────────────────────

async function renderSessions() {
  stopSSE();
  state.session = null;
  state.sessions = await API.get('/api/sessions');
  state.presets = await API.get('/api/preset-templates');

  el('app').innerHTML = `
    <div class="max-w-5xl mx-auto p-6">
      <div class="flex items-center justify-between mb-6">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">ML Smell Activity Analyzer</h1>
          <p class="text-sm text-gray-500 mt-1">CodeSmile + LLM pipeline for ML-specific code smell classification</p>
        </div>
        <button onclick="showCreateSession()" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 font-medium text-sm">
          + New Session
        </button>
      </div>

      <div id="create-form" class="hidden mb-6 bg-white rounded-xl border p-5 shadow-sm">
        <h2 class="font-semibold text-lg mb-4">Create Session</h2>
        <div class="grid grid-cols-2 gap-4">
          <div class="col-span-2"><label class="text-sm font-medium">Session Name *</label>
            <input id="new-name" class="mt-1 w-full border rounded-lg px-3 py-2 text-sm" placeholder="e.g. Experiment 1 - Few-shot vs Zero-shot" /></div>
          <div><label class="text-sm font-medium">OpenAI API Key</label>
            <input id="new-key" type="password" class="mt-1 w-full border rounded-lg px-3 py-2 text-sm font-mono" placeholder="sk-..." /></div>
          <div><label class="text-sm font-medium">GitHub Token</label>
            <input id="new-gh" type="password" class="mt-1 w-full border rounded-lg px-3 py-2 text-sm font-mono" placeholder="ghp_..." /></div>
          <div><label class="text-sm font-medium">Model</label>
            <select id="new-model" class="mt-1 w-full border rounded-lg px-3 py-2 text-sm">
              <option value="gpt-4o-mini">gpt-4o-mini (recommended)</option>
              <option value="gpt-4o">gpt-4o</option>
              <option value="gpt-4-turbo">gpt-4-turbo</option>
              <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
            </select></div>
          <div><label class="text-sm font-medium">Temperature</label>
            <input id="new-temp" type="number" min="0" max="2" step="0.1" value="0" class="mt-1 w-full border rounded-lg px-3 py-2 text-sm" /></div>
          <div><label class="text-sm font-medium">Runs per commit</label>
            <input id="new-runs" type="number" min="1" max="20" value="10" class="mt-1 w-full border rounded-lg px-3 py-2 text-sm" /></div>
          <div><label class="text-sm font-medium">Max parallel LLM calls</label>
            <input id="new-parallel" type="number" min="1" max="100" value="20" class="mt-1 w-full border rounded-lg px-3 py-2 text-sm" /></div>
        </div>
        <div class="flex gap-3 mt-4">
          <button onclick="createSession()" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 text-sm font-medium">Create</button>
          <button onclick="hideCreateSession()" class="border px-4 py-2 rounded-lg hover:bg-gray-50 text-sm">Cancel</button>
        </div>
      </div>

      <div class="bg-white rounded-xl border shadow-sm overflow-hidden">
        ${state.sessions.length === 0 ? `
          <div class="p-12 text-center text-gray-400">
            <div class="text-4xl mb-3">🧪</div>
            <div class="font-medium">No sessions yet</div>
            <div class="text-sm">Create a session to start analyzing ML code smells</div>
          </div>` : `
          <table class="w-full text-sm">
            <thead class="bg-gray-50 border-b"><tr class="text-left text-xs text-gray-500 uppercase tracking-wide">
              <th class="px-4 py-3">Name</th>
              <th class="px-4 py-3">Status</th>
              <th class="px-4 py-3">Phase 1 (Scan)</th>
              <th class="px-4 py-3">Phase 2 (LLM)</th>
              <th class="px-4 py-3">Model</th>
              <th class="px-4 py-3">Created</th>
              <th class="px-4 py-3"></th>
            </tr></thead>
            <tbody class="divide-y">
              ${state.sessions.map(s => `
                <tr class="hover:bg-gray-50 cursor-pointer" onclick="location.hash='#/session/${s.id}'">
                  <td class="px-4 py-3 font-medium">${s.name}</td>
                  <td class="px-4 py-3">${badge(s.status)}</td>
                  <td class="px-4 py-3">${badge(s.phase1_status)} <span class="text-xs text-gray-400">${s.phase1_done}/${s.phase1_total}</span></td>
                  <td class="px-4 py-3">${badge(s.phase2_status)} <span class="text-xs text-gray-400">${s.phase2_done}/${s.phase2_total}</span></td>
                  <td class="px-4 py-3 font-mono text-xs text-gray-600">${s.model}</td>
                  <td class="px-4 py-3 text-gray-400 text-xs">${new Date(s.created_at).toLocaleDateString()}</td>
                  <td class="px-4 py-3"><button onclick="event.stopPropagation();deleteSession(${s.id})"
                    class="text-red-400 hover:text-red-600 text-xs px-2 py-1 rounded hover:bg-red-50">Delete</button></td>
                </tr>`).join('')}
            </tbody>
          </table>`}
      </div>
    </div>`;
}

function showCreateSession() { el('create-form').classList.remove('hidden'); }
function hideCreateSession() { el('create-form').classList.add('hidden'); }

async function createSession() {
  const name = el('new-name').value.trim();
  if (!name) { alert('Session name required'); return; }
  const s = await API.post('/api/sessions', {
    name,
    openai_api_key: el('new-key').value.trim() || null,
    github_token: el('new-gh').value.trim() || null,
    model: el('new-model').value,
    temperature: parseFloat(el('new-temp').value),
    n_runs: parseInt(el('new-runs').value),
    max_parallel_llm: parseInt(el('new-parallel').value),
  });
  location.hash = `#/session/${s.id}`;
}

async function deleteSession(id) {
  if (!confirm('Delete session and all its data?')) return;
  await API.delete(`/api/sessions/${id}`);
  await renderSessions();
}

// ─── Session Detail ───────────────────────────────────────────────────────────

async function loadSession(id) {
  stopSSE();
  state.session = await API.get(`/api/sessions/${id}`);
  state.repos = await API.get(`/api/sessions/${id}/repos`);
  state.patterns = await API.get(`/api/sessions/${id}/patterns`);
  state.tab = 'config';
  renderSessionPage();
  startSSE(id);
}

function renderSessionPage() {
  const s = state.session;
  if (!s) return;

  const tabs = ['config', 'repos', 'patterns', 'data', 'run', 'results'];
  const tabLabels = { config: 'Config', repos: 'Repositories', patterns: 'Prompt Patterns', data: 'Data', run: 'Run', results: 'Results' };

  el('app').innerHTML = `
    <div class="min-h-screen">
      <!-- Header -->
      <div class="bg-white border-b sticky top-0 z-10">
        <div class="max-w-6xl mx-auto px-6">
          <div class="flex items-center gap-3 py-3">
            <a href="#/" class="text-blue-500 hover:text-blue-700 text-sm">← Sessions</a>
            <span class="text-gray-300">/</span>
            <h2 class="font-semibold text-gray-900">${s.name}</h2>
            <span id="hdr-status">${badge(s.status)}</span>
            <div class="ml-auto flex items-center gap-2 text-xs text-gray-500">
              <span class="font-mono">${s.model}</span>
              <span>temp=${s.temperature}</span>
              <span>runs=${s.n_runs}</span>
            </div>
          </div>
          <!-- Tabs -->
          <div class="flex gap-0 border-b-0">
            ${tabs.map(t => `
              <button onclick="switchTab('${t}')"
                class="px-4 py-2.5 text-sm border-b-2 transition-colors ${state.tab === t ? 'border-blue-500 text-blue-600 font-medium' : 'border-transparent text-gray-500 hover:text-gray-700'}"
                id="tab-${t}">${tabLabels[t]}</button>`).join('')}
          </div>
        </div>
      </div>

      <!-- Content -->
      <div class="max-w-6xl mx-auto px-6 py-6" id="tab-content">
        ${renderTab(state.tab)}
      </div>
    </div>`;
}

function switchTab(tab) {
  state.tab = tab;
  // Update tab styles
  ['config','repos','patterns','data','run','results'].forEach(t => {
    const btn = el(`tab-${t}`);
    if (!btn) return;
    if (t === tab) {
      btn.className = btn.className.replace('border-transparent text-gray-500 hover:text-gray-700', 'border-blue-500 text-blue-600 font-medium');
    } else {
      btn.className = btn.className.replace('border-blue-500 text-blue-600 font-medium', 'border-transparent text-gray-500 hover:text-gray-700');
    }
  });
  el('tab-content').innerHTML = renderTab(tab);
  if (tab === 'results') setTimeout(renderCharts, 100);
  if (tab === 'data') loadSmells();
}

function renderTab(tab) {
  switch (tab) {
    case 'config':   return renderConfig();
    case 'repos':    return renderRepos();
    case 'patterns': return renderPatterns();
    case 'data':     return renderData();
    case 'run':      return renderRun();
    case 'results':  return renderResults();
    default: return '';
  }
}

// ─── Tab: Config ──────────────────────────────────────────────────────────────

function renderConfig() {
  const s = state.session;
  return `
    <div class="max-w-2xl">
      <h3 class="font-semibold text-lg mb-4">Session Configuration</h3>
      <div class="bg-white rounded-xl border shadow-sm p-5 space-y-4">
        <div>
          <label class="text-sm font-medium text-gray-700">Session Name</label>
          <input id="cfg-name" value="${s.name}" class="mt-1 w-full border rounded-lg px-3 py-2 text-sm" />
        </div>

        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="text-sm font-medium text-gray-700">LLM Model</label>
            <select id="cfg-model" class="mt-1 w-full border rounded-lg px-3 py-2 text-sm">
              ${['gpt-4o-mini','gpt-4o','gpt-4-turbo','gpt-3.5-turbo'].map(m =>
                `<option value="${m}" ${s.model === m ? 'selected' : ''}>${m}</option>`
              ).join('')}
            </select>
          </div>
          <div>
            <label class="text-sm font-medium text-gray-700">Temperature <span class="text-gray-400">(0 = deterministic)</span></label>
            <input id="cfg-temp" type="number" min="0" max="2" step="0.1" value="${s.temperature}"
              class="mt-1 w-full border rounded-lg px-3 py-2 text-sm" />
          </div>
          <div>
            <label class="text-sm font-medium text-gray-700">Runs per commit <span class="text-gray-400">(for majority vote)</span></label>
            <input id="cfg-runs" type="number" min="1" max="20" value="${s.n_runs}"
              class="mt-1 w-full border rounded-lg px-3 py-2 text-sm" />
          </div>
          <div>
            <label class="text-sm font-medium text-gray-700">Max parallel LLM calls</label>
            <input id="cfg-parallel" type="number" min="1" max="100" value="${s.max_parallel_llm}"
              class="mt-1 w-full border rounded-lg px-3 py-2 text-sm" />
          </div>
          <div>
            <label class="text-sm font-medium text-gray-700">Default branch</label>
            <input id="cfg-branch" value="${s.branch || 'main'}"
              class="mt-1 w-full border rounded-lg px-3 py-2 text-sm font-mono" />
          </div>
          <div>
            <label class="text-sm font-medium text-gray-700">Max commits <span class="text-gray-400">(empty = all)</span></label>
            <input id="cfg-maxcommits" type="number" min="1" value="${s.max_commits || ''}" placeholder="all"
              class="mt-1 w-full border rounded-lg px-3 py-2 text-sm" />
          </div>
        </div>

        <div>
          <label class="text-sm font-medium text-gray-700">OpenAI API Key</label>
          <input id="cfg-key" type="password" value="${s.openai_api_key || ''}"
            class="mt-1 w-full border rounded-lg px-3 py-2 text-sm font-mono" placeholder="sk-..." />
        </div>
        <div>
          <label class="text-sm font-medium text-gray-700">GitHub Token <span class="text-gray-400">(for private repos & higher rate limits)</span></label>
          <input id="cfg-gh" type="password" value="${s.github_token || ''}"
            class="mt-1 w-full border rounded-lg px-3 py-2 text-sm font-mono" placeholder="ghp_..." />
        </div>

        <div class="flex gap-3 pt-2">
          <button onclick="saveConfig()" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 text-sm font-medium">Save Changes</button>
          <span id="cfg-saved" class="text-green-600 text-sm self-center hidden">✓ Saved</span>
        </div>
      </div>
    </div>`;
}

async function saveConfig() {
  const s = state.session;
  const updated = await API.put(`/api/sessions/${s.id}`, {
    name: el('cfg-name').value.trim(),
    model: el('cfg-model').value,
    temperature: parseFloat(el('cfg-temp').value),
    n_runs: parseInt(el('cfg-runs').value),
    max_parallel_llm: parseInt(el('cfg-parallel').value),
    branch: el('cfg-branch').value.trim(),
    max_commits: el('cfg-maxcommits').value ? parseInt(el('cfg-maxcommits').value) : null,
    openai_api_key: el('cfg-key').value.trim() || null,
    github_token: el('cfg-gh').value.trim() || null,
  });
  state.session = updated;
  el('cfg-saved').classList.remove('hidden');
  setTimeout(() => el('cfg-saved')?.classList.add('hidden'), 2000);
  addLog('Config saved', 'success');
}

// ─── Tab: Repos ───────────────────────────────────────────────────────────────

function renderRepos() {
  return `
    <div class="max-w-3xl space-y-5">
      <h3 class="font-semibold text-lg">Repositories</h3>

      <!-- Manual add (GitHub) -->
      <div class="bg-white rounded-xl border shadow-sm p-4">
        <div class="text-xs font-medium text-gray-500 uppercase tracking-wide mb-3">Add GitHub Repo</div>
        <div class="flex gap-2">
          <input id="repo-owner" placeholder="owner" class="border rounded-lg px-3 py-1.5 text-sm w-32" />
          <span class="self-center text-gray-400">/</span>
          <input id="repo-name" placeholder="repo-name" class="border rounded-lg px-3 py-1.5 text-sm flex-1" />
          <button onclick="addRepo()" class="bg-blue-600 text-white px-3 py-1.5 rounded-lg text-sm hover:bg-blue-700 whitespace-nowrap">Add Repo</button>
        </div>
      </div>

      <!-- Import from CSV -->
      <div class="bg-white rounded-xl border shadow-sm p-4">
        <div class="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">Import GitHub Repos from CSV</div>
        <p class="text-xs text-gray-400 mb-3">Columns: <code class="bg-gray-100 px-1 rounded">repo</code> (owner/name) — or separate <code class="bg-gray-100 px-1 rounded">owner</code> + <code class="bg-gray-100 px-1 rounded">name</code> columns.</p>
        <div class="flex gap-3 items-center">
          <input type="file" id="repos-csv-file" accept=".csv"
            class="text-sm text-gray-500 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border file:text-sm file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100" />
          <button onclick="importReposCSV()" class="bg-blue-600 text-white px-4 py-1.5 rounded-lg text-sm hover:bg-blue-700 whitespace-nowrap">Import CSV</button>
        </div>
      </div>

      <!-- Import from local path -->
      <div class="bg-white rounded-xl border shadow-sm p-4">
        <div class="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">Import from Local Directory</div>
        <p class="text-xs text-gray-400 mb-3">Each subdirectory = one project. Git repos scanned commit-by-commit. Plain dirs scanned as current state (no git).</p>
        <div class="flex gap-3">
          <input id="local-base-path" placeholder="/path/to/projects"
            class="border rounded-lg px-3 py-1.5 text-sm flex-1 font-mono" />
          <button onclick="importLocalPath()" class="bg-green-600 text-white px-4 py-1.5 rounded-lg text-sm hover:bg-green-700 whitespace-nowrap">Scan Dir</button>
        </div>
      </div>

      <!-- Repo list -->
      <div class="bg-white rounded-xl border shadow-sm overflow-hidden">
        <div class="px-4 py-3 border-b flex items-center justify-between">
          <span class="text-sm font-medium text-gray-700">${state.repos.length} repositor${state.repos.length !== 1 ? 'ies' : 'y'}</span>
          ${state.repos.length > 0 ? `<button onclick="clearAllRepos()" class="text-red-400 hover:text-red-600 text-xs">Clear all</button>` : ''}
        </div>
        ${state.repos.length === 0 ? `
          <div class="p-8 text-center text-gray-400 text-sm">No repositories added yet.</div>` : `
          <table class="w-full text-sm">
            <thead class="bg-gray-50 border-b text-xs text-gray-500 uppercase tracking-wide">
              <tr>
                <th class="px-4 py-2 text-left">Repository</th>
                <th class="px-4 py-2 text-left">Local Path</th>
                <th class="px-4 py-2 text-left">Status</th>
                <th class="px-4 py-2 text-left">Commits</th>
                <th class="px-4 py-2"></th>
              </tr>
            </thead>
            <tbody class="divide-y">
              ${state.repos.map(r => `
                <tr>
                  <td class="px-4 py-2.5 font-mono text-sm">${r.owner}/${r.name}</td>
                  <td class="px-4 py-2.5 text-xs text-gray-400 max-w-xs truncate" title="${r.local_path || ''}">${r.local_path ? '📁 ' + r.local_path.split('/').slice(-2).join('/') : '☁ GitHub'}</td>
                  <td class="px-4 py-2.5">${badge(r.status)}</td>
                  <td class="px-4 py-2.5 text-xs text-gray-500">
                    ${r.total_commits ? `${r.current_commit_index}/${r.total_commits}` : '—'}
                    ${r.error_message ? `<div class="text-red-500 mt-0.5 truncate max-w-xs" title="${r.error_message}">${r.error_message.slice(0,60)}</div>` : ''}
                  </td>
                  <td class="px-4 py-2.5">
                    <button onclick="removeRepo(${r.id})" class="text-red-400 hover:text-red-600 text-xs px-2 py-1 rounded hover:bg-red-50">✕</button>
                  </td>
                </tr>`).join('')}
            </tbody>
          </table>`}
      </div>
    </div>`;
}

async function addRepo() {
  const owner = el('repo-owner').value.trim();
  const name = el('repo-name').value.trim();
  if (!owner || !name) { alert('Owner and repo name required'); return; }
  const r = await API.post(`/api/sessions/${state.session.id}/repos`, { owner, name });
  state.repos.push(r);
  el('repo-owner').value = '';
  el('repo-name').value = '';
  el('tab-content').innerHTML = renderTab('repos');
  addLog(`Added repo ${owner}/${name}`, 'success');
}

async function removeRepo(id) {
  if (!confirm('Remove repository?')) return;
  await API.delete(`/api/sessions/${state.session.id}/repos/${id}`);
  state.repos = state.repos.filter(r => r.id !== id);
  el('tab-content').innerHTML = renderTab('repos');
}

async function importReposCSV() {
  const f = el('repos-csv-file')?.files[0];
  if (!f) { alert('Select a CSV file first'); return; }
  const result = await API.upload(`/api/sessions/${state.session.id}/repos/import-csv`, f);
  addLog(`Imported ${result.inserted} repos from CSV`, 'success');
  state.repos = await API.get(`/api/sessions/${state.session.id}/repos`);
  el('tab-content').innerHTML = renderTab('repos');
}

async function importLocalPath() {
  const path = el('local-base-path')?.value.trim();
  if (!path) { alert('Enter a local directory path'); return; }
  try {
    const result = await API.post(`/api/sessions/${state.session.id}/repos/import-local`, { base_path: path });
    addLog(`Imported ${result.inserted} local projects from ${path}`, 'success');
    state.repos = await API.get(`/api/sessions/${state.session.id}/repos`);
    el('tab-content').innerHTML = renderTab('repos');
  } catch(e) { addLog(`Import error: ${e.message}`, 'error'); alert(e.message); }
}

async function clearAllRepos() {
  if (!confirm('Remove all repositories?')) return;
  await Promise.all(state.repos.map(r => API.delete(`/api/sessions/${state.session.id}/repos/${r.id}`)));
  state.repos = [];
  el('tab-content').innerHTML = renderTab('repos');
}

// ─── Tab: Patterns ────────────────────────────────────────────────────────────

function renderPatterns() {
  const VARS = '{diff}, {commit_message}, {smell_type}, {smell_name_suffix}, {issue_context}, {pr_context}';
  return `
    <div class="max-w-4xl">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h3 class="font-semibold text-lg">Prompt Patterns</h3>
          <p class="text-xs text-gray-400 mt-0.5">Up to 10 patterns run simultaneously. Available variables: <code class="bg-gray-100 px-1 rounded text-xs">${VARS}</code></p>
        </div>
        <div class="flex gap-2">
          <select id="preset-select" class="border rounded-lg px-3 py-1.5 text-sm">
            <option value="">— Insert preset —</option>
            ${state.presets.map(p => `<option value="${p.name}">${p.name}</option>`).join('')}
          </select>
          <button onclick="addPreset()" class="border px-3 py-1.5 rounded-lg text-sm hover:bg-gray-50">+ Add Preset</button>
          <button onclick="addEmptyPattern()" class="bg-blue-600 text-white px-3 py-1.5 rounded-lg text-sm hover:bg-blue-700">+ Empty</button>
        </div>
      </div>

      <div id="patterns-list" class="space-y-4">
        ${state.patterns.map((p, i) => renderPatternCard(p, i)).join('')}
      </div>

      ${state.patterns.length === 0 ? `<div class="bg-white rounded-xl border p-8 text-center text-gray-400 text-sm">No patterns yet.</div>` : ''}

      <button onclick="saveAllPatterns()" class="mt-4 bg-green-600 text-white px-5 py-2 rounded-lg hover:bg-green-700 text-sm font-medium">
        Save All Patterns
      </button>
      <span id="patterns-saved" class="text-green-600 text-sm ml-3 hidden">✓ Saved</span>
    </div>`;
}

function renderPatternCard(p, i) {
  return `
    <div class="bg-white rounded-xl border shadow-sm" id="pattern-card-${i}">
      <div class="flex items-center gap-3 px-4 py-3 border-b bg-gray-50 rounded-t-xl">
        <span class="w-6 h-6 bg-blue-100 text-blue-700 rounded-full text-xs flex items-center justify-center font-bold">${i+1}</span>
        <input class="flex-1 font-medium text-sm border-0 bg-transparent outline-none focus:ring-1 focus:ring-blue-300 rounded px-1"
          id="pat-name-${i}" value="${escHtml(p.name)}" placeholder="Pattern name" />
        <label class="flex items-center gap-1.5 text-xs text-gray-500 cursor-pointer">
          <input type="checkbox" id="pat-enabled-${i}" ${p.enabled ? 'checked' : ''} class="rounded" />
          Enabled
        </label>
        <button onclick="removePatternAt(${i})" class="text-red-400 hover:text-red-600 text-xs ml-2">✕ Remove</button>
      </div>
      <div class="p-4">
        <textarea id="pat-template-${i}" rows="10" class="w-full border rounded-lg px-3 py-2 code text-xs resize-y"
          placeholder="Prompt template...">${escHtml(p.template)}</textarea>
      </div>
    </div>`;
}

async function addEmptyPattern() {
  if (state.patterns.length >= 10) { alert('Max 10 patterns'); return; }
  state.patterns.push({ id: null, position: state.patterns.length, name: 'New Pattern', template: '', enabled: true });
  el('tab-content').innerHTML = renderTab('patterns');
}

async function addPreset() {
  if (state.patterns.length >= 10) { alert('Max 10 patterns'); return; }
  const name = el('preset-select').value;
  if (!name) return;
  const preset = state.presets.find(p => p.name === name);
  if (!preset) return;
  state.patterns.push({ id: null, position: state.patterns.length, name, template: preset.template, enabled: true });
  el('tab-content').innerHTML = renderTab('patterns');
}

function removePatternAt(i) {
  state.patterns.splice(i, 1);
  el('tab-content').innerHTML = renderTab('patterns');
}

async function saveAllPatterns() {
  const updates = state.patterns.map((p, i) => ({
    position: i,
    name: el(`pat-name-${i}`)?.value || p.name,
    template: el(`pat-template-${i}`)?.value || p.template,
    enabled: el(`pat-enabled-${i}`)?.checked ?? true,
  }));
  state.patterns = await API.put(`/api/sessions/${state.session.id}/patterns`, updates);
  el('patterns-saved').classList.remove('hidden');
  setTimeout(() => el('patterns-saved')?.classList.add('hidden'), 2000);
  addLog('Patterns saved', 'success');
  el('tab-content').innerHTML = renderTab('patterns');
}

// ─── Tab: Data ────────────────────────────────────────────────────────────────

function renderData() {
  const s = state.session;
  return `
    <div class="max-w-5xl">
      <h3 class="font-semibold text-lg mb-4">Smell Instances Data</h3>

      <div class="grid grid-cols-2 gap-4 mb-6">
        <div class="bg-white rounded-xl border shadow-sm p-4">
          <div class="text-xs text-gray-500 font-medium uppercase tracking-wide mb-2">Upload CSV/JSON (skip Phase 1)</div>
          <p class="text-xs text-gray-400 mb-3">Pre-computed CodeSmile output. Required columns: <code class="bg-gray-100 px-1 rounded">repo</code> (owner/name), <code class="bg-gray-100 px-1 rounded">commit_hash</code>, <code class="bg-gray-100 px-1 rounded">file_path</code>, <code class="bg-gray-100 px-1 rounded">smell_type</code></p>
          <input type="file" id="upload-file" accept=".csv,.json" class="text-sm text-gray-500 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100" />
          <button onclick="uploadSmells()" class="mt-3 bg-blue-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-700 block">Upload</button>
        </div>
        <div class="bg-white rounded-xl border shadow-sm p-4">
          <div class="text-xs text-gray-500 font-medium uppercase tracking-wide mb-2">Status Overview</div>
          <div id="smell-stats" class="text-sm text-gray-500">Loading...</div>
          <div class="mt-3 flex gap-2">
            <button onclick="resetFailed()" class="border px-3 py-1.5 rounded-lg text-sm hover:bg-gray-50">Reset Failed</button>
            <button onclick="clearAllSmells()" class="border border-red-200 text-red-600 px-3 py-1.5 rounded-lg text-sm hover:bg-red-50">Clear All</button>
          </div>
        </div>
      </div>

      <div class="bg-white rounded-xl border shadow-sm overflow-hidden">
        <div class="px-4 py-3 border-b flex items-center gap-3">
          <input id="smell-filter" placeholder="Filter by smell type, repo, status..." onkeyup="filterSmells()"
            class="flex-1 border rounded-lg px-3 py-1.5 text-sm" />
          <select id="status-filter" onchange="filterSmells()" class="border rounded-lg px-3 py-1.5 text-sm">
            <option value="">All statuses</option>
            <option value="pending">Pending</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
        </div>
        <div id="smells-table">
          <div class="p-6 text-center text-gray-400 text-sm">Loading...</div>
        </div>
        <div id="smells-pagination" class="px-4 py-3 border-t flex items-center justify-between text-sm text-gray-500"></div>
      </div>
    </div>`;
}

async function loadSmells(page = 1) {
  state.smellPage = page;
  const filter = el('status-filter')?.value || '';
  const url = `/api/sessions/${state.session.id}/smells?page=${page}&per_page=50${filter ? `&status=${filter}` : ''}`;
  const data = await API.get(url);
  state.smells = data;
  renderSmellsTable(data);
  renderSmellStats();
}

function renderSmellsTable(data) {
  const tbl = el('smells-table');
  if (!tbl) return;
  if (data.smells.length === 0) {
    tbl.innerHTML = '<div class="p-6 text-center text-gray-400 text-sm">No smell instances found.</div>';
    return;
  }
  tbl.innerHTML = `
    <table class="w-full text-xs">
      <thead class="bg-gray-50 border-b text-gray-500 uppercase tracking-wide">
        <tr>
          <th class="px-4 py-2 text-left">Repository</th>
          <th class="px-4 py-2 text-left">Commit</th>
          <th class="px-4 py-2 text-left">File</th>
          <th class="px-4 py-2 text-left">Smell Type</th>
          <th class="px-4 py-2 text-left">Function</th>
          <th class="px-4 py-2 text-left">Status</th>
        </tr>
      </thead>
      <tbody class="divide-y">
        ${data.smells.map(sc => `
          <tr class="hover:bg-gray-50 cursor-pointer" onclick="viewSmellDetail(${sc.id})">
            <td class="px-4 py-2 font-mono">${sc.repo_id}</td>
            <td class="px-4 py-2 font-mono text-gray-500">${sc.commit_hash.slice(0,8)}</td>
            <td class="px-4 py-2 text-gray-600 max-w-xs truncate" title="${sc.file_path}">${sc.file_path}</td>
            <td class="px-4 py-2 font-medium">${sc.smell_type}</td>
            <td class="px-4 py-2 text-gray-500">${sc.function_name || '—'}</td>
            <td class="px-4 py-2">${badge(sc.status)}</td>
          </tr>`).join('')}
      </tbody>
    </table>`;

  const pag = el('smells-pagination');
  if (pag) {
    const totalPages = Math.ceil(data.total / 50);
    pag.innerHTML = `<span>Showing ${data.smells.length} of ${data.total} smell instances</span>
      <div class="flex gap-2">
        ${state.smellPage > 1 ? `<button onclick="loadSmells(${state.smellPage-1})" class="px-3 py-1 border rounded hover:bg-gray-50">Prev</button>` : ''}
        <span>Page ${state.smellPage}/${totalPages}</span>
        ${state.smellPage < totalPages ? `<button onclick="loadSmells(${state.smellPage+1})" class="px-3 py-1 border rounded hover:bg-gray-50">Next</button>` : ''}
      </div>`;
  }
}

async function renderSmellStats() {
  const stats = el('smell-stats');
  if (!stats) return;
  const data = state.smells;
  stats.innerHTML = `
    <div class="space-y-1">
      <div>Total: <strong>${data.total}</strong></div>
    </div>`;
}

function filterSmells() { loadSmells(1); }

async function uploadSmells() {
  const f = el('upload-file')?.files[0];
  if (!f) { alert('Select a file first'); return; }
  const result = await API.upload(`/api/sessions/${state.session.id}/smells/upload`, f);
  addLog(`Uploaded ${result.inserted} smell instances (total: ${result.total})`, 'success');
  loadSmells();
}

async function resetFailed() {
  await API.post(`/api/sessions/${state.session.id}/reset-failed`);
  addLog('Reset failed tasks to pending', 'info');
  loadSmells();
}

async function clearAllSmells() {
  if (!confirm('Delete all smell instances and LLM results?')) return;
  await API.post(`/api/sessions/${state.session.id}/clear-smells`);
  addLog('Cleared all smell data', 'warn');
  loadSmells();
}

async function viewSmellDetail(id) {
  const detail = await API.get(`/api/sessions/${state.session.id}/smells/${id}`);
  const votes = detail.votes.map(v => `
    <div class="border rounded-lg p-3">
      <div class="font-medium text-sm">${v.pattern_name}</div>
      <div class="mt-1 text-sm">Winner: <strong>${v.primary_activity || (v.tied ? 'TIE: ' + JSON.parse(v.tied_activities||'[]').join(', ') : '—')}</strong></div>
      <div class="text-xs text-gray-400">Votes: ${v.vote_count}/${v.total_votes}</div>
    </div>`).join('');

  alert(`Smell #${id}\nType: ${detail.smell_type}\nCommit: ${detail.commit_hash}\n\nVotes:\n${detail.votes.map(v => `${v.pattern_name}: ${v.primary_activity || 'TIE'} (${v.vote_count}/${v.total_votes})`).join('\n')}`);
}

// ─── Tab: Run ─────────────────────────────────────────────────────────────────

function renderRun() {
  const s = state.session;
  const running = ['running', 'pausing'].includes(s.status);
  return `
    <div class="max-w-3xl">
      <h3 class="font-semibold text-lg mb-4">Pipeline Execution</h3>

      <!-- Phase 1 -->
      <div class="bg-white rounded-xl border shadow-sm p-5 mb-4">
        <div class="flex items-center justify-between mb-3">
          <div>
            <h4 class="font-semibold">Phase 1: CodeSmile Scan</h4>
            <p class="text-xs text-gray-400 mt-0.5">Clone repos & scan commit-by-commit for smell-introducing commits</p>
          </div>
          <div class="flex items-center gap-2">
            ${badge(s.phase1_status)}
            ${!running && s.phase1_status !== 'running' ? `
              <button onclick="startPhase1()" class="bg-blue-600 text-white px-4 py-1.5 rounded-lg text-sm hover:bg-blue-700">
                ${s.phase1_status === 'paused' ? '▶ Resume' : '▶ Start'} Phase 1
              </button>` : ''}
          </div>
        </div>
        ${progressBar(s.phase1_done, s.phase1_total || 1, 'blue')}
        <div id="repo-progress" class="mt-3 space-y-1.5">
          ${state.repos.map(r => `
            <div class="flex items-center gap-2 text-xs">
              <span class="font-mono text-gray-600">${r.owner}/${r.name}</span>
              ${badge(r.status)}
              ${r.total_commits ? `<span class="text-gray-400">${r.current_commit_index}/${r.total_commits} commits</span>` : ''}
            </div>`).join('')}
        </div>
      </div>

      <!-- Phase 2 -->
      <div class="bg-white rounded-xl border shadow-sm p-5 mb-4">
        <div class="flex items-center justify-between mb-3">
          <div>
            <h4 class="font-semibold">Phase 2: LLM Classification</h4>
            <p class="text-xs text-gray-400 mt-0.5">
              ${s.n_runs} runs × ${state.patterns.filter(p => p.enabled !== false && p.enabled !== 0).length} patterns in parallel • majority voting
            </p>
          </div>
          <div class="flex items-center gap-2">
            ${badge(s.phase2_status)}
            ${!running && s.phase2_status !== 'running' ? `
              <button onclick="startPhase2()" class="bg-green-600 text-white px-4 py-1.5 rounded-lg text-sm hover:bg-green-700">
                ${s.phase2_status === 'paused' ? '▶ Resume' : '▶ Start'} Phase 2
              </button>` : ''}
          </div>
        </div>
        ${progressBar(s.phase2_done, s.phase2_total || 1, 'green')}
      </div>

      <!-- Controls -->
      <div class="flex gap-3 mb-4">
        ${running ? `
          <button onclick="pauseSession()" class="bg-yellow-500 text-white px-5 py-2 rounded-lg hover:bg-yellow-600 font-medium">
            ⏸ Pause
          </button>` : ''}
      </div>

      <!-- Live Log -->
      <div class="bg-gray-900 rounded-xl p-4">
        <div class="flex items-center justify-between mb-2">
          <span class="text-gray-300 text-xs font-medium uppercase tracking-wide">Live Log</span>
          <button onclick="state.log=[];renderLog()" class="text-gray-500 text-xs hover:text-gray-300">Clear</button>
        </div>
        <div id="log-container" class="space-y-0.5 max-h-64 overflow-y-auto text-gray-300"></div>
      </div>
    </div>`;
}

async function startPhase1() {
  try {
    const r = await API.post(`/api/sessions/${state.session.id}/start-phase1`);
    addLog(`Phase 1 started: ${r.status}`, 'success');
    await refreshSession();
    renderSessionPage();
    switchTab('run');
  } catch(e) { addLog(`Error: ${e.message}`, 'error'); }
}

async function startPhase2() {
  try {
    const r = await API.post(`/api/sessions/${state.session.id}/start-phase2`);
    addLog(`Phase 2 started: ${r.status}`, 'success');
    await refreshSession();
    renderSessionPage();
    switchTab('run');
  } catch(e) { addLog(`Error: ${e.message}`, 'error'); }
}

async function pauseSession() {
  await API.post(`/api/sessions/${state.session.id}/pause`);
  addLog('Pause requested...', 'warn');
}

// ─── Tab: Results ─────────────────────────────────────────────────────────────

function renderResults() {
  const s = state.session;
  return `
    <div class="max-w-5xl">
      <div class="flex items-center justify-between mb-4">
        <h3 class="font-semibold text-lg">Results</h3>
        <div class="flex gap-2">
          <button onclick="refreshResults()" class="border px-3 py-1.5 rounded-lg text-sm hover:bg-gray-50">↻ Refresh</button>
          <a href="/api/sessions/${s.id}/export" class="bg-blue-600 text-white px-4 py-1.5 rounded-lg text-sm hover:bg-blue-700">
            ↓ Export CSV
          </a>
        </div>
      </div>
      <div id="results-content">
        <div class="text-center text-gray-400 py-8">Loading results...</div>
      </div>
    </div>`;
}

async function refreshResults() {
  const data = await API.get(`/api/sessions/${state.session.id}/results/summary`);
  state.results = data;
  renderResultsContent(data);
  setTimeout(renderCharts, 100);
}

function renderResultsContent(data) {
  const rc = el('results-content');
  if (!rc) return;

  const ACTIVITIES = ['Feature Introduction', 'Bug Fixing', 'Enhancement', 'Refactoring'];
  const COLORS = { 'Feature Introduction': '#3b82f6', 'Bug Fixing': '#ef4444', 'Enhancement': '#f59e0b', 'Refactoring': '#8b5cf6', 'Unknown': '#9ca3af' };

  const tokenCost = (inp, out) => {
    const cost = (inp / 1e6 * 0.15) + (out / 1e6 * 0.60);
    return `$${cost.toFixed(4)}`;
  };

  const statusSummary = Object.entries(data.smell_status || {}).map(([k, v]) =>
    `<div class="flex justify-between text-sm"><span>${badge(k)}</span><span class="font-medium">${v}</span></div>`
  ).join('');

  rc.innerHTML = `
    <!-- Stats row -->
    <div class="grid grid-cols-3 gap-4 mb-6">
      <div class="bg-white rounded-xl border shadow-sm p-4">
        <div class="text-xs text-gray-500 uppercase tracking-wide font-medium mb-2">Smell Instances</div>
        <div class="space-y-1.5">${statusSummary || '<div class="text-gray-400 text-sm">No data</div>'}</div>
      </div>
      <div class="bg-white rounded-xl border shadow-sm p-4">
        <div class="text-xs text-gray-500 uppercase tracking-wide font-medium mb-2">Token Usage</div>
        <div class="text-sm space-y-1">
          <div class="flex justify-between"><span>Input tokens</span><span class="font-medium">${(data.tokens?.input || 0).toLocaleString()}</span></div>
          <div class="flex justify-between"><span>Output tokens</span><span class="font-medium">${(data.tokens?.output || 0).toLocaleString()}</span></div>
          <div class="flex justify-between border-t pt-1 mt-1"><span>Est. cost (gpt-4o-mini)</span><span class="font-medium text-green-600">${tokenCost(data.tokens?.input||0, data.tokens?.output||0)}</span></div>
        </div>
      </div>
      <div class="bg-white rounded-xl border shadow-sm p-4">
        <div class="text-xs text-gray-500 uppercase tracking-wide font-medium mb-2">Patterns</div>
        <div class="text-sm space-y-1">
          ${(data.patterns || []).map(p => `
            <div class="flex justify-between items-center">
              <span class="${p.enabled ? '' : 'text-gray-400 line-through'}">${p.pattern_name}</span>
              <span class="text-xs text-gray-500">${p.total} classified</span>
            </div>`).join('')}
        </div>
      </div>
    </div>

    <!-- Per-pattern charts -->
    <div class="grid grid-cols-${Math.min(data.patterns?.length || 1, 2)} gap-4">
      ${(data.patterns || []).map(p => `
        <div class="bg-white rounded-xl border shadow-sm p-4">
          <div class="font-medium text-sm mb-1">${p.pattern_name}</div>
          <div class="text-xs text-gray-400 mb-3">${p.total} smell commits classified${p.tied_count > 0 ? ` • ${p.tied_count} tied` : ''}</div>
          <canvas id="chart-${p.pattern_id}" height="200"></canvas>
          <div class="mt-3 space-y-1">
            ${ACTIVITIES.map(act => {
              const count = p.distribution[act] || 0;
              const pct2 = p.total ? Math.round(count/p.total*100) : 0;
              return `<div class="flex items-center gap-2 text-xs">
                <div class="w-3 h-3 rounded-sm flex-shrink-0" style="background:${COLORS[act]}"></div>
                <span class="flex-1">${act}</span>
                <span class="font-medium">${count}</span>
                <span class="text-gray-400 w-8 text-right">${pct2}%</span>
              </div>`;
            }).join('')}
          </div>
        </div>`).join('')}
    </div>`;
}

function renderCharts() {
  if (!state.results) return;
  const ACTIVITIES = ['Feature Introduction', 'Bug Fixing', 'Enhancement', 'Refactoring'];
  const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#8b5cf6'];

  for (const p of (state.results.patterns || [])) {
    const canvas = el(`chart-${p.pattern_id}`);
    if (!canvas) continue;

    if (state.charts[p.pattern_id]) {
      state.charts[p.pattern_id].destroy();
    }

    state.charts[p.pattern_id] = new Chart(canvas, {
      type: 'doughnut',
      data: {
        labels: ACTIVITIES,
        datasets: [{
          data: ACTIVITIES.map(a => p.distribution[a] || 0),
          backgroundColor: COLORS,
          borderWidth: 2,
          borderColor: '#fff',
        }],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: ctx => `${ctx.label}: ${ctx.raw} (${p.total ? Math.round(ctx.raw/p.total*100) : 0}%)` } },
        },
        cutout: '60%',
      },
    });
  }
}

// ─── SSE Event Handling ───────────────────────────────────────────────────────

function startSSE(sessionId) {
  stopSSE();
  state.es = new EventSource(`/api/sessions/${sessionId}/events`);

  state.es.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      handleSSEMessage(msg);
    } catch (_) {}
  };

  state.es.onerror = () => {
    addLog('SSE connection lost, reconnecting...', 'warn');
    setTimeout(() => startSSE(sessionId), 3000);
  };
}

function stopSSE() {
  if (state.es) { state.es.close(); state.es = null; }
}

function handleSSEMessage(msg) {
  switch (msg.type) {
    case 'init':
    case 'status':
      if (state.session) {
        state.session = { ...state.session, ...msg };
        updateRunTab();
        const hdrStatus = el('hdr-status');
        if (hdrStatus) hdrStatus.innerHTML = badge(msg.status);
      }
      if (msg.type === 'status') addLog(`Status: ${msg.status}${msg.phase ? ` (phase ${msg.phase})` : ''}`, 'info');
      break;

    case 'progress':
      if (state.session) {
        state.session.phase2_done = msg.done;
        state.session.phase2_total = msg.total;
        updateRunTab();
      }
      break;

    case 'phase1_complete':
      addLog(`Phase 1 complete! Found ${msg.smell_count} smell instances.`, 'success');
      refreshSession();
      break;

    case 'phase2_complete':
      addLog('Phase 2 complete! All smell commits classified.', 'success');
      refreshSession();
      break;

    case 'repo_status':
      addLog(`Repo ${msg.repo}: ${msg.status}${msg.total_commits ? ` (${msg.total_commits} commits)` : ''}`, 'info');
      break;

    case 'repo_complete':
      addLog(`✓ ${msg.repo}: ${msg.smells_found} smell instances found`, 'success');
      refreshRepos();
      break;

    case 'repo_error':
      addLog(`✗ ${msg.repo}: ${msg.error}`, 'error');
      refreshRepos();
      break;

    case 'task_failed':
      addLog(`Task ${msg.sc_id} failed: ${msg.error}`, 'error');
      break;

    case 'error':
      addLog(`Error: ${msg.message}`, 'error');
      break;
  }
}

function updateRunTab() {
  if (state.tab !== 'run') return;
  el('tab-content').innerHTML = renderTab('run');
  renderLog();
}

async function refreshSession() {
  state.session = await API.get(`/api/sessions/${state.session.id}`);
  state.repos = await API.get(`/api/sessions/${state.session.id}/repos`);
  if (state.tab === 'run') {
    el('tab-content').innerHTML = renderTab('run');
    renderLog();
  }
  const hdrStatus = el('hdr-status');
  if (hdrStatus) hdrStatus.innerHTML = badge(state.session.status);
}

async function refreshRepos() {
  state.repos = await API.get(`/api/sessions/${state.session.id}/repos`);
  if (state.tab === 'repos') el('tab-content').innerHTML = renderTab('repos');
  if (state.tab === 'run') {
    const rpDiv = el('repo-progress');
    if (rpDiv) rpDiv.innerHTML = state.repos.map(r => `
      <div class="flex items-center gap-2 text-xs">
        <span class="font-mono text-gray-600">${r.owner}/${r.name}</span>
        ${badge(r.status)}
        ${r.total_commits ? `<span class="text-gray-400">${r.current_commit_index}/${r.total_commits} commits</span>` : ''}
      </div>`).join('');
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function escHtml(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// Auto-load results when switching to results tab
const origSwitchTab = switchTab;
window.switchTab = async function(tab) {
  origSwitchTab(tab);
  if (tab === 'results') {
    await refreshResults();
  }
};

// Expose globals for onclick handlers
window.createSession = createSession;
window.hideCreateSession = hideCreateSession;
window.showCreateSession = showCreateSession;
window.deleteSession = deleteSession;
window.saveConfig = saveConfig;
window.addRepo = addRepo;
window.removeRepo = removeRepo;
window.saveAllPatterns = saveAllPatterns;
window.addEmptyPattern = addEmptyPattern;
window.addPreset = addPreset;
window.removePatternAt = removePatternAt;
window.uploadSmells = uploadSmells;
window.resetFailed = resetFailed;
window.clearAllSmells = clearAllSmells;
window.filterSmells = filterSmells;
window.loadSmells = loadSmells;
window.viewSmellDetail = viewSmellDetail;
window.startPhase1 = startPhase1;
window.startPhase2 = startPhase2;
window.pauseSession = pauseSession;
window.refreshResults = refreshResults;
window.importReposCSV = importReposCSV;
window.importLocalPath = importLocalPath;
window.clearAllRepos = clearAllRepos;
