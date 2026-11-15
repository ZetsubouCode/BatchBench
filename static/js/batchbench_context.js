(() => {
  const roles = new Set(['project', 'dataset', 'source', 'output', 'working']);
  const key = role => `bb-path-${role}`;
  const recentKey = role => `bb-path-recent-${role}`;
  const read = name => { try { return localStorage.getItem(name) || ''; } catch (_) { return ''; } };
  const write = (name, value) => { try { localStorage.setItem(name, value); } catch (_) {} };
  const recent = role => { try { return JSON.parse(read(recentKey(role))) || []; } catch (_) { return []; } };
  const roleName = role => ({project:'Project root', dataset:'Dataset folder', source:'Source folder', output:'Output folder', working:'Working folder'})[role] || 'Folder';

  function renderContext() {
    const railProject = document.getElementById('bb-rail-project');
    if (railProject) {
      railProject.textContent = read(key('project')) || 'No project selected';
      railProject.title = read(key('project'));
    }
    for (const role of ['project', 'dataset']) {
      const node = document.getElementById(`bb-context-${role}`);
      if (!node) continue;
      const value = read(key(role));
      node.textContent = value || `No ${role} selected`;
      node.title = value;
      node.dataset.path = value;
    }
  }

  function remember(role, value, derived = false) {
    if (!roles.has(role) || !value) return;
    const oldValue = read(key(role));
    if (role === 'dataset' && !derived) write('bb-dataset-derived-from', '');
    write(key(role), value);
    write(recentKey(role), JSON.stringify([value, ...recent(role).filter(item => item !== value)].slice(0, 6)));
    renderContext();
    if (role === 'dataset') {
      document.querySelectorAll('input[data-path-role="dataset"]').forEach(input => {
        if (input.value.trim() && input.value.trim() !== oldValue) return;
        if (input.value.trim() === value) return;
        input.value = value;
        input.dispatchEvent(new Event('input', {bubbles:true}));
      });
    }
    document.dispatchEvent(new CustomEvent('bb:path-context', {detail:{role, value}}));
  }

  async function deriveProjectDataset(project) {
    const path = project.replace(/[\\/]$/, '') + (project.includes('\\') ? '\\' : '/') + 'dataset';
    try {
      const response = await fetch('/api/list-dir?path=' + encodeURIComponent(path));
      const result = await response.json();
      const note = document.getElementById('bb-context-note');
      if (response.ok && result.ok !== false) {
        const previousProject = read('bb-dataset-derived-from');
        if (!read(key('dataset')) || previousProject) {
          remember('dataset', result.path || path, true);
          write('bb-dataset-derived-from', project);
        }
        if (note) note.textContent = '';
      } else {
        if (read('bb-dataset-derived-from')) {
          const oldValue = read(key('dataset'));
          write(key('dataset'), '');
          write('bb-dataset-derived-from', '');
          renderContext();
          document.querySelectorAll('input[data-path-role="dataset"]').forEach(input => {
            if (input.value.trim() === oldValue) {
              input.value = '';
              input.dispatchEvent(new Event('input', {bubbles:true}));
            }
          });
        }
        if (note) note.textContent = 'This project has no dataset folder.';
      }
    } catch (_) {
      const note = document.getElementById('bb-context-note');
      if (note) note.textContent = 'Dataset folder could not be checked.';
    }
  }

  function bindInputs(root = document) {
    root.querySelectorAll('input[data-path-role]').forEach(input => {
      if (input.dataset.pathBound === '1') return;
      input.dataset.pathBound = '1';
      const role = input.dataset.pathRole;
      if (!roles.has(role)) return;
      if (!input.value.trim() && role !== 'output') {
        const stored = read(key(role));
        if (stored) {
          input.value = stored;
          input.dispatchEvent(new Event('change', {bubbles:true}));
        }
      }
      input.addEventListener('change', () => {
        const value = input.value.trim();
        if (!value) return;
        remember(role, value);
        if (role === 'project') deriveProjectDataset(value);
      });
    });
  }

  function init() {
    bindInputs();
    renderContext();
    if (read(key('project')) && (!read(key('dataset')) || read('bb-dataset-derived-from'))) deriveProjectDataset(read(key('project')));
    document.querySelectorAll('[data-context-copy]').forEach(button => button.addEventListener('click', () => {
      const path = read(key(button.dataset.contextCopy));
      if (path) navigator.clipboard?.writeText(path);
    }));
    document.querySelectorAll('[data-context-change]').forEach(button => button.addEventListener('click', () => {
      const role = button.dataset.contextChange;
      if (role === 'project') {
        document.querySelector('[data-bs-target="#tab-tags"]')?.click();
        const focusProject=()=>{
          const input=document.getElementById('folder_tags');
          if(input && !input.checkVisibility())document.getElementById('left-drawer-project-toggle')?.click();
          input?.focus();
        };
        focusProject();
        document.addEventListener('bb:tool-loaded', event => {
          if (event.detail.tool === 'tags') focusProject();
        }, {once:true});
      } else if (role === 'dataset') {
        document.querySelector('[data-bs-target="#tab-health"]')?.click();
        document.getElementById('health-dataset')?.focus();
      }
    }));
  }
  window.bbPathContext = {recent, readRole: role => read(key(role)), remember, roleName, bindInputs};
  document.addEventListener('DOMContentLoaded', init);
})();
