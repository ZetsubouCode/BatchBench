(() => {
  document.addEventListener('DOMContentLoaded', () => {
    const rail = document.getElementById('bb-navbar-tools');
    const toggle = document.getElementById('bb-rail-toggle');
    const backdrop = document.getElementById('bb-rail-backdrop');
    const search = document.getElementById('bb-tool-search');
    if (!rail || !toggle) return;
    const mobile = () => window.matchMedia('(max-width: 991.98px)').matches;
    const close = () => { document.body.classList.remove('bb-rail-open'); backdrop.hidden = true; toggle.setAttribute('aria-expanded', 'false'); };
    const setDesktop = collapsed => {
      document.body.classList.toggle('bb-rail-collapsed', collapsed);
      toggle.setAttribute('aria-expanded', String(!collapsed));
      try { localStorage.setItem('bb-rail-collapsed', collapsed ? '1' : '0'); } catch (_) {}
    };
    if (!mobile()) {
      try { setDesktop(localStorage.getItem('bb-rail-collapsed') === '1'); } catch (_) {}
    } else close();
    toggle.addEventListener('click', () => {
      if (mobile()) {
        const opening = !document.body.classList.contains('bb-rail-open');
        document.body.classList.toggle('bb-rail-open', opening);
        backdrop.hidden = !opening;
        toggle.setAttribute('aria-expanded', String(opening));
        if (opening) search?.focus();
      } else setDesktop(!document.body.classList.contains('bb-rail-collapsed'));
    });
    backdrop.addEventListener('click', close);
    document.addEventListener('keydown', event => { if (event.key === 'Escape' && mobile()) close(); });
    rail.addEventListener('click', event => {
      if (event.target.closest('[data-bs-toggle="tab"]') && mobile()) close();
    });
    document.addEventListener('click', event => {
      const target = event.target.closest('[data-bb-target]')?.dataset.bbTarget;
      if (target) rail.querySelector(`[data-bs-target="${target}"]`)?.click();
    });
    const recentKey = 'bb-recent-tabs';
    const renderRecent = () => {
      const wrap = document.getElementById('bb-recent-tools');
      if (!wrap) return;
      let targets = [];
      try { targets = JSON.parse(localStorage.getItem(recentKey) || '[]'); } catch (_) {}
      wrap.replaceChildren();
      targets.forEach(target => {
        const source = rail.querySelector(`[data-bs-target="${target}"]`);
        if (!source) return;
        const button = document.createElement('button');
        button.type = 'button'; button.className = 'btn btn-sm btn-outline-secondary';
        button.dataset.bbTarget = target;
        button.textContent = source.textContent.trim();
        wrap.appendChild(button);
      });
      if (!wrap.childElementCount) wrap.textContent = 'No recent tools yet.';
    };
    renderRecent();
    rail.addEventListener('shown.bs.tab', event => {
      const target = event.target?.dataset.bsTarget;
      if (!target || ['#tab-home', '#tab-guide'].includes(target)) return;
      let targets = [];
      try { targets = JSON.parse(localStorage.getItem(recentKey) || '[]'); } catch (_) {}
      targets = [target, ...targets.filter(item => item !== target)].slice(0, 4);
      try { localStorage.setItem(recentKey, JSON.stringify(targets)); } catch (_) {}
      renderRecent();
    });
    search?.addEventListener('input', () => {
      const query = search.value.trim().toLowerCase();
      rail.querySelectorAll('[data-bs-toggle="tab"]').forEach(button => {
        button.classList.toggle('d-none', !!query && !button.textContent.toLowerCase().includes(query));
      });
      rail.querySelectorAll('.bb-nav-heading').forEach(heading => {
        let next = heading.nextElementSibling;
        let visible = false;
        while (next && !next.classList.contains('bb-nav-heading')) {
          visible ||= next.matches('[data-bs-toggle="tab"]') && !next.classList.contains('d-none');
          next = next.nextElementSibling;
        }
        heading.classList.toggle('d-none', !!query && !visible);
      });
    });
  });
})();
