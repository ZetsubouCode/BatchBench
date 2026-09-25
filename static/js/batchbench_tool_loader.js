(() => {
  const pending = new Map();
  async function loadPane(pane) {
    const placeholder = pane?.querySelector('[data-lazy-tool]');
    if (!placeholder) return;
    const tool = placeholder.dataset.lazyTool;
    if (pending.has(tool)) return pending.get(tool);
    const task = (async () => {
      try {
        const response = await fetch(`/tool-pane/${encodeURIComponent(tool)}`, {cache:'no-store'});
        if (!response.ok) throw new Error(`Could not load ${tool} (${response.status}).`);
        const html = await response.text();
        const holder = document.createElement('template');
        holder.innerHTML = html;
        pane.replaceChildren(holder.content);
        for (const oldScript of [...pane.querySelectorAll('script')]) {
          const script = document.createElement('script');
          for (const attribute of oldScript.attributes) script.setAttribute(attribute.name, attribute.value);
          if (!oldScript.src) script.textContent = oldScript.textContent;
          const loaded = oldScript.src ? new Promise((resolve, reject) => {
            script.addEventListener('load', resolve, {once:true});
            script.addEventListener('error', () => reject(new Error(`Could not load ${oldScript.src}`)), {once:true});
          }) : null;
          oldScript.replaceWith(script);
          if (loaded) await loaded;
        }
        window.bbPathContext?.bindInputs(pane);
        window.bbInitTooltips?.(pane);
        document.dispatchEvent(new CustomEvent('bb:tool-loaded', {detail:{tool, pane}}));
      } catch (error) {
        pane.innerHTML = '';
        const message = document.createElement('div');
        message.className = 'alert alert-danger'; message.textContent = error.message;
        const retry = document.createElement('button');
        retry.type = 'button'; retry.className = 'btn btn-sm btn-outline-light ms-2'; retry.textContent = 'Retry';
        retry.addEventListener('click', () => { pane.replaceChildren(placeholder); pending.delete(tool); loadPane(pane); });
        message.appendChild(retry); pane.appendChild(message);
      }
    })();
    pending.set(tool, task);
    return task;
  }
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelector('.bb-tool-menu')?.addEventListener('shown.bs.tab', event => {
      loadPane(document.querySelector(event.target?.dataset.bsTarget || ''));
    });
    loadPane(document.querySelector('.tab-pane.active'));
  });
})();
