(() => {
  let generated = 0;
  function bindLabels(root = document) {
    const labels = root.matches?.('label') ? [root] : [...root.querySelectorAll('label')];
    labels.forEach(label => {
      if (label.htmlFor || label.querySelector('input,select,textarea')) return;
      const sibling = label.nextElementSibling;
      const control = sibling?.matches('input,select,textarea')
        ? sibling : sibling?.querySelector('input:not([type="hidden"]),select,textarea');
      if (!control || control.type === 'hidden') return;
      if (!control.id) control.id = `bb-labeled-control-${++generated}`;
      label.htmlFor = control.id;
    });
  }
  function showFieldError(control) {
    if (!control.matches('input,select,textarea')) return;
    if (!control.id) control.id = `bb-validated-control-${++generated}`;
    const hintId = `${control.id}-validation`;
    let hint = document.getElementById(hintId);
    if (!hint) {
      hint = document.createElement('div');
      hint.id = hintId;
      hint.className = 'invalid-feedback d-block';
      (control.closest('.input-group') || control).insertAdjacentElement('afterend', hint);
    }
    hint.textContent = control.validationMessage;
    control.setAttribute('aria-invalid', 'true');
    const describedBy = (control.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean);
    if (!describedBy.includes(hintId)) control.setAttribute('aria-describedby', [...describedBy, hintId].join(' '));
  }
  function clearFieldError(control) {
    if (!control.matches('input,select,textarea') || !control.validity?.valid) return;
    const hintId = `${control.id}-validation`;
    document.getElementById(hintId)?.remove();
    control.removeAttribute('aria-invalid');
    control.setAttribute('aria-describedby', (control.getAttribute('aria-describedby') || '').split(/\s+/).filter(id => id && id !== hintId).join(' '));
  }
  document.addEventListener('DOMContentLoaded', () => {
    bindLabels();
    document.addEventListener('invalid', event => showFieldError(event.target), true);
    document.addEventListener('input', event => clearFieldError(event.target), true);
    document.addEventListener('change', event => clearFieldError(event.target), true);
    const main = document.querySelector('main');
    if (main) new MutationObserver(records => {
      records.forEach(record => record.addedNodes.forEach(node => {
        if (node.nodeType === Node.ELEMENT_NODE) bindLabels(node);
      }));
    }).observe(main, {childList:true, subtree:true});
  });
  document.addEventListener('bb:tool-loaded', event => bindLabels(event.detail.pane));
})();
