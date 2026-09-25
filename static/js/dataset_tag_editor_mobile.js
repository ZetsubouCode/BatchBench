(() => {
  'use strict';

  let bridge = null;
  let initialized = false;
  let lastSegmentKey = null;

  const get = id => document.getElementById(id);
  const mobileQuery = window.matchMedia('(max-width: 767.98px)');

  function state(){
    return bridge?.getState?.() || {};
  }

  function invoke(name, ...args){
    const action = bridge?.[name];
    if (typeof action !== 'function') return Promise.resolve(null);
    return Promise.resolve(action(...args))
      .catch(error => {
        console.error(`Dataset Tag Editor mobile action failed: ${name}`, error);
        return null;
      })
      .finally(() => sync(state()));
  }

  function bind(id, eventName, handler){
    get(id)?.addEventListener(eventName, handler);
  }

  function scrollToSurface(selector){
    const target = document.querySelector(selector);
    if (!target) return;
    target.scrollIntoView({behavior: 'smooth', block: 'start'});
    const focusTarget = target.matches('button, input, select, textarea, [tabindex]')
      ? target
      : target.querySelector('button, input, select, textarea, [tabindex]');
    focusTarget?.focus?.({preventScroll: true});
  }

  function syncSegmentSelector(current){
    const wrap = get('tagging-mobile-segment-wrap');
    const select = get('tagging-mobile-segment-select');
    const label = get('tagging-mobile-segment-label');
    const segments = Array.isArray(current.segments) ? current.segments : [];
    const hidden = !!current.freeMode || !segments.length;
    wrap?.classList.toggle('d-none', hidden);
    if (!select || hidden) return;

    const signature = segments.map(segment => `${segment.id}:${segment.label}`).join('|');
    if (select.dataset.signature !== signature){
      select.innerHTML = '';
      segments.forEach((segment, index) => {
        const option = document.createElement('option');
        option.value = String(index + 1);
        option.textContent = segment.label || `Segment ${index + 1}`;
        select.appendChild(option);
      });
      select.dataset.signature = signature;
    }
    const index = Math.max(0, Math.min(segments.length - 1, Number(current.segmentIndex || 0)));
    select.value = String(index + 1);
    if (label) label.textContent = `Segment ${index + 1} / ${segments.length}`;
    get('tagging-mobile-prev-segment')?.toggleAttribute('disabled', index <= 0);
    get('tagging-mobile-next-segment')?.toggleAttribute('disabled', index >= segments.length - 1);
  }

  function sync(current = {}){
    const tagLabel = get('dataset-mobile-tag-label');
    if (tagLabel) tagLabel.textContent = current.resumable ? 'Continue' : 'Start Tagging';
    syncSegmentSelector(current);
    const segment = Array.isArray(current.segments) ? current.segments[current.segmentIndex || 0] : null;
    const imageKey = get('tagging-quiz-image-name')?.textContent || '';
    const segmentKey = `${imageKey}:${current.freeMode ? '__free__' : (segment?.id || '')}`;
    if (lastSegmentKey !== null && segmentKey && segmentKey !== lastSegmentKey){
      const tagPane = document.querySelector('#tagging-quiz-modal .quiz-tag-pane');
      if (tagPane) tagPane.scrollTop = 0;
      const modalBody = document.querySelector('#tagging-quiz-modal .modal-body');
      if (modalBody && tagPane && window.matchMedia('(max-width: 991.98px)').matches){
        const bodyRect = modalBody.getBoundingClientRect();
        const tagRect = tagPane.getBoundingClientRect();
        const stickyImage = document.querySelector('#tagging-quiz-modal .quiz-image-pane');
        const stickyHeight = window.matchMedia('(max-width: 767.98px)').matches
          ? Number(stickyImage?.getBoundingClientRect?.().height || 0)
          : 0;
        modalBody.scrollTop += tagRect.top - bodyRect.top - stickyHeight - 8;
      }
    }
    lastSegmentKey = segmentKey || lastSegmentKey;

    const fullscreen = get('tagging-mobile-fullscreen');
    if (fullscreen){
      fullscreen.disabled = !current.fullscreenAvailable;
      fullscreen.classList.toggle('d-none', !current.fullscreenAvailable);
    }
    get('tagging-mobile-retry')?.classList.toggle('d-none', !current.freeMode);
    get('tagging-mobile-prev-image')?.toggleAttribute('disabled', !current.canPreviousImage);
    get('tagging-mobile-save-image')?.toggleAttribute('disabled', !current.hasSession);
  }

  function syncVisualViewport(ensureFocusedControl = false){
    const viewport = window.visualViewport;
    const height = Number(viewport?.height || window.innerHeight);
    document.documentElement.style.setProperty('--bb-tag-editor-visual-height', `${height}px`);
    const keyboardOpen = mobileQuery.matches && height < window.innerHeight * 0.78;
    document.body.classList.toggle('bb-tag-editor-keyboard-open', keyboardOpen);
    if (keyboardOpen && ensureFocusedControl){
      const active = document.activeElement;
      if (active?.matches?.('#tagging-quiz-modal input, #tagging-quiz-modal textarea, #galleryPreviewModal input')){
        window.setTimeout(() => active.scrollIntoView({block: 'center', behavior: 'smooth'}), 40);
      }
    }
  }

  function isGuidedTextControl(element){
    return !!element?.matches?.('#tagging-quiz-modal input:not([type="checkbox"]):not([type="radio"]), #tagging-quiz-modal textarea');
  }

  function syncGuidedInputState(element = document.activeElement){
    const active = isGuidedTextControl(element);
    document.body.classList.toggle('bb-tag-editor-input-active', active);
    if (!active) return;
    const more = document.querySelector('#tagging-quiz-modal .quiz-mobile-more');
    if (more) more.open = false;
    window.setTimeout(() => {
      element.scrollIntoView({block: 'center', behavior: 'smooth'});
      syncVisualViewport(true);
    }, 40);
  }

  function setGuidedModalState(open){
    document.body.classList.toggle('bb-tag-editor-guided-open', !!open);
    get('tag-editor-layout')?.classList.toggle('bb-guided-open', !!open);
    if (!open){
      document.body.classList.remove('bb-tag-editor-keyboard-open');
      document.body.classList.remove('bb-tag-editor-input-active');
    }
  }

  function init(){
    if (initialized) return;
    bridge = window.BatchBenchDatasetTagEditor;
    if (!bridge) return;
    initialized = true;

    bind('dataset-mobile-tag', 'click', () => invoke('startTagging'));
    bind('dataset-mobile-dataset', 'click', () => scrollToSurface('.gallery-section'));
    bind('dataset-mobile-workspace', 'click', () => scrollToSurface('.workdir-panel'));
    bind('dataset-mobile-zip', 'click', () => invoke('generateZip'));
    bind('gallery-send-selected-mobile', 'click', () => get('workdir-send-selected')?.click());

    bind('tagging-mobile-prev-image', 'click', () => invoke('previousImage'));
    bind('tagging-mobile-prev-segment', 'click', () => invoke('previousSegment'));
    bind('tagging-mobile-next-segment', 'click', () => invoke('nextSegment'));
    bind('tagging-mobile-save-image', 'click', () => invoke('saveImage'));
    bind('tagging-mobile-skip', 'click', () => invoke('skipSegment'));
    bind('tagging-mobile-retry', 'click', () => invoke('retryGuided'));
    bind('tagging-mobile-later', 'click', () => invoke('continueLater'));
    bind('tagging-mobile-overview', 'click', () => invoke('showOverview'));
    bind('tagging-mobile-segment-select', 'change', event => invoke('jumpToSegment', Number(event.target.value || 1)));

    bind('tagging-mobile-zoom-out', 'click', () => invoke('setImageZoom', Math.max(1, Number(state().zoom || 1) - 0.2)));
    bind('tagging-mobile-zoom-fit', 'click', () => invoke('resetImageZoom'));
    bind('tagging-mobile-zoom-in', 'click', () => invoke('setImageZoom', Number(state().zoom || 1) + 0.2));
    bind('tagging-mobile-fullscreen', 'click', () => invoke('toggleImageFullscreen'));
    bind('tagging-quiz-completion-zip', 'click', () => invoke('generateZip'));

    const guidedModal = get('tagging-quiz-modal');
    guidedModal?.addEventListener('shown.bs.modal', () => setGuidedModalState(true));
    guidedModal?.addEventListener('hidden.bs.modal', () => setGuidedModalState(false));
    guidedModal?.addEventListener('focusin', event => syncGuidedInputState(event.target));
    guidedModal?.addEventListener('focusout', () => {
      window.setTimeout(() => syncGuidedInputState(document.activeElement), 0);
    });
    document.addEventListener('fullscreenchange', () => {
      const button = get('tagging-mobile-fullscreen');
      if (button) button.textContent = document.fullscreenElement ? 'Exit Fullscreen' : 'Fullscreen';
    });
    document.addEventListener('webkitfullscreenchange', () => {
      const button = get('tagging-mobile-fullscreen');
      if (button) button.textContent = document.webkitFullscreenElement ? 'Exit Fullscreen' : 'Fullscreen';
    });

    window.addEventListener('bb:dataset-tag-editor-state', event => sync(event.detail || state()));
    window.visualViewport?.addEventListener('resize', () => syncVisualViewport(true));
    window.visualViewport?.addEventListener('scroll', () => syncVisualViewport(false));
    window.addEventListener('resize', () => syncVisualViewport(true));
    syncVisualViewport();
    sync(state());
  }

  window.addEventListener('bb:dataset-tag-editor-ready', init, {once: true});
  if (document.readyState !== 'loading') init();
})();
