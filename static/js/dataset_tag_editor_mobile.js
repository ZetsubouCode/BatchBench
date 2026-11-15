(() => {
  'use strict';

  let bridge = null;
  let initialized = false;
  let lastSegmentKey = null;
  let mobileSavePending = false;

  const get = id => document.getElementById(id);
  const mobileQuery = window.matchMedia('(max-width: 767.98px), (max-width: 991.98px) and (orientation: landscape) and (max-height: 500px)');

  function state(){
    return bridge?.getState?.() || {};
  }

  function invoke(name, ...args){
    const action = bridge?.[name];
    if (typeof action !== 'function') return Promise.resolve(null);
    if (name === 'saveImage'){
      if (mobileSavePending) return Promise.resolve(null);
      mobileSavePending = true;
      const saveButton = get('tagging-mobile-save-image');
      if (saveButton){ saveButton.disabled = true; saveButton.textContent = 'Saving…'; }
    }
    return Promise.resolve().then(() => action(...args))
      .catch(error => {
        console.error(`Dataset Tag Editor mobile action failed: ${name}`, error);
        if (name === 'saveImage'){
          const feedback = get('tagging-mobile-feedback');
          if (feedback){
            feedback.textContent = `Save image failed: ${error.message}`;
            feedback.classList.remove('d-none');
            if (mobileQuery.matches) requestAnimationFrame(() => feedback.scrollIntoView({block: 'nearest'}));
          }
        }
        return null;
      })
      .finally(() => {
        if (name === 'saveImage'){
          mobileSavePending = false;
          const saveButton = get('tagging-mobile-save-image');
          if (saveButton) saveButton.textContent = 'Save & Next';
        }
        sync(state());
      });
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
    const index = Math.max(0, Math.min(segments.length - 1, Number(current.segmentIndex || 0)));
    get('tagging-mobile-prev-segment')?.toggleAttribute('disabled', hidden || index <= 0);
    get('tagging-mobile-next-segment')?.toggleAttribute('disabled', hidden || index >= segments.length - 1);
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
    Array.from(select.options).forEach((option,i)=>{option.disabled=i>(current.unlockedSegment ?? 0);});
    select.value = String(index + 1);
    if (label) label.textContent = `Segment ${index + 1} / ${segments.length}`;
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
      if (tagPane && mobileQuery.matches){
        const content = document.querySelector('#tagging-quiz-modal .modal-content');
        const stage = get('tagging-quiz-image-stage');
        if (content && stage) content.scrollTop += tagPane.getBoundingClientRect().top - stage.getBoundingClientRect().bottom - 8;
      } else if (tagPane){
        tagPane.scrollTop = 0;
      }
      const modalBody = document.querySelector('#tagging-quiz-modal .modal-body');
      if (modalBody && tagPane && window.matchMedia('(min-width: 768px) and (max-width: 991.98px)').matches){
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
    get('tagging-mobile-skip')?.toggleAttribute('disabled', !current.canSkipSegment);
    get('tagging-mobile-prev-image')?.toggleAttribute('disabled', !current.canPreviousImage);
    get('tagging-mobile-save-image')?.toggleAttribute('disabled', !current.hasSession || mobileSavePending);
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
        active.scrollIntoView({block: 'nearest'});
      }
    }
  }

  function syncCaptionPlacement(){
    const caption = document.querySelector('#tagging-quiz-modal .quiz-complete-caption');
    const stage = get('tagging-quiz-image-stage');
    const manual = get('tagging-quiz-manual-box');
    if (!caption || !stage || !manual) return;
    if (mobileQuery.matches && caption.previousElementSibling !== manual){
      manual.after(caption);
    } else if (!mobileQuery.matches && caption.previousElementSibling !== stage){
      stage.after(caption);
    }
  }

  function setMoreOpen(open, restoreFocus = false){
    const sheet = get('tagging-mobile-more-sheet');
    const trigger = get('tagging-mobile-more-trigger');
    if (!sheet || !trigger) return;
    if (open && isGuidedTextControl(document.activeElement)) document.activeElement.blur();
    sheet.hidden = !open;
    trigger.setAttribute('aria-expanded', String(open));
    if (open) get('tagging-mobile-more-close')?.focus();
    else if (restoreFocus) trigger.focus({preventScroll: true});
  }

  function isGuidedTextControl(element){
    return !!element?.matches?.('#tagging-quiz-modal input:not([type="checkbox"]):not([type="radio"]), #tagging-quiz-modal textarea');
  }

  function syncGuidedInputState(element = document.activeElement){
    const active = isGuidedTextControl(element);
    document.body.classList.toggle('bb-tag-editor-input-active', active);
    if (!active) return;
    setMoreOpen(false);
    element.scrollIntoView({block: 'nearest'});
  }

  function setGuidedModalState(open){
    document.body.classList.toggle('bb-tag-editor-guided-open', !!open);
    get('tag-editor-layout')?.classList.toggle('bb-guided-open', !!open);
    if (!open){
      setMoreOpen(false);
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
    bind('dataset-mobile-workspace', 'click', () => { window.BatchBenchTagWorkspace?.open(true); scrollToSurface('.workdir-panel'); });
    bind('dataset-mobile-zip', 'click', () => invoke('generateZip'));
    bind('gallery-send-selected-mobile', 'click', () => get('workdir-send-selected')?.click());

    bind('tagging-mobile-prev-image', 'click', () => invoke('previousImage'));
    bind('tagging-mobile-prev-segment', 'click', () => { setMoreOpen(false); invoke('previousSegment'); });
    bind('tagging-mobile-next-segment', 'click', () => { setMoreOpen(false); invoke('nextSegment'); });
    bind('tagging-mobile-save-image', 'click', () => invoke('saveImage'));
    bind('tagging-mobile-skip', 'click', () => { setMoreOpen(false); invoke('skipSegment'); });
    bind('tagging-mobile-retry', 'click', () => { setMoreOpen(false); invoke('retryGuided'); });
    bind('tagging-mobile-later', 'click', () => { setMoreOpen(false); invoke('continueLater'); });
    bind('tagging-mobile-overview', 'click', () => { setMoreOpen(false); invoke('showOverview'); });
    bind('tagging-mobile-segment-select', 'change', event => { setMoreOpen(false); invoke('jumpToSegment', Number(event.target.value || 1)); });
    bind('tagging-mobile-more-trigger', 'click', () => setMoreOpen(true));
    bind('tagging-mobile-more-close', 'click', () => setMoreOpen(false, true));

    bind('tagging-mobile-zoom-out', 'click', () => invoke('setImageZoom', Math.max(1, Number(state().zoom || 1) - 0.2)));
    bind('tagging-mobile-zoom-fit', 'click', () => invoke('resetImageZoom'));
    bind('tagging-mobile-zoom-in', 'click', () => invoke('setImageZoom', Number(state().zoom || 1) + 0.2));
    bind('tagging-mobile-fullscreen', 'click', () => invoke('toggleImageFullscreen'));
    bind('tagging-quiz-completion-zip', 'click', () => invoke('generateZip'));

    const guidedModal = get('tagging-quiz-modal');
    guidedModal?.addEventListener('keydown', event => {
      const sheet = get('tagging-mobile-more-sheet');
      if (sheet?.hidden) return;
      if (event.key === 'Escape'){
        event.preventDefault();
        event.stopPropagation();
        setMoreOpen(false, true);
      }
    });
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
    window.addEventListener('resize', () => { syncVisualViewport(true); syncCaptionPlacement(); });
    syncVisualViewport();
    syncCaptionPlacement();
    sync(state());
  }

  window.addEventListener('bb:dataset-tag-editor-ready', init, {once: true});
  if (document.readyState !== 'loading') init();
})();
