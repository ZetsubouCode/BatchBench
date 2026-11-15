/* Desktop controls share the existing tagging actions and session model. */
(() => {
  'use strict';
  const get = id => document.getElementById(id);

  window.BatchBenchTagWorkspace = {
    tagLimit:40,
    progress(segments, entry) {
      const resolved = segments.map(segment => {
        const state = entry?.segments?.[segment.id];
        return !!state && !!(state.skipped || state.selected?.length || state.manual?.length);
      });
      const first = resolved.indexOf(false);
      return {resolved, unlocked:first < 0 ? Math.max(0,segments.length - 1) : first};
    },

    install({tagging, management, rail, bridge, move, moveFolder, selectedFolder, renderTags, refreshTemp, previewBatch}) {
      const context = document.createElement('div');
      context.className = 'te-context';
      context.innerHTML = `
        <strong id="te-project-name">Tag Editor</strong>
        <span class="te-context-path" id="te-project-path" tabindex="0">Choose a project</span>
        <span class="te-context-status" id="te-project-state" role="status">Setup needed</span>`;
      rail.prepend(context);
      rail.append(get('te-project-state'));
      tagging.hidden = false;
      management.hidden = false;
      // Keep recommendations, manual entry, and glossary in their original neo order.
      const segmentActions = document.createElement('div');
      segmentActions.className = 'te-segment-actions';
      segmentActions.setAttribute('aria-label', 'Segment navigation');
      segmentActions.append(get('tagging-quiz-prev-segment'), get('tagging-quiz-next-segment'), get('tagging-quiz-skip'));
      get('tagging-quiz-segment-title').parentElement.append(segmentActions);
      get('workdir-send-selected').hidden = true;
      const maintenance = document.createElement('details');
      maintenance.className = 'te-maintenance';
      maintenance.innerHTML = '<summary>Folder maintenance</summary>';
      get('workdir-cleanup-subdirs').before(maintenance);
      maintenance.append(get('workdir-cleanup-subdirs'), get('workdir-delete'));
      const source = management.querySelector('.gallery-section');
      const workspace = management.querySelector('.workdir-panel');
      const right = management.querySelector('.tag-editor-right');
      const shell = management.closest('.tag-editor-shell');
      const batch = management.querySelector('.advanced-tool-panel');
      const exportPanel = management.querySelector('.dataset-export-panel');
      batch.open = true;
      shell.append(batch, exportPanel);
      const heading = workspace.querySelector('.fw-semibold');
      heading.textContent = 'Temporary Workspace';
      const open = document.createElement('button');
      open.type = 'button'; open.id = 'te-workspace-toggle'; open.className = 'btn btn-sm btn-outline-warning';
      open.setAttribute('aria-controls','tag-editor-sidebar');
      source.querySelector('.gallery-selection-toolbar').append(open);
      let visible = false, ratio = 55;
      try { visible = localStorage.getItem('bb.tags.workspace.open') === 'true'; ratio = Number(localStorage.getItem('bb.tags.workspace.ratio')) || 55; } catch (_) {}
      const divider = document.createElement('div'); divider.className = 'te-divider';
      divider.innerHTML = `<button type="button" class="btn btn-sm btn-outline-light" id="te-source-focus" aria-label="Expand Dataset Browser" title="Expand Dataset Browser">←</button>
        <div id="te-resize" role="separator" tabindex="0" aria-label="Resize browser panels" aria-orientation="vertical" aria-valuemin="35" aria-valuemax="65" title="Drag to resize; arrow keys resize; Home resets">⋮</div>
        <button type="button" class="btn btn-sm btn-outline-light" id="te-temp-focus" aria-label="Expand Temporary Workspace" title="Expand Temporary Workspace">→</button>`;
      management.insertBefore(divider,right);
      const size = value => {
        ratio = Math.min(65,Math.max(35,value));
        management.style.setProperty('--te-source-size', `${ratio}fr`);
        management.style.setProperty('--te-temp-size', `${100-ratio}fr`);
        get('te-resize').setAttribute('aria-valuenow',String(Math.round(ratio)));
        try {localStorage.setItem('bb.tags.workspace.ratio',String(ratio));} catch (_) {}
      };
      this.open = value => {
        visible = value; right.hidden = !visible; divider.hidden = !visible;
        management.classList.toggle('te-workspace-open',visible);
        open.setAttribute('aria-expanded',String(visible));
        open.textContent = `${visible ? 'Close' : 'Open'} Batch Workspace · ${this.stagedCount || 0} staged`;
        try {localStorage.setItem('bb.tags.workspace.open',String(visible));} catch (_) {}
      };
      this.open(visible); size(ratio);
      open.addEventListener('click',()=>this.open(!visible));
      const close = document.createElement('button'); close.type='button'; close.className='btn btn-sm btn-outline-light'; close.textContent='Hide workspace';
      close.addEventListener('click',()=>{this.open(false); open.focus();}); workspace.querySelector('.workdir-head').append(close);
      get('te-source-focus').addEventListener('click',()=>size(65));
      get('te-temp-focus').addEventListener('click',()=>size(35));
      const handle = get('te-resize');
      handle.addEventListener('dblclick',()=>size(55));
      handle.addEventListener('keydown',event=>{
        if (!['ArrowLeft','ArrowRight','Home'].includes(event.key)) return;
        event.preventDefault(); size(event.key === 'Home' ? 55 : ratio + (event.key === 'ArrowLeft' ? -5 : 5));
      });
      handle.addEventListener('pointerdown',event=>{if (event.button === 0) {handle.setPointerCapture(event.pointerId); event.preventDefault();}});
      handle.addEventListener('pointermove',event=>{
        if (!handle.hasPointerCapture(event.pointerId)) return;
        const bounds = management.getBoundingClientRect(); size(100*(event.clientX-bounds.left)/bounds.width);
      });
      handle.addEventListener('pointerup',event=>{if(handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);});
      // File moves use the app MIME types only; operating-system drops never become moves.
      for (const [panel,dst,staging] of [[source,'',false],[workspace,'_temp',true]]) {
        panel.dataset.dropLabel = staging ? 'Drop images here to stage' : 'Drop staged images here to return';
        const drop = staging ? get('workdir-temp-drop') : get('workdir-root-drop');
        panel.prepend(drop); drop.textContent=panel.dataset.dropLabel;
        const compatible = event => [...event.dataTransfer.types].some(type=>['application/x-bb-files','application/x-bb-file','application/x-bb-dir'].includes(type));
        panel.addEventListener('dragover',event=>{
          event.preventDefault();
          const valid = compatible(event); event.dataTransfer.dropEffect=valid?'move':'none';
          panel.classList.toggle('te-drop-active',valid); panel.classList.toggle('te-drop-invalid',!valid);
        });
        panel.addEventListener('dragleave',event=>{if(!panel.contains(event.relatedTarget)) panel.classList.remove('te-drop-active','te-drop-invalid');});
        panel.addEventListener('drop',event=>{
          if (event.target.closest('.workdir-row')) return;
          if (!compatible(event) && event.target.closest('#gallery-drop')) return;
          event.preventDefault(); event.stopPropagation(); panel.classList.remove('te-drop-active','te-drop-invalid');
          if (!compatible(event)) return;
          let rels=[];
          try {rels=JSON.parse(event.dataTransfer.getData('application/x-bb-files')||'[]');} catch (_) {}
          const single=event.dataTransfer.getData('application/x-bb-file');
          if(!rels.length && single) rels=[single];
          rels=rels.filter(rel=>staging ? !/^_temp\//i.test(rel) : /^_temp\//i.test(rel));
          if(rels.length) {move(rels,dst);return;}
          const folder=event.dataTransfer.getData('application/x-bb-dir'); if(folder) moveFolder(folder,dst);
        },true);
      }
      document.addEventListener('dragend',()=>[source,workspace].forEach(panel=>panel.classList.remove('te-drop-active','te-drop-invalid')));
      const datasetMaintenance=document.createElement('details'); datasetMaintenance.className='te-maintenance';
      datasetMaintenance.innerHTML='<summary>Dataset folder maintenance</summary><div class="small text-muted">Select a dataset folder below. Delete asks for confirmation.</div><button type="button" id="te-flatten-dataset" class="btn btn-sm btn-outline-warning">Return subfolder files to dataset root</button>';
      datasetMaintenance.append(get('workdir-tree'),get('workdir-actions') || workspace.querySelector('.workdir-actions'),maintenance);
      source.append(datasetMaintenance);
      get('workdir-delete').textContent='Delete selected folder';
      get('workdir-cleanup-subdirs').textContent='Remove empty dataset subfolders';
      workspace.querySelector('.workdir-quick-grid').hidden=true;
      get('workdir-move-all-temp').textContent='Move staged files into a new dataset folder';
      get('workdir-return-temp').textContent='Return All to matching dataset folders';
      const tempTools=document.createElement('div'); tempTools.className='te-temp-tools';
      tempTools.innerHTML=`<input id="te-temp-search" type="search" class="form-control form-control-sm" aria-label="Search staged filenames" placeholder="Find staged filename">
        <select id="te-temp-sort" class="form-select form-select-sm" aria-label="Sort staged images"><option value="az">Name A–Z</option><option value="za">Name Z–A</option><option value="new">Newest first</option></select>
        <select id="te-temp-folder" class="form-select form-select-sm" aria-label="Browse staged folders"><option value="">All staged folders</option></select>
        <button type="button" class="btn btn-sm btn-outline-light" id="te-temp-select">Select visible</button>
        <button type="button" class="btn btn-sm btn-outline-info" id="te-temp-return" disabled>Return Selected</button>
        <span id="te-temp-selection" role="status">0 selected</span>`;
      get('workdir-temp-list').before(tempTools);
      this.tempSelected=new Set();
      ['te-temp-search','te-temp-sort','te-temp-folder'].forEach(id=>get(id).addEventListener('input',()=>this.renderTemp()));
      get('te-temp-select').addEventListener('click',()=>{
        const all=this.tempVisible.every(item=>this.tempSelected.has(item.rel));
        this.tempVisible.forEach(item=>all?this.tempSelected.delete(item.rel):this.tempSelected.add(item.rel)); this.renderTemp();
      });
      get('te-temp-return').addEventListener('click',()=>move([...this.tempSelected].map(rel=>`_temp/${rel}`),''));
      const toFolder=document.createElement('button');toFolder.type='button';toFolder.id='te-temp-to-folder';toFolder.className='btn btn-sm btn-outline-light';toFolder.textContent='Move Selected to Folder';
      toFolder.title='Choose a destination in Dataset folder maintenance first; no folder selected means dataset root.';
      toFolder.addEventListener('click',()=>move([...this.tempSelected].map(rel=>`_temp/${rel}`),selectedFolder()||''));tempTools.append(toFolder);
      const more=document.createElement('button'); more.type='button'; more.id='te-temp-more'; more.className='btn btn-sm btn-outline-light';
      more.addEventListener('click',()=>this.tempData?.more()); get('workdir-temp-list').after(more);
      const selected=document.createElement('div'); selected.id='te-selected-tags'; selected.className='te-selected-tags'; selected.setAttribute('aria-label','Selected explorer tags');
      get('tagviz-cloud').before(selected);
      const only=document.createElement('label'); only.className='form-check-label te-selected-only';
      only.innerHTML='<input type="checkbox" class="form-check-input me-2" id="te-tags-selected-only">Selected Only';
      get('tagviz-filter').closest('.row').append(only);
      get('te-tags-selected-only').addEventListener('change',()=>{this.tagLimit=40;renderTags();});
      get('tagviz-filter').addEventListener('input',()=>{this.tagLimit=40;});
      get('tagviz-sync-status').classList.remove('d-none');
      selected.after(get('tagviz-sync'),get('tagviz-clear'),get('tagviz-sync-status'));
      get('tagviz-scan').textContent='Rescan';
      get('tagviz-stats').closest('.col-md-2').classList.add('te-tag-stats-wrap');
      get('tagviz-stats').parentElement.before(get('tagviz-stats'));
      const preview=document.createElement('button'); preview.type='button'; preview.className='btn btn-outline-light'; preview.id='te-preview-batch'; preview.textContent='Preview Changes';
      preview.addEventListener('click',previewBatch); get('btn-run').before(preview);
      const scope=document.createElement('div'); scope.className='small text-warning'; scope.textContent='Target: dataset/_temp/ · staged captions only';
      get('tags').before(scope);
      get('tags').setAttribute('aria-describedby','mode-desc');
      const updateLabel=()=>{
        const mode=get('mode-select').value;
        document.querySelector('label[for="tags"]').textContent=({insert:'Tags to add',delete:'Tags to remove',replace:'Old tag → New tag mapping',dedup:'No tag input needed — clean duplicate tags'})[mode]||'Tags / mapping';
        document.querySelectorAll('.mode-btn').forEach(btn=>btn.setAttribute('aria-pressed',String(btn.dataset.mode===mode)));
      };
      this.mode=updateLabel; get('mode-select').addEventListener('change',updateLabel); updateLabel();
      const exportNote=document.createElement('p'); exportNote.className='small text-muted mb-2';
      exportNote.textContent='Archive files in dataset/. Everything under dataset/_temp/ is excluded. Other dataset files are included.';
      const warning=document.createElement('div'); warning.id='te-export-warning'; warning.className='small text-warning'; warning.setAttribute('role','status');
      exportPanel.querySelector('.dataset-export-head').after(exportNote); exportPanel.append(warning);
      get('btn-dataset-zip').textContent='Export ZIP'; get('btn-dataset-zip').classList.remove('w-100');
      const refresh=get('workdir-refresh'); refresh.textContent='Rescan files';
      batch.classList.add('workspace-lock-target');
      ['workdir-move-all-temp','workdir-return-temp'].forEach(id => {
        const button = get(id);
        if (button) button.className = 'btn btn-sm btn-outline-light mt-2 w-100';
      });
      const result = document.createElement('div');
      result.id = 'te-file-result'; result.setAttribute('role','status');
      get('workdir-status').after(result);
      const bind = (id, action) => get(id)?.addEventListener('click', action);
      bind('tagging-desktop-zoom-in', () => bridge.setImageZoom(bridge.getState().zoom + .2));
      bind('tagging-desktop-zoom-out', () => bridge.setImageZoom(bridge.getState().zoom - .2));
      bind('tagging-desktop-zoom-fit', bridge.resetImageZoom);
      bind('tagging-desktop-fullscreen', bridge.toggleImageFullscreen);

      const modal = get('tagging-quiz-modal');
      let returnFocus = null;
      modal.addEventListener('show.bs.modal', () => { returnFocus = document.activeElement; });
      modal.addEventListener('shown.bs.modal', () => {
        if (!window.matchMedia('(min-width:768px)').matches) return;
        if (get('tagging-quiz-manual-box')?.classList.contains('d-none')) get('tagging-quiz-search')?.focus();
        else get('tagging-quiz-manual')?.focus();
      });
      modal.addEventListener('hidden.bs.modal', () => {
        const candidates = [returnFocus, get('tagging-quiz-continue'), get('tagging-quiz-start')];
        const target = candidates.find(element => element?.matches?.('button,input,select,textarea,a') && element.checkVisibility() && !element.disabled);
        target?.focus({preventScroll:true});
      });
      window.addEventListener('bb:dataset-tag-editor-state', event => {
        get('tagging-desktop-zoom-value').textContent = `${Math.round((event.detail.zoom || 1) * 100)}%`;
      });
      const statusUpdate=()=>this.updateStatus();
      new MutationObserver(statusUpdate).observe(get('tagging-quiz-autosave'),{childList:true,subtree:true,characterData:true});
      new MutationObserver(statusUpdate).observe(get('tagging-caption-status'),{childList:true,subtree:true,characterData:true});
    },

    project({root, dataset, ready, matches}) {
      if(this.projectRoot !== root) {this.tempSelected?.clear(); this.projectRoot=root;}
      this.projectReady=ready; this.projectMatches=matches;
      const path = get('te-project-path');
      if (path) {
        const name = get('te-project-name');
        if (name) name.textContent = root ? root.replace(/[\\/]+$/, '').split(/[\\/]/).pop() : 'Tag Editor';
        path.textContent = root ? 'Dataset: dataset/ · Batch edits: dataset/_temp/' : 'Choose a project';
        path.title = dataset || root || 'Choose a project';
        path.setAttribute('aria-label', `Working dataset: ${dataset || 'not selected'}`);
      }
      const status = get('te-project-state');
      if (status) {
        status.dataset.ready = String(ready);
        status.dataset.mismatch = String(!matches);
        this.updateStatus();
      }
    },

    updateStatus() {
      const status=get('te-project-state'); if(!status)return;
      const caption=get('tagging-caption-status');
      const draft=get('tagging-quiz-autosave')?.textContent || '';
      const text=caption?.textContent || '';
      const active=!!get('tagging-quiz-image-name')?.textContent?.trim() && get('tagging-quiz-image-name')?.textContent !== '-';
      let state='ready', label='Editor Ready', hint='Project is ready for local editing.';
      if(!this.projectMatches) {state='error';label='Session project mismatch';hint='Reopen a session belonging to this project.';}
      else if(!this.projectReady) {state='warning';label='Setup needed';hint='Choose and initialize a project.';}
      else if(this.operationBusy) {state='busy';label='Processing';hint='Updating local files and browser data. Wait before starting another operation.';}
      else if(this.operationError) {state='error';label='Operation Error';hint=this.operationError;}
      else if(active && /failed|error/i.test(draft+' '+text)) {state='error';label='Save Error';hint=draft+' · '+text;}
      else if(active && /saving|writing/i.test(draft+' '+text)) {state='busy';label='Saving';hint=draft+' · '+text;}
      else if(active && (caption?.dataset.state==='draft' || /pending/i.test(draft))) {state='warning';label='Unsaved Changes';hint='Session drafts and caption files are saved separately. '+text;}
      status.dataset.state=state; status.textContent=label; status.title=hint;
    },

    busy(value) {
      this.operationBusy=value;
      get('tag-editor-layout')?.setAttribute('aria-busy',String(value));
      ['te-temp-return','te-temp-select','workdir-return-temp','te-flatten-dataset','gallery-send-selected-mobile','te-temp-to-folder','te-preview-batch','btn-run'].forEach(id=>{
        const button=get(id); if(!button)return;
        if(value){button.dataset.wasDisabled=String(button.disabled);button.disabled=true;}
        else button.disabled=button.dataset.wasDisabled==='true';
      });
      if(!value)this.renderTemp();
      this.updateStatus();
    },

    moved(items) {
      items.forEach(item=>this.tempSelected?.delete(item.src.replace(/^_temp\//i,'')));
    },

    tempItems(data) {
      this.tempData=data; this.stagedCount=data.total;
      if(data.items.length===data.total){const available=new Set(data.items.map(item=>item.rel));this.tempSelected=new Set([...this.tempSelected].filter(rel=>available.has(rel)));}
      const folders=[...new Set(data.items.map(item=>item.rel.includes('/')?item.rel.slice(0,item.rel.lastIndexOf('/')):''))].sort();
      const selector=get('te-temp-folder'); const value=selector.value; selector.replaceChildren(new Option('All staged folders',''));
      folders.forEach(folder=>selector.add(new Option(folder || 'Staging root',folder || '.'))); selector.value=value;
      if(selector.selectedIndex<0)selector.value='';
      this.renderTemp();
      const toggle=get('te-workspace-toggle'); toggle.textContent=`${toggle.getAttribute('aria-expanded')==='true'?'Close':'Open'} Batch Workspace · ${data.total} staged`;
      get('te-export-warning').textContent=data.total ? `${data.total} staged images will be excluded. Return staged files first to include them.` : 'No staged images excluded.';
      get('workdir-temp-count').textContent=`${data.items.length} loaded / ${data.total} staged images`;
      get('te-temp-more').hidden=data.items.length>=data.total;
      get('te-temp-more').textContent=`Load more staged images (${data.items.length} / ${data.total})`;
    },

    renderTemp() {
      const data=this.tempData; if(!data)return;
      const search=get('te-temp-search').value.trim().toLowerCase(); const folder=get('te-temp-folder').value;
      let items=data.items.filter(item=>(item.rel||'').toLowerCase().includes(search) && (!folder || (folder==='.'?!item.rel.includes('/'):item.rel.startsWith(folder+'/'))));
      const sort=get('te-temp-sort').value;
      items.sort((a,b)=>sort==='new' ? (b.created_ts||0)-(a.created_ts||0) : (sort==='za'?-1:1)*a.rel.localeCompare(b.rel,undefined,{numeric:true}));
      this.tempVisible=items;
      const grid=get('workdir-temp-list'); grid.replaceChildren();
      if(!items.length) {const empty=document.createElement('p'); empty.className='small text-muted'; empty.textContent=data.total?'No staged images match these filters.':'No staged images. Stage selected dataset images or drop them here.';grid.append(empty);}
      items.forEach(item=>{
        const card=document.createElement('article'); card.className='te-temp-card'+(this.tempSelected.has(item.rel)?' is-selected':''); card.draggable=true;
        const label=document.createElement('label');label.className='gallery-select';
        const check=document.createElement('input');check.type='checkbox';check.className='form-check-input';check.checked=this.tempSelected.has(item.rel);check.setAttribute('aria-label',`Select staged ${item.rel}`);
        check.addEventListener('change',()=>{check.checked?this.tempSelected.add(item.rel):this.tempSelected.delete(item.rel);this.renderTemp();});label.append(check,document.createTextNode(' Select'));
        const thumb=document.createElement('button');thumb.type='button';thumb.className='gallery-thumb';thumb.setAttribute('aria-label',`Preview staged ${item.rel}`);
        const image=document.createElement('img');image.loading='lazy';image.alt=item.name;image.src=`/api/tags/image?folder=${encodeURIComponent(data.root)}&area=temp&path=${encodeURIComponent(item.rel)}`;
        image.addEventListener('error',()=>{image.hidden=true;thumb.append(document.createTextNode('Preview unavailable'));});thumb.append(image);thumb.addEventListener('click',()=>data.preview(item.rel));
        const name=document.createElement('div');name.className='gallery-name';name.textContent=item.name;name.title=item.rel;
        const caption=document.createElement('div');caption.className='gallery-meta';caption.textContent=item.has_txt?'Caption present':'Missing caption';
        const tags=document.createElement('div');tags.className='te-gallery-summary';tags.textContent=(item.tags||[]).join(', ') || 'No tags';tags.title=(item.tags||[]).join(', ');
        card.append(label,thumb,name,caption,tags);
        card.addEventListener('dragstart',event=>{
          const rels=(this.tempSelected.has(item.rel)?[...this.tempSelected]:[item.rel]).map(rel=>`_temp/${rel}`);
          event.dataTransfer.setData('application/x-bb-files',JSON.stringify(rels));event.dataTransfer.effectAllowed='move';
        });grid.append(card);
      });
      get('te-temp-selection').textContent=`${this.tempSelected.size} selected`;
      get('te-temp-return').disabled=!this.tempSelected.size;
    },

    selectedTags(tags, remove) {
      const host=get('te-selected-tags'); if(!host)return;host.replaceChildren();
      const count=document.createElement('span');count.textContent=`${tags.size} selected`;host.append(count);
      tags.forEach(tag=>{const button=document.createElement('button');button.type='button';button.className='btn btn-sm btn-outline-info';button.textContent=tag+' ×';button.setAttribute('aria-label',`Remove selection ${tag}`);button.addEventListener('click',()=>remove(tag));host.append(button);});
    },

    segments({segments, entry, index, free, jump}) {
      const nav = get('tagging-desktop-segments');
      if (!nav) return;
      nav.hidden = free;
      const signature = JSON.stringify(segments.map(segment => [segment.id, segment.label]));
      if (nav.dataset.signature !== signature) {
        nav.replaceChildren();
        segments.forEach((segment, i) => {
          const button = document.createElement('button');
          button.type = 'button';
          button.addEventListener('click', () => jump(i + 1));
          const tooltip=document.createElement('span');tooltip.className='te-step-tooltip';tooltip.textContent=segment.label||segment.id;
          button.append(tooltip);
          nav.append(button);
        });
        nav.dataset.signature = signature;
      }
      const {resolved,unlocked}=this.progress(segments,entry);
      const header=get('tagging-quiz-segment-title').parentElement.querySelector('.quiz-segment-banner-label');
      header.textContent=`CURRENT STEP · ${String(index+1).padStart(2,'0')} / ${String(segments.length).padStart(2,'0')}`;
      Array.from(nav.children).forEach((button, i) => {
        const complete = resolved[i];
        button.dataset.complete = String(complete);
        const locked=i>unlocked;button.dataset.locked=String(locked);button.setAttribute('aria-disabled',String(locked));
        button.onclick=event=>{if(locked){event.stopImmediatePropagation();event.preventDefault();}};
        button.setAttribute('aria-label',`${i+1}. ${segments[i].label}: ${i===index?'current':locked?'locked':complete?'completed':'available'}`);
        let number=button.querySelector('.te-step-number');
        if(!number){number=document.createElement('span');number.className='te-step-number';button.prepend(number);}
        number.textContent=i===index?String(i+1):complete&&!locked?'✓':String(i+1);
        if(locked){const lock=document.createElement('i');lock.className='bi bi-lock-fill te-lock';lock.setAttribute('aria-hidden','true');number.append(lock);}
        if (i === index) button.setAttribute('aria-current', 'step');
        else button.removeAttribute('aria-current');
        button.title = `${segments[i].label}: ${locked ? 'resolve earlier steps first' : complete ? 'completed — revisit' : 'choose tags or Not Applicable'}`;
      });
    }
  };
})();
