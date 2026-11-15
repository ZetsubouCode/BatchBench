/* Local-only workflow checks. NODE_PATH points at an existing Playwright install. */
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {execFileSync}=require('node:child_process');

(async()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'batchbench-further-'));
  const artifacts=path.resolve('_work/tag-editor-further');fs.mkdirSync(artifacts,{recursive:true});
  execFileSync(process.env.PYTHON||path.resolve('.venv/Scripts/python.exe'),['-c',`
from pathlib import Path
from PIL import Image
import sys
p=Path(sys.argv[1]);(p/'database').mkdir();(p/'dataset/_temp').mkdir(parents=True)
(p/'prompt.txt').write_text('trigger\\n\\nappearance:\\nlong hair, blue eyes\\n')
(p/'database/untouched.bin').write_bytes(b'original')
for rel in ['a','b','nested/c','nested/d','_temp/staged']:
 f=p/'dataset'/rel;f.parent.mkdir(parents=True,exist_ok=True)
 Image.new('RGB',(300,450),(44,65,84)).save(f.with_suffix('.png'))
 f.with_suffix('.txt').write_text('trigger, long hair, blue eyes')
(p/'dataset/_temp/staged.txt').write_text(', '.join(f'tag {i:03}' for i in range(310)))
`,root]);
  const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
  const errors=[],geometry=[];
  try{
    const page=await browser.newPage({viewport:{width:1440,height:900}});page.setDefaultTimeout(15000);page.setDefaultNavigationTimeout(30000);
    page.on('pageerror',e=>errors.push(e.message));page.on('dialog',dialog=>dialog.accept());
    await page.route('**/*',route=>['127.0.0.1','localhost'].includes(new URL(route.request().url()).hostname)?route.continue():route.abort());
    const base=process.env.BATCHBENCH_URL||'http://127.0.0.1:5091';
    const post=async(url,data)=>{const result=await(await page.request.post(base+url,{data})).json();assert.ok(result.ok,JSON.stringify(result));return result;};
    await page.goto(base+'/?tab=tags',{waitUntil:'domcontentloaded'});await page.locator('#te-project-state').waitFor();
    await page.locator('#folder_tags').fill(root);await page.locator('#folder_tags').dispatchEvent('change');
    await page.waitForFunction(()=>document.querySelector('#te-project-state').textContent==='Editor Ready');
    await page.waitForFunction(()=>document.querySelectorAll('#gallery-grid .gallery-card').length===4);
    assert.equal(await page.locator('#tag-editor-sidebar').isVisible(),false);
    await page.waitForFunction(()=>document.querySelector('#te-workspace-toggle').textContent.includes('1 staged'));
    await page.locator('#bb-rail-toggle').click(); // Use the workspace's available desktop width.
    await page.locator('#te-workspace-toggle').click();await page.locator('.te-temp-card').waitFor();
    assert.equal(await page.locator('#tagviz-cloud .tag-chip').count(),40);
    await page.locator('.te-tags-more').click();assert.equal(await page.locator('#tagviz-cloud .tag-chip').count(),80);
    await page.locator('#tagviz-cloud .tag-chip').first().click();
    assert.ok(!(await page.locator('#danbooruOffcanvas').getAttribute('class')).includes('show'));
    const selectedTag=await page.locator('#tagviz-cloud .tag-chip.is-selected .tag-name').innerText();
    await page.locator('.mode-btn[data-mode="insert"]').click();assert.equal(await page.locator('#tags').inputValue(),selectedTag);
    await page.locator('#tags').fill('manual value');
    await page.locator('#tagviz-cloud .tag-chip').nth(1).click();assert.equal(await page.locator('#tags').inputValue(),'manual value');
    assert.match(await page.locator('#tagviz-sync-status').innerText(),/paused/);
    await page.locator('#tagviz-filter').fill('no match');assert.equal(await page.locator('#tagviz-cloud .tag-chip').count(),0);
    assert.match(await page.locator('#te-selected-tags').innerText(),/2 selected/);
    await page.locator('#tagviz-sync').click();assert.match(await page.locator('#tags').inputValue(),/tag/);
    await page.locator('#tags').fill('old tag->new tag');await page.locator('.mode-btn[data-mode="replace"]').click();
    await page.locator('.mode-btn[data-mode="insert"]').click();assert.equal(await page.locator('#tags').inputValue(),'old tag->new tag');
    await page.locator('#tagviz-filter').fill('tag 00');assert.equal(await page.locator('#tagviz-cloud .tag-chip').count(),10);
    await page.locator('#tagviz-min').fill('2');assert.equal(await page.locator('#tagviz-cloud .tag-chip').count(),0);
    await page.locator('#tagviz-min').fill('');await page.locator('#tagviz-filter').fill('');
    await page.locator('#te-tags-selected-only').check();assert.equal(await page.locator('#tagviz-cloud .tag-chip').count(),2);
    await page.locator('#te-tags-selected-only').uncheck();
    await page.locator('#tagviz-clear').click();
    // Resizing, persistence, and representative visual states.
    await page.locator('#te-temp-focus').click();assert.equal(await page.locator('#te-resize').getAttribute('aria-valuenow'),'35');
    await page.locator('#te-source-focus').click();assert.equal(await page.locator('#te-resize').getAttribute('aria-valuenow'),'65');
    await page.locator('#te-resize').focus();await page.keyboard.press('Home');assert.equal(await page.locator('#te-resize').getAttribute('aria-valuenow'),'55');
    const handle=await page.locator('#te-resize').boundingBox();await page.mouse.move(handle.x+10,handle.y+30);await page.mouse.down();await page.mouse.move(handle.x+70,handle.y+30);await page.mouse.up();
    assert.ok(Number(await page.locator('#te-resize').getAttribute('aria-valuenow'))>55);
    await page.locator('#te-resize').dblclick();assert.equal(await page.locator('#te-resize').getAttribute('aria-valuenow'),'55');
    for(const width of [1920,1440,1024,768,375]){
      await page.setViewportSize({width,height:900});await page.evaluate(()=>scrollTo(0,0));
      await page.screenshot({path:path.join(artifacts,`workspace-${width}.png`),fullPage:true});
      const result=await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,source:document.querySelector('.gallery-section').getBoundingClientRect().toJSON(),temp:document.querySelector('.workdir-panel').getBoundingClientRect().toJSON()}));
      assert.ok(result.scrollWidth<=width+1,`workspace overflow at ${width}: ${result.scrollWidth}`);
      if(width<=1024) assert.ok(result.temp.top>=result.source.bottom-1);else assert.ok(Math.abs(result.temp.top-result.source.top)<2);
      geometry.push(result);
    }
    await page.setViewportSize({width:1440,height:900});
    await page.locator('#te-workspace-toggle').click();assert.equal(await page.locator('#tag-editor-sidebar').isVisible(),false);
    assert.ok(fs.existsSync(path.join(root,'dataset/_temp/staged.png')));
    await page.reload({waitUntil:'domcontentloaded'});await page.locator('#te-workspace-toggle').waitFor();assert.equal(await page.locator('#tag-editor-sidebar').isVisible(),false);
    await page.waitForFunction(()=>document.querySelectorAll('#gallery-grid .gallery-card').length===4);
    // Multi-file staging opens the pane; return preserves nested paths and captions.
    await page.locator('#gallery-filename-search').fill('nested');
    await page.locator('#gallery-select-toggle').click();
    await page.locator('#gallery-send-selected-mobile').click();
    await page.waitForFunction(()=>document.querySelectorAll('.te-temp-card').length===3);
    assert.equal(await page.locator('#tag-editor-sidebar').isVisible(),true);
    assert.ok(fs.existsSync(path.join(root,'dataset/_temp/nested/c.txt')));
    await page.locator('#te-temp-folder').selectOption('nested');assert.equal(await page.locator('.te-temp-card').count(),2);
    await page.locator('#te-temp-search').fill('c.png');assert.equal(await page.locator('.te-temp-card').count(),1);
    await page.locator('#te-temp-select').click();await page.locator('#te-temp-return').click();
    await page.waitForFunction(()=>document.querySelector('#workdir-temp-count').textContent.includes('2 staged'));
    assert.ok(fs.existsSync(path.join(root,'dataset/nested/c.txt')));
    await page.waitForFunction(()=>document.querySelector('#tag-editor-layout').getAttribute('aria-busy')!=='true');
    await page.locator('#te-temp-search').fill('');await page.locator('#te-temp-folder').selectOption('');
    // Native internal multi-drag and external payload rejection.
    const drop=async(selector,rels)=>{
      const dt=await page.evaluateHandle(rels=>{const dt=new DataTransfer();dt.setData('application/x-bb-files',JSON.stringify(rels));return dt;},rels);
      await page.locator(selector).dispatchEvent('drop',{dataTransfer:dt});
    };
    await drop('.gallery-section',['_temp/nested/d.png']);
    await page.waitForFunction(()=>document.querySelector('#workdir-temp-count').textContent.includes('1 staged'));
    assert.ok(fs.existsSync(path.join(root,'dataset/nested/d.png')));
    await page.waitForFunction(()=>document.querySelector('#tag-editor-layout').getAttribute('aria-busy')!=='true');
    await drop('.workdir-panel',['a.png','b.png']);
    await page.waitForFunction(()=>document.querySelectorAll('.te-temp-card').length===3);
    const external=await page.evaluateHandle(()=>{const dt=new DataTransfer();dt.setData('text/plain','file:_temp/staged.png');return dt;});
    await page.locator('.gallery-section').dispatchEvent('drop',{dataTransfer:external});
    assert.ok(fs.existsSync(path.join(root,'dataset/_temp/staged.png')));
    await page.waitForFunction(()=>document.querySelector('#tag-editor-layout').getAttribute('aria-busy')!=='true');
    // Preview and all four edit modes retain their safeguards and target.
    await page.locator('.mode-btn[data-mode="insert"]').click();await page.locator('#tags').fill('batch marker');
    const staged=path.join(root,'dataset/_temp/a.txt');const before=fs.readFileSync(staged,'utf8');
    await page.locator('#tags').fill('long hair'); // Existing tag must never turn a preview into Apply.
    await page.locator('#te-preview-batch').click();await page.waitForFunction(()=>document.querySelector('#tags-status').textContent.includes('Preview only'));
    assert.equal(await page.locator('#tagsConfirmModal').isVisible(),false);
    await page.waitForFunction(()=>document.querySelector('#tag-editor-layout').getAttribute('aria-busy')!=='true');
    assert.equal(fs.readFileSync(staged,'utf8'),before);
    const batch=async(mode,tags)=>{
      await page.locator(`.mode-btn[data-mode="${mode}"]`).click();
      if(mode!=='dedup')await page.locator('#tags').fill(tags);
      const response=page.waitForResponse(r=>r.request().method()==='POST'&&new URL(r.url()).pathname==='/');
      await page.locator('#btn-run').click();
      if(mode==='delete'){await page.locator('#tags-confirm-ok').click();await page.locator('#tagsConfirmModal.show').waitFor({state:'hidden'});}
      await response;await page.waitForFunction(()=>document.querySelector('#tags-status').textContent.includes('Inspected')&&document.querySelector('#tag-editor-layout').getAttribute('aria-busy')!=='true');
    };
    await page.locator('#backup').check();await batch('insert','batch marker');assert.match(fs.readFileSync(staged,'utf8'),/batch marker/);
    await batch('replace','batch marker->renamed marker');assert.match(fs.readFileSync(staged,'utf8'),/renamed marker/);
    await batch('delete','renamed marker');assert.ok(!fs.readFileSync(staged,'utf8').includes('renamed marker'));
    fs.appendFileSync(staged,', repeat, repeat');await batch('dedup','');assert.equal(fs.readFileSync(staged,'utf8').split('repeat').length-1,1);
    assert.ok(fs.existsSync(staged+'.bak'));
    // Return All handles staged files only and export still omits session/staging files.
    await page.locator('#workdir-return-temp').click();await page.locator('#tags-confirm-ok').click();
    await page.waitForFunction(()=>document.querySelector('#workdir-temp-count').textContent.includes('0 staged'));
    assert.equal(fs.readFileSync(path.join(root,'database/untouched.bin'),'utf8'),'original');
    await page.locator('#btn-dataset-zip').click();await page.locator('#dataset-zip-download-wrap:not(.d-none)').waitFor();
    // Start a ten-step session through the ordinary setup API.
    const settings=await(await page.request.get(base+'/api/tagging-quiz/settings')).json();
    const segments=settings.settings.segments;
    const started=await post('/api/tagging-quiz/session/start',{project_root:root,image_exts:['.png'],mapping_rows:[...segments.map(s=>({left_sections:[],right_segment:s.id})),...['fixture_detail_a','fixture_detail_b','fixture_detail_c'].map(id=>({left_sections:[id],right_segment:id}))],recommendations:Object.fromEntries(segments.map(s=>[s.id,['long hair','blue eyes']])),session_defaults:{}});
    await page.locator('#tagging-quiz-start').click();await page.locator('#tagging-quiz-modal.show').waitFor();
    const nodes=page.locator('#tagging-desktop-segments button');assert.equal(started.session.quiz_segments.length,10);
    const firstName=await page.locator('#tagging-quiz-image-name').innerText();
    await nodes.nth(9).dispatchEvent('click');assert.equal(await nodes.nth(0).getAttribute('aria-current'),'step');
    await page.locator('#tagging-quiz-save-image').click();assert.equal(await page.locator('#tagging-quiz-image-name').innerText(),firstName);
    // Not Applicable resolves each current step explicitly; revisiting retains later choices.
    const resolveStep=async i=>{
      if(await page.locator('#tagging-quiz-skip').isEnabled())await page.locator('#tagging-quiz-skip').click();
      else {
        if(await page.locator('#tagging-quiz-manual').isVisible()){await page.locator('#tagging-quiz-manual').fill(`resolved step ${i}`);await page.locator('#tagging-quiz-manual-apply').click();}
        else await page.locator('#tagging-quiz-recommended button:not(:disabled)').first().click();
        const current=await page.evaluate(()=>window.BatchBenchDatasetTagEditor.getState().segmentIndex);
        if(current===i && i<9)await page.locator('#tagging-quiz-next-segment').click();
      }
    };
    for(let attempts=0;attempts<10;attempts++){
      const index=await page.evaluate(()=>window.BatchBenchDatasetTagEditor.getState().segmentIndex);if(index>=4)break;await resolveStep(index);
    }
    assert.equal(await nodes.nth(4).getAttribute('aria-current'),'step');
    const progress=await page.locator('.quiz-progress-percent').innerText();
    await nodes.nth(1).click();assert.equal(await nodes.nth(4).getAttribute('aria-disabled'),'false');
    assert.equal(await page.locator('.quiz-progress-percent').innerText(),progress);
    await nodes.nth(4).click();
    for(const width of [1920,1440,1024,768,375]){
      await page.setViewportSize({width,height:900});await page.screenshot({path:path.join(artifacts,`guided-${width}.png`)});
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`guided overflow at ${width}`);
      assert.ok(await page.locator('#tagging-quiz-image-stage').isVisible());
    }
    await page.setViewportSize({width:1440,height:900});
    for(let attempts=0;attempts<10;attempts++){
      const index=await page.evaluate(()=>window.BatchBenchDatasetTagEditor.getState().segmentIndex);await resolveStep(index);if(index===9)break;
    }
    await page.locator('#tagging-quiz-save-image').click();
    await page.waitForFunction(first=>document.querySelector('#tagging-quiz-image-name').textContent!==first,firstName);
    await page.locator('#tagging-quiz-modal .quiz-close-btn').click();await page.locator('#tagging-quiz-modal.show').waitFor({state:'hidden'});
    await page.reload({waitUntil:'domcontentloaded'});await page.locator('#tagging-quiz-start').click();await page.locator('#tagging-quiz-modal.show').waitFor();
    assert.notEqual(await page.locator('#tagging-quiz-image-name').innerText(),firstName);
    assert.equal(errors.length,0,errors.join('\n'));
    fs.writeFileSync(path.join(artifacts,'layout-results.json'),JSON.stringify(geometry,null,2));
    console.log('Passed: split browsers, staging/return, nested pairs, resizing, explorer, autosync, four batch modes, preview, export, guided locking/revisit/save/resume, five viewport widths.');
  } finally {await browser.close();fs.rmSync(root,{recursive:true,force:true});}
})().catch(err=>{console.error(err);process.exitCode=1;});
