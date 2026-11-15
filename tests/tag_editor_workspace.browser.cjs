/* Run against a local test server: NODE_PATH=<Playwright install> node tests/tag_editor_workspace.browser.cjs
 * Uses only a temporary project. No models, accounts, or external network required.
 */
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {execFileSync} = require('node:child_process');

(async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'batchbench-ui-'));
  const screenshots = path.resolve(process.env.BATCHBENCH_SCREENSHOTS || '_work/tag-editor-corrective/C-corrected');
  fs.mkdirSync(screenshots, {recursive:true});
  const python = process.env.PYTHON || path.resolve('.venv/Scripts/python.exe');
  execFileSync(python, ['-c', `
from pathlib import Path
from PIL import Image, ImageDraw
import sys
p=Path(sys.argv[1]); (p/'dataset'/'_temp').mkdir(parents=True); (p/'database').mkdir()
(p/'prompt.txt').write_text('testtrigger\\n\\nappearance:\\nlong hair, blue eyes\\n\\nbackground:\\noutdoors, sky\\n')
for i in range(4):
 im=Image.new('RGB',(640,900),(47+i*10,65,79)); d=ImageDraw.Draw(im)
 d.rectangle((70,120,570,780),outline=(150,175,190),width=3)
 d.text((100,400),f'Inspection fixture {i+1}',fill='white',font_size=38)
 im.save(p/'dataset'/f'image_{i:02d}.png')
 (p/'dataset'/f'image_{i:02d}.txt').write_text('testtrigger, existing tag\\n')
`, root]);
  const extension = path.join(root, '_zoom-extension');
  fs.mkdirSync(extension);
  fs.writeFileSync(path.join(extension,'manifest.json'), JSON.stringify({manifest_version:3,name:'Local zoom verification',version:'1.0',permissions:['tabs'],background:{service_worker:'worker.js'}}));
  fs.writeFileSync(path.join(extension,'worker.js'),'chrome.runtime.onInstalled.addListener(() => {});');
  const browser = await chromium.launchPersistentContext(path.join(root,'_browser-profile'), {
    headless:true, channel:'chromium', ...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {}), viewport:{width:1366,height:768},
    args:[`--disable-extensions-except=${extension}`,`--load-extension=${extension}`],
  });
  const errors = [];
  try {
    const page = await browser.newPage();
    const zoomWorker = browser.serviceWorkers()[0] || await browser.waitForEvent('serviceworker');
    const setBrowserZoom = factor => zoomWorker.evaluate(async ({factor,base}) => {
      const tabs=await chrome.tabs.query({});
      const tab=tabs.find(tab=>tab.url.startsWith(base));
      await chrome.tabs.setZoom(tab.id,factor);
      return chrome.tabs.getZoom(tab.id);
    },{factor,base:process.env.BATCHBENCH_URL || 'http://127.0.0.1:5091'});
    page.setDefaultTimeout(7000);
    page.on('pageerror', error => { errors.push(error.message); console.error('Page error:',error.message); });
    // Keep optional remote references out of this local workflow verification.
    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      return ['127.0.0.1','localhost'].includes(url.hostname) ? route.continue() : route.abort();
    });
    const base = process.env.BATCHBENCH_URL || 'http://127.0.0.1:5091';
    const post = async (url, data) => {
      const response = await page.request.post(base + url, {data});
      const result = await response.json();
      assert.ok(result.ok, JSON.stringify(result));
      return result;
    };
    await page.goto(base + '/');
    await page.locator('.bb-tool-rail [data-bs-target="#tab-tags"]').click();
    await page.locator('#te-project-state').waitFor();
    assert.equal(await page.locator('#te-project-state').innerText(), 'Setup needed');
    // Labels and validation occupy shared grid rows, including at narrower zoom-equivalent widths.
    for (const width of [1366,1093,911]) {
      await page.setViewportSize({width,height:768});
      await page.locator('label[for="exts"]').evaluate(label => { label.textContent='Image extensions included when scanning the selected project'; });
      await page.locator('#folder_tags').evaluate(input => {
        const message=document.createElement('div'); message.className='invalid-feedback d-block'; message.textContent='Choose an available project folder.';
        input.closest('.input-group').after(message);
      });
      const tops = await page.evaluate(() => [document.querySelector('#folder_tags'),document.querySelector('#exts')].map(input => input.getBoundingClientRect().top));
      assert.ok(Math.abs(tops[0]-tops[1])<2,`wrapped labels misaligned at ${width}: ${tops}`);
      await page.locator('#tag-editor-layout .invalid-feedback').evaluate(el => el.remove());
    }
    await page.setViewportSize({width:1366,height:768});
    await page.locator('label[for="exts"]').evaluate(label => {label.textContent='Image extensions';});
    await page.screenshot({path:path.join(screenshots,'setup.png')});
    await page.locator('#folder_tags').fill(root);
    await page.locator('#folder_tags').dispatchEvent('change');
    await page.waitForFunction(() => document.querySelector('#te-project-state')?.textContent === 'Editor Ready');
    await page.locator('.gallery-card').first().waitFor();
    assert.equal(await page.locator('.gallery-card').count(),4);
    fs.unlinkSync(path.join(root,'dataset/image_03.txt'));
    await page.locator('#gallery-refresh').click();
    await page.waitForFunction(()=>[...document.querySelectorAll('.gallery-meta')].some(el=>el.textContent==='Missing caption'));
    assert.equal(await page.locator('#te-tag-images').count(),0);
    assert.ok(await page.locator('#tagging-quiz-start').isVisible());
    assert.equal(await page.locator('#gallery-grid-cols').inputValue(),'auto');
    await page.locator('#gallery-grid-cols').selectOption('2');
    assert.equal(await page.locator('#gallery-grid').evaluate(el=>getComputedStyle(el).gridTemplateColumns.split(' ').length),2);
    await page.locator('#gallery-grid-cols').selectOption('auto');
    await page.locator('#gallery-filename-search').click();
    await page.locator('#gallery-filename-search').fill('image_01');
    assert.equal(await page.locator('.gallery-card').count(),1);
    await page.locator('#gallery-filename-search').fill('no_match');
    assert.ok((await page.locator('#gallery-grid').innerText()).includes('No matching'));
    await page.locator('#gallery-clear-filters').click();
    await page.locator('#gallery-caption-filter').selectOption('with');
    assert.equal(await page.locator('.gallery-card').count(),3);
    await page.locator('#gallery-caption-filter').selectOption('missing');
    assert.equal(await page.locator('.gallery-card').count(),1);
    await page.locator('#gallery-clear-filters').click();
    await page.screenshot({path:path.join(screenshots,'manage-1366.png')});
    fs.writeFileSync(path.join(root,'dataset/image_03.txt'),'testtrigger, existing tag\n');
    await page.locator('#gallery-refresh').click();
    await page.waitForFunction(()=>[...document.querySelectorAll('.gallery-meta')].every(el=>el.textContent!=='Missing caption'));
    // Seed a small ordinary session through the same API used by Setup.
    const settings = await (await page.request.get(base + '/api/tagging-quiz/settings')).json();
    const segments = settings.settings.segments;
    assert.ok(segments.length > 1);
    const recommendations = Object.fromEntries(segments.map(s => [s.id, ['long hair','blue eyes','outdoors']]));
    const started = await post('/api/tagging-quiz/session/start', {project_root:root,image_exts:['.png'],
      mapping_rows:segments.map(s => ({left_sections:[],right_segment:s.id})), recommendations, session_defaults:Object.fromEntries(segments.map((s,i)=>[s.id,[`baseline choice ${i}`]]))});
    await page.locator('#tagging-quiz-start').click();
    await page.locator('#tagging-quiz-modal.show').waitFor();
    await page.locator('#tagging-quiz-image-stage img:not([hidden])').waitFor();
    const review = await (await page.request.get(base + '/api/settings/review-quiz')).json();
    const manualIndex = started.session.quiz_segments.findIndex(segment => review.config.quiz_review.steps.some(step => step.id === segment.id && step.mode === 'multi'));
    assert.ok(manualIndex >= 0);
    const otherIndex = manualIndex === 0 ? 1 : 0;
    await page.locator('#tagging-desktop-segments button').nth(manualIndex).click();
    const manual = page.locator('#tagging-quiz-manual');
    const name = () => page.locator('#tagging-quiz-image-name').innerText();
    const first = await name();
    const original = fs.readFileSync(path.join(root,first.replace(/\.png$/,'.txt')),'utf8');
    const recommendation=page.locator('#tagging-quiz-recommended button').first();
    const recommendedTag=await recommendation.locator('span').first().innerText();
    await recommendation.focus(); await recommendation.press('Enter');
    await page.locator('#tagging-quiz-selected button').filter({hasText:recommendedTag}).waitFor();
    await page.locator('#tagging-quiz-selected button').filter({hasText:recommendedTag}).click();
    assert.equal(await page.locator('#tagging-quiz-glossary-panel').evaluate(el=>el.parentElement.tagName),'DIV');
    await manual.fill('custom tag, custom_tag, another tag');
    await page.locator('#tagging-quiz-manual-apply').click();
    await page.waitForFunction(() => document.querySelector('#tagging-quiz-autosave').textContent.includes('Draft saved'));
    assert.equal(fs.readFileSync(path.join(root, first.replace(/\.png$/,'.txt')),'utf8'),original,'draft must not write caption');
    assert.ok((await page.locator('#tagging-caption-status').innerText()).includes('unsaved'));
    assert.equal(await page.locator('#tagging-quiz-selected button').filter({hasText:'custom tag'}).count(),1);
    await page.locator('#tagging-quiz-selected button').filter({hasText:'another tag'}).focus();
    await page.locator('#tagging-quiz-selected button').filter({hasText:'another tag'}).press('Enter');
    await page.locator('#tagging-desktop-segments button').nth(otherIndex).click();
    assert.equal(await page.locator('#tagging-desktop-segments [aria-current="step"]').count(),1);
    await page.locator('#tagging-desktop-segments button').nth(manualIndex).click();
    await manual.fill('cursor test');
    await manual.press('ArrowRight');
    assert.equal(await name(), first, 'typing must not navigate');
    await page.locator('#tagging-desktop-segments button').nth(otherIndex).click();
    await page.waitForFunction(index => document.querySelectorAll('#tagging-desktop-segments button')[index].getAttribute('aria-current') === 'step', otherIndex);
    await page.waitForFunction(() => document.querySelector('#tagging-quiz-autosave').textContent.includes('Draft saved'));
    const draft = JSON.parse(fs.readFileSync(path.join(root,'dataset/_temp/tagging_session.json'),'utf8'));
    assert.deepEqual(draft.images[first].segments[started.session.quiz_segments[manualIndex].id].manual,['cursor test']);
    assert.ok(!(draft.images[first].segments[started.session.quiz_segments[otherIndex].id].manual || []).includes('cursor test'),'manual text must stay in its original segment');
    await page.locator('#tagging-desktop-segments button').nth(manualIndex).click();
    await manual.waitFor({state:'visible'});
    await page.locator('#tagging-quiz-manual-clear').click();
    // The image node must survive segment and chip changes.
    await page.locator('#tagging-quiz-image-stage img').evaluate(img => { img.dataset.testIdentity='same'; });
    await page.locator('#tagging-desktop-segments button').nth(otherIndex).click();
    assert.equal(await page.locator('#tagging-quiz-image-stage img').getAttribute('data-test-identity'),'same');
    await page.locator('#tagging-desktop-zoom-in').click();
    assert.equal(await page.locator('#tagging-desktop-zoom-value').innerText(),'120%');
    await page.locator('#tagging-desktop-zoom-fit').click();
    await page.locator('#tagging-desktop-segments button').nth(manualIndex).click();
    await manual.waitFor({state:'visible'});

    const layouts = [];
    for (const [width,height] of [[1366,768],[1440,900],[1920,1080],[1093,614],[911,512]]) {
      await page.setViewportSize({width,height});
      await page.screenshot({path:path.join(screenshots,`guided-${width}x${height}.png`)});
      const geometry = await page.evaluate(() => {
        const r = selector => {const x=document.querySelector(selector).getBoundingClientRect();return {x:x.x,y:x.y,width:x.width,height:x.height,bottom:x.bottom,right:x.right};};
        return {overflow:document.documentElement.scrollWidth>innerWidth,stage:r('#tagging-quiz-image-stage'),save:r('#tagging-quiz-save-image'),pane:r('.quiz-tag-pane'),height:innerHeight};
      });
      assert.equal(geometry.overflow,false,`horizontal overflow at ${width}`);
      assert.ok(geometry.stage.height >= 140, `inspection image too short at ${width}`);
      assert.ok(geometry.save.bottom <= height && geometry.save.y >= 0,`save not visible at ${width}`);
      assert.ok(geometry.pane.right <= width,`editor clipped at ${width}`);
      layouts.push({width,height,...geometry});
    }
    await page.setViewportSize({width:1366,height:768});
    for (const factor of [1.25,1.5]) {
      assert.equal(await setBrowserZoom(factor),factor);
      await page.waitForFunction(factor => Math.abs(innerWidth - 1366/factor)<2,factor);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
      assert.ok(await page.locator('#tagging-quiz-save-image').evaluate(el=>el.getBoundingClientRect().bottom<=innerHeight));
      await page.screenshot({path:path.join(screenshots,`guided-browser-zoom-${factor*100}.png`)});
    }
    await setBrowserZoom(1);
    await manual.fill(Array.from({length:100},(_,i)=>`a long descriptive custom tag ${i}`).join(', '));
    await page.screenshot({path:path.join(screenshots,'guided-long-tags.png')});
    assert.ok(await page.locator('#tagging-quiz-save-image').evaluate(el => {const r=el.getBoundingClientRect();return r.top>=0&&r.bottom<=innerHeight;}));
    await page.locator('#tagging-quiz-manual-clear').click();
    // A failed caption request must leave the current image and choices open.
    await page.route('**/api/tagging-quiz/image/save', route => route.fulfill({status:500,contentType:'application/json',body:JSON.stringify({ok:false,error:'Simulated write failure'})}));
    await page.locator('#tagging-quiz-save-image').click();
    await page.waitForFunction(() => document.querySelector('#tagging-caption-status').dataset.state === 'error');
    assert.equal(await name(),first);
    assert.equal(fs.readFileSync(path.join(root,first.replace(/\.png$/,'.txt')),'utf8'),original);
    await page.unroute('**/api/tagging-quiz/image/save');
    let writes=0;
    page.on('request', req => {if(req.url().endsWith('/api/tagging-quiz/image/save')) writes++;});
    await page.locator('#tagging-quiz-save-image').dblclick();
    await page.waitForFunction(old => document.querySelector('#tagging-quiz-image-name').textContent !== old,first);
    assert.equal(writes,1,'double save must be ignored');
    const caption = fs.readFileSync(path.join(root,first.replace(/\.png$/,'.txt')),'utf8');
    assert.ok(caption.includes('custom tag'));
    assert.ok(!caption.includes('another tag'));
    assert.ok(fs.existsSync(path.join(root,first.replace(/\.png$/,'.txt.bak'))));
    await page.locator('#tagging-quiz-prev-image').click();
    await page.waitForFunction(old => document.querySelector('#tagging-quiz-image-name').textContent === old,first);
    await manual.fill('typing stays here');
    await manual.press('ArrowRight');
    await manual.press('Shift+D');
    assert.equal(await name(),first,'free tagging shortcuts must leave text editing alone');
    // Closing after a failed draft write must leave the editor open.
    await page.route('**/api/tagging-quiz/session/save', route => route.fulfill({status:500,contentType:'application/json',body:JSON.stringify({ok:false,error:'Simulated draft failure'})}));
    await manual.fill('recoverable draft');
    await page.locator('#tagging-quiz-modal .quiz-close-btn').click();
    await page.waitForFunction(() => document.querySelector('#tagging-quiz-autosave').textContent.includes('Draft save failed'));
    assert.ok(await page.locator('#tagging-quiz-modal').evaluate(el => el.classList.contains('show')));
    await page.unroute('**/api/tagging-quiz/session/save');
    await page.locator('#tagging-quiz-modal .quiz-close-btn').click();
    await page.locator('#tagging-quiz-modal').waitFor({state:'hidden'});
    await page.waitForFunction(() => ['tagging-quiz-start','tagging-quiz-continue'].includes(document.activeElement.id));
    assert.ok(fs.readFileSync(path.join(root,'dataset/_temp/tagging_session.json'),'utf8').includes('recoverable draft'));
    // Browse, stage pairs, perform every batch operation, then return their locations.
    await page.waitForFunction(()=>!document.querySelector('#gallery-refresh').disabled && !document.querySelector('#tagviz-scan').disabled);
    const cards = page.locator('.gallery-card');
    await cards.first().evaluate(card => {card.dataset.identity='preserved';});
    await cards.nth(0).locator('input[type="checkbox"]').check();
    await cards.nth(1).locator('input[type="checkbox"]').check();
    assert.equal(await cards.first().getAttribute('data-identity'),'preserved','selection should not rebuild thumbnails');
    assert.equal(await page.locator('#gallery-selection-count').innerText(),'2 selected');
    await page.locator('#gallery-caption-filter').selectOption('selected');
    assert.equal(await page.locator('.gallery-card').count(),2);
    await page.locator('#gallery-clear-filters').click();
    assert.equal(await page.locator('#workdir-send-selected').isVisible(),false);
    page.once('dialog', dialog => dialog.accept());
    await Promise.all([
      page.waitForResponse(response => response.url().endsWith('/api/tags/move-files') && response.request().method()==='POST'),
      page.locator('#gallery-send-selected-mobile').click()
    ]);
    for (const stem of ['image_00','image_01']) {
      for (const ext of ['.png','.txt']) {
        assert.ok(fs.existsSync(path.join(root,'dataset/_temp',stem+ext)));
        assert.ok(!fs.existsSync(path.join(root,'dataset',stem+ext)));
      }
    }
    if (!await page.locator('#backup').isVisible()) await page.locator('.advanced-tool-panel > summary').click();
    await page.locator('#backup').check();
    const batch = async (mode, tags, confirm=false) => {
      await page.locator(`.mode-btn[data-mode="${mode}"]`).click();
      if (mode !== 'dedup') await page.locator('#tags').fill(tags);
      const completed = page.waitForResponse(response => response.request().method()==='POST' && new URL(response.url()).pathname==='/');
      await page.locator('#btn-run').click();
      if (confirm) await page.locator('#tags-confirm-ok').click();
      await completed;
      await page.waitForFunction(() => !document.querySelector('#btn-run').disabled);
      assert.ok((await page.locator('#tags-status').innerText()).includes('Inspected'));
    };
    const stagedCaption = path.join(root,'dataset/_temp/image_00.txt');
    await batch('insert','batch marker');
    assert.ok(fs.readFileSync(stagedCaption,'utf8').includes('batch marker'));
    await batch('replace','batch marker->renamed marker');
    assert.ok(fs.readFileSync(stagedCaption,'utf8').includes('renamed marker'));
    await batch('delete','renamed marker',true);
    assert.ok(!fs.readFileSync(stagedCaption,'utf8').includes('renamed marker'));
    fs.appendFileSync(stagedCaption,', repeated tag, repeated tag');
    await batch('dedup','');
    assert.equal(fs.readFileSync(stagedCaption,'utf8').split('repeated tag').length-1,1);
    assert.ok(await page.locator('#tags').evaluate(el => el.hidden));
    const returned=page.waitForResponse(response => response.url().endsWith('/api/tags/return-temp'));
    await page.locator('#workdir-return-temp').click();await page.locator('#tags-confirm-ok').click();await returned;
    await page.locator('#tagsConfirmModal.show').waitFor({state:'hidden'});
    assert.ok(fs.existsSync(path.join(root,'dataset/image_00.png')));
    assert.ok(fs.existsSync(path.join(root,'dataset/image_00.txt')));
    assert.ok(!fs.existsSync(path.join(root,'dataset/_temp/image_00.png')));
    assert.ok((await page.locator('#te-file-result').innerText()).includes('Returned: 2'));
    for (const [width,height] of [[1366,768],[1440,900],[1920,1080],[1093,614],[911,512]]) {
      await page.setViewportSize({width,height});
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`management overflow at ${width}`);
      await page.screenshot({path:path.join(screenshots,`manage-${width}x${height}.png`)});
    }
    await page.setViewportSize({width:1366,height:768});
    for (const factor of [1.25,1.5]) {
      await setBrowserZoom(factor);
      await page.waitForFunction(factor=>Math.abs(innerWidth-1366/factor)<2,factor);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
      await page.screenshot({path:path.join(screenshots,`main-browser-zoom-${factor*100}.png`)});
      assert.ok(await page.evaluate(()=>{const source=document.querySelector('.gallery-section').getBoundingClientRect(),temp=document.querySelector('.workdir-panel').getBoundingClientRect();return innerWidth>1100?temp.left>source.left:temp.top>=source.bottom;}),'browser panels should split or stack at the responsive breakpoint');
    }
    await setBrowserZoom(1);
    for(let i=0;i<160;i++) {
      fs.copyFileSync(path.join(root,'dataset/image_00.png'),path.join(root,'dataset',`large_collection_${i}.png`));
    }
    fs.copyFileSync(path.join(root,'dataset/image_00.png'),path.join(root,'dataset',`${'long_filename_'.repeat(8)}.png`));
    fs.writeFileSync(path.join(root,'dataset',`${'long_filename_'.repeat(8)}.txt`),Array.from({length:250},(_,i)=>`tag ${i}`).join(', '));
    await page.locator('#gallery-limit').fill('500');
    await page.locator('#gallery-refresh').click();
    await page.waitForFunction(()=>document.querySelector('#gallery-count').textContent.includes('165'));
    assert.ok(await page.locator('.gallery-card').count() <= 165);
    await page.screenshot({path:path.join(screenshots,'large-collection.png')});
    await page.locator('#gallery-filename-search').fill('long_filename_');
    assert.equal(await page.locator('.te-gallery-more').innerText(),'+246');
    await page.locator('.gallery-thumb').click();
    await page.waitForFunction(()=>document.querySelector('#gallery-full-caption pre')?.textContent.includes('tag 249'));
    await page.locator('#galleryPreviewModal').evaluate(async modal=>{await Promise.all(modal.getAnimations({subtree:true}).map(animation=>animation.finished));});
    await page.locator('#galleryPreviewModal [data-bs-dismiss="modal"]').click();
    await page.locator('#galleryPreviewModal').waitFor({state:'hidden'});
    await page.locator('#gallery-clear-filters').click();
    await page.setViewportSize({width:390,height:844});
    await page.locator('#dataset-mobile-tag').click();
    await page.locator('#tagging-quiz-modal.show').waitFor();
    await page.waitForFunction(()=>document.body.classList.contains('bb-tag-editor-guided-open'));
    await page.screenshot({path:path.join(screenshots,'guided-mobile-smoke.png')});
    assert.ok(await page.locator('#tagging-mobile-save-image').isVisible());
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    await page.locator('#tagging-quiz-modal .quiz-mobile-utility-actions [data-bs-dismiss="modal"]').click();
    await page.locator('#tagging-quiz-modal').waitFor({state:'hidden'});
    await page.setViewportSize({width:1366,height:768});
    assert.equal(await page.locator('#pipeline-review-banner').count(),1,'pipeline review handoff must remain wired');
    // An invalid path gates the workspace, then an empty valid project has a useful empty state.
    await page.locator('#left-drawer-project-toggle').click();
    await page.locator('#folder_tags').fill(path.join(root,'does-not-exist'));
    await page.locator('#folder_tags').dispatchEvent('change');
    await page.waitForFunction(()=>document.querySelector('#tag-editor-layout').classList.contains('workspace-locked'));
    const emptyRoot=path.join(root,'empty-project');
    fs.mkdirSync(path.join(emptyRoot,'dataset/_temp'),{recursive:true}); fs.mkdirSync(path.join(emptyRoot,'database'));
    fs.writeFileSync(path.join(emptyRoot,'prompt.txt'),'emptytrigger\n');
    await page.locator('#folder_tags').fill(emptyRoot);
    await page.locator('#folder_tags').dispatchEvent('change');
    await page.waitForFunction(()=>document.querySelector('#te-project-state').dataset.ready==='true');
    await page.keyboard.press('Escape');
    await page.waitForFunction(()=>document.querySelector('#gallery-grid').textContent.includes('No images'));
    assert.equal(await page.locator('.gallery-card').count(),0);
    await page.screenshot({path:path.join(screenshots,'empty-project.png')});
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(screenshots,'layout-results.json'),JSON.stringify(layouts,null,2));
    console.log('PASS: project, invalid path, empty project, 165-image collection, long tags and filenames, pair staging/return, all four batch operations, backup, draft/caption separation, keyboard removal, segment ownership, image reuse, zoom, save failure, double save, previous image, typing shortcuts, close guard, focus restoration, wrapped labels, five desktop viewports and mobile smoke.');
    console.log(`Screenshots: ${screenshots}`);
  } finally {
    await browser.close();
    fs.rmSync(root,{recursive:true,force:true});
  }
})().catch(error => {console.error(error);process.exitCode=1;});
