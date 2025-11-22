const form = document.getElementById('process-form');
const urlInput = document.getElementById('video-url');
const thresholdInput = document.getElementById('conf-threshold');
const fpsInput = document.getElementById('fps');
const processBtn = document.getElementById('process-btn');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('error');
const resultsEl = document.getElementById('results');

const titleEl = document.getElementById('title');
const durationEl = document.getElementById('duration');
const thumbEl = document.getElementById('thumb');
const videoIdEl = document.getElementById('video-id');
const summaryListEl = document.getElementById('summary-list');
const videoLinkEl = document.getElementById('video-link');
const jsonLinkEl = document.getElementById('json-link');
const metadataLinkEl = document.getElementById('metadata-link');
const objectsListEl = document.getElementById('objects-list');
const modalEl = document.getElementById('object-modal');
const modalTitleEl = document.getElementById('modal-title');
const modalImageEl = document.getElementById('modal-image');
const modalLoadingEl = document.getElementById('modal-loading');
const modalDetailsEl = document.getElementById('modal-details');
const modalCloseBtn = document.getElementById('modal-close');
const prevFrameBtn = document.getElementById('prev-frame');
const nextFrameBtn = document.getElementById('next-frame');
const zoomInBtn = document.getElementById('zoom-in');
const zoomOutBtn = document.getElementById('zoom-out');
const exportFrameLink = document.getElementById('export-frame');
const modalErrorEl = document.getElementById('modal-error');
const modalFrameListEl = document.getElementById('modal-frame-list');
const frameCounterEl = document.getElementById('frame-counter');
const thumbStripEl = document.getElementById('thumb-strip');

let lastActiveElement = null;
let focusableModalEls = [];

let objectsIndex = {};
let currentClass = null;
let currentFrames = [];
let currentIndex = 0;
let processedFramesBase = '';
let zoomLevel = 1;
const frameIntervalSec = 1;

function secondsToHms(sec) {
  if (!sec && sec !== 0) return '';
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  return [h, m, s]
    .map((v) => String(v).padStart(2, '0'))
    .join(':');
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  errorEl.hidden = true;
  statusEl.textContent = 'Starting processing…';
  processBtn.disabled = true;
  resultsEl.hidden = true;

  const payload = {
    url: urlInput.value.trim(),
    conf_threshold: parseFloat(thresholdInput.value),
    fps: parseInt(fpsInput.value, 10),
  };

  try {
    const res = await fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({ error: 'Unknown error' }));
      if (res.status === 409 && body && body.video_id) {
        statusEl.textContent = 'Loaded existing results.';
        await renderResultsFromVideoId(body.video_id);
        return;
      }
      throw new Error(body.error || `HTTP ${res.status}`);
    }

    statusEl.textContent = 'Processing completed.';
    const data = await res.json();
    const meta = data.metadata || {};
    const sum = data.summary || {};
    const out = data.results || {};

    // Populate metadata
    titleEl.textContent = meta.title || '—';
    durationEl.textContent = secondsToHms(meta.duration);
    videoIdEl.textContent = data.video_id || '—';
    if (meta.thumbnail) {
      thumbEl.src = meta.thumbnail;
      thumbEl.alt = 'YouTube thumbnail';
    }

    
    summaryListEl.innerHTML = '';
    const items = [
      ['Confidence threshold', sum.confidence_threshold],
      ['Total frames', sum.total_frames],
      ['Total detections', sum.total_detections],
    ];
    for (const [label, value] of items) {
      const li = document.createElement('li');
      li.textContent = `${label}: ${value ?? '—'}`;
      summaryListEl.appendChild(li);
    }

    // Output links
    if (out.output_video_url) videoLinkEl.href = out.output_video_url;
    if (out.detection_json_url) jsonLinkEl.href = out.detection_json_url;
    if (out.metadata_url) metadataLinkEl.href = out.metadata_url;

    await renderResultsFromVideoId(data.video_id);
  } catch (err) {
    console.error(err);
    errorEl.textContent = String(err.message || err);
    errorEl.hidden = false;
    statusEl.textContent = '';
  } finally {
    processBtn.disabled = false;
  }
});

async function renderResultsFromVideoId(videoId) {
  processedFramesBase = `/results/${videoId}/processed_frames/`;
  objectsIndex = {};
  const out = {
    output_video_url: `/results/${videoId}/detections_video.mp4`,
    detection_json_url: `/results/${videoId}/detection_results.json`,
    metadata_url: `/results/${videoId}/metadata.txt`,
  };
  if (out.output_video_url) videoLinkEl.href = out.output_video_url;
  if (out.detection_json_url) jsonLinkEl.href = out.detection_json_url;
  if (out.metadata_url) metadataLinkEl.href = out.metadata_url;
  try {
    const dj = await fetch(out.detection_json_url);
    const djData = await dj.json();
    const framesArr = Array.isArray(djData.frames) ? djData.frames : [];
    const byClass = {};
    let totalFrames = 0;
    let totalDetections = 0;
    const map = {};
    for (const f of framesArr) {
      const fname = f.frame;
      const dets = Array.isArray(f.detections) ? f.detections : [];
      map[fname] = dets;
      totalFrames += 1;
      totalDetections += dets.length;
      for (const d of dets) {
        const cls = String(d.class || 'unknown');
        byClass[cls] = (byClass[cls] || 0) + 1;
        if (!objectsIndex[cls]) objectsIndex[cls] = {};
        if (!objectsIndex[cls][fname]) objectsIndex[cls][fname] = [];
        objectsIndex[cls][fname].push({ bbox: d.bbox, conf: d.conf });
      }
    }
    summaryListEl.innerHTML = '';
    const items = [
      ['Confidence threshold', djData.confidence_threshold],
      ['Total frames', totalFrames],
      ['Total detections', totalDetections],
    ];
    for (const [label, value] of items) {
      const li = document.createElement('li');
      li.textContent = `${label}: ${value ?? '—'}`;
      summaryListEl.appendChild(li);
    }
    const entries = Object.entries(objectsIndex).map(([cls, framesMap]) => [cls, Object.keys(framesMap).length]);
    entries.sort((a,b)=>b[1]-a[1]);
    objectsListEl.innerHTML = '';
    for (const [cls, count] of entries) {
      const li = document.createElement('li');
      li.className = 'clickable';
      li.setAttribute('role','button');
      li.setAttribute('tabindex','0');
      li.dataset.cls = cls;
      li.textContent = `${cls} (${count})`;
      li.addEventListener('click', () => openModalForClass(cls));
      li.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); openModalForClass(cls); } });
      objectsListEl.appendChild(li);
    }
    resultsEl.hidden = false;
  } catch (e) {
    console.error(e);
    errorEl.textContent = 'Failed to load existing results.';
    errorEl.hidden = false;
  }
}

function frameNameToIndex(name) {
  const m = String(name).match(/(\d+)/);
  if (!m) return 0;
  return parseInt(m[1], 10) || 0;
}

function formatTimestampFromFrame(name) {
  const idx = frameNameToIndex(name);
  const sec = Math.max(0, (idx - 1) * frameIntervalSec);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  return [h, m, s].map((v)=>String(v).padStart(2,'0')).join(':');
}

function openModalForClass(cls) {
  if (!modalEl || !modalTitleEl) return;
  currentClass = cls;
  const framesMap = objectsIndex[cls] || {};
  currentFrames = Object.keys(framesMap).sort();
  currentIndex = 0;
  zoomLevel = 1;
  modalTitleEl.textContent = cls;
  modalImageEl.style.transform = 'scale(1)';
  lastActiveElement = document.activeElement;
  modalEl.hidden = false;
  modalEl.classList.add('open');
  modalEl.setAttribute('aria-hidden','false');
  document.body.style.overflow = 'hidden';

  // Build frame list
  modalFrameListEl.innerHTML = '';
  if (thumbStripEl) { thumbStripEl.innerHTML = ''; }
  if (!currentFrames.length) {
    modalErrorEl.textContent = 'No frames found for this object.';
    modalErrorEl.hidden = false;
    prevFrameBtn.disabled = true;
    nextFrameBtn.disabled = true;
    exportFrameLink.href = '#';
    exportFrameLink.setAttribute('aria-disabled','true');
  } else {
    modalErrorEl.hidden = true;
    for (let i=0;i<currentFrames.length;i++) {
      const fname = currentFrames[i];
      const li = document.createElement('li');
      const btn = document.createElement('button');
      btn.textContent = `${formatTimestampFromFrame(fname)} — ${fname}`;
      btn.addEventListener('click', () => { currentIndex = i; loadCurrentFrame(); btn.focus(); });
      li.appendChild(btn);
      modalFrameListEl.appendChild(li);

      if (thumbStripEl) {
        const img = document.createElement('img');
        img.src = processedFramesBase + fname;
        img.alt = `Frame ${i+1}`;
        img.setAttribute('role','listitem');
        img.tabIndex = 0;
        img.addEventListener('click', () => { currentIndex = i; loadCurrentFrame(); img.focus(); });
        img.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); currentIndex = i; loadCurrentFrame(); } });
        thumbStripEl.appendChild(img);
      }
    }
  }

  // Focus management
  focusableModalEls = Array.from(modalEl.querySelectorAll('a[href], button, textarea, input, select, [tabindex]:not([tabindex="-1"])')).filter(el => !el.hasAttribute('disabled'));
  const firstEl = focusableModalEls[0] || modalCloseBtn;
  firstEl.focus();
  loadCurrentFrame();
}

function closeModal() {
  modalEl.classList.remove('open');
  modalEl.setAttribute('aria-hidden','true');
  const onTransitionEnd = () => {
    modalEl.hidden = true;
    modalEl.removeEventListener('transitionend', onTransitionEnd);
  };
  modalEl.addEventListener('transitionend', onTransitionEnd);
  document.body.style.overflow = '';
  if (lastActiveElement && typeof lastActiveElement.focus === 'function') {
    lastActiveElement.focus();
  }
  modalLoadingEl.hidden = true;
}

function setControlsState() {
  prevFrameBtn.disabled = currentIndex <= 0;
  nextFrameBtn.disabled = currentIndex >= currentFrames.length - 1;
}

function loadCurrentFrame() {
  setControlsState();
  modalLoadingEl.hidden = false;
  modalErrorEl.hidden = true;
  const fname = currentFrames[currentIndex];
  const url = processedFramesBase + fname;
  const preload = new Image();
  preload.onload = () => { modalImageEl.src = url; modalLoadingEl.hidden = true; };
  preload.onerror = () => { modalLoadingEl.hidden = true; modalErrorEl.textContent = 'Failed to load frame image.'; modalErrorEl.hidden = false; };
  preload.src = url;
  exportFrameLink.href = url;
  modalDetailsEl.innerHTML = '';
  if (frameCounterEl) {
    const total = currentFrames.length;
    frameCounterEl.textContent = `Frame ${currentIndex+1} of ${total}`;
  }
  const tsLi = document.createElement('li');
  tsLi.textContent = `Timestamp: ${formatTimestampFromFrame(fname)}`;
  modalDetailsEl.appendChild(tsLi);
  const dets = (objectsIndex[currentClass]||{})[fname] || [];
  for (const d of dets) {
    const li = document.createElement('li');
    const b = (d.bbox || []).map(v => Math.round(Number(v) || 0));
    const fullText = `bbox: [${b.join(', ')}], conf: ${typeof d.conf==='number'?d.conf.toFixed(3):String(d.conf)}`;
    const truncated = fullText.length > 200 ? fullText.slice(0,200) + '…' : fullText;
    const span = document.createElement('span');
    span.textContent = truncated;
    li.appendChild(span);
    if (fullText.length > 200) {
      const btn = document.createElement('button');
      btn.className = 'btn-secondary';
      btn.style.marginLeft = '0.5rem';
      btn.textContent = 'Show more';
      btn.addEventListener('click', () => {
        const expanded = span.textContent === fullText;
        span.textContent = expanded ? truncated : fullText;
        btn.textContent = expanded ? 'Show more' : 'Show less';
      });
      li.appendChild(btn);
    }
    modalDetailsEl.appendChild(li);
  }
}

if (modalImageEl) {
  modalImageEl.addEventListener('load', () => { modalLoadingEl.hidden = true; });
  modalImageEl.addEventListener('error', () => { modalLoadingEl.hidden = true; modalErrorEl.textContent = 'Failed to load frame image.'; modalErrorEl.hidden = false; });
}
modalCloseBtn.addEventListener('click', () => closeModal());
modalEl.addEventListener('click', (e) => { if (e.target === modalEl) closeModal(); });
prevFrameBtn.addEventListener('click', () => { if (currentIndex>0) { currentIndex--; loadCurrentFrame(); } });
nextFrameBtn.addEventListener('click', () => { if (currentIndex<currentFrames.length-1) { currentIndex++; loadCurrentFrame(); } });
zoomInBtn.addEventListener('click', () => { zoomLevel = Math.min(5, zoomLevel + 0.25); modalImageEl.style.transform = `scale(${zoomLevel})`; });
zoomOutBtn.addEventListener('click', () => { zoomLevel = Math.max(0.5, zoomLevel - 0.25); modalImageEl.style.transform = `scale(${zoomLevel})`; });
document.addEventListener('keydown', (e) => {
  if (modalEl.hidden) return;
  if (e.key === 'Escape') closeModal();
  if (e.key === 'ArrowLeft') prevFrameBtn.click();
  if (e.key === 'ArrowRight') nextFrameBtn.click();
  if (e.key === 'Tab') {
    const focusables = focusableModalEls;
    if (!focusables.length) return;
    const idx = focusables.indexOf(document.activeElement);
    if (e.shiftKey) {
      if (idx <= 0) {
        focusables[focusables.length - 1].focus();
        e.preventDefault();
      }
    } else {
      if (idx === focusables.length - 1) {
        focusables[0].focus();
        e.preventDefault();
      }
    }
  }
});