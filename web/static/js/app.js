const form = document.getElementById('process-form');
const urlInput = document.getElementById('video-url');
const thresholdInput = document.getElementById('conf-threshold');
const fpsInput = document.getElementById('fps');
const scanStartInput = document.getElementById('scan-start');
const scanEndInput = document.getElementById('scan-end');
const forceRescanInput = document.getElementById('force-rescan');
const objectModelSelect = document.getElementById('object-model');
const faceModelSelect = document.getElementById('face-model');
const processBtn = document.getElementById('process-btn');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('error');
const resultsEl = document.getElementById('results');
const resultsPlaceholder = document.getElementById('results-placeholder');
const resultVideoEl = document.getElementById('result-video');
const thumbEl = document.getElementById('thumb');
const titleEl = document.getElementById('title');
const durationEl = document.getElementById('duration');
const videoIdEl = document.getElementById('video-id');
const summaryListEl = document.getElementById('summary-list');
const videoLinkEl = document.getElementById('video-link');
const jsonLinkEl = document.getElementById('json-link');
const metadataLinkEl = document.getElementById('metadata-link');
const objectsListEl = document.getElementById('objects-list');
const facesListEl = document.getElementById('faces-list');
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
const systemInfoListEl = document.getElementById('system-info-list');

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
  return [h, m, s].map((v) => String(v).padStart(2, '0')).join(':');
}

function formatFaceModelLabel(value) {
  const labels = { buffalo_l: 'Buffalo L (InsightFace)', buffalo_s: 'Buffalo S (InsightFace)', buffalo_sc: 'Buffalo SC (InsightFace)' };
  return labels[value] || value || '—';
}

function formatObjectModelLabel(value) {
  const labels = { yolov8n: 'YOLOv8 Nano', yolov8s: 'YOLOv8 Small', yolov8m: 'YOLOv8 Medium', yolov8l: 'YOLOv8 Large', yolov8x: 'YOLOv8 Extra-large' };
  return labels[value] || value || '—';
}

function formatDuration(sec) {
  if (sec == null || typeof sec !== 'number') return '—';
  if (sec < 60) return sec.toFixed(1) + ' s';
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1);
  return m + ' m ' + s + ' s';
}

function buildSummaryItems(sum, objectModelSelect, faceModelSelect, runStats) {
  const items = [
    ['Confidence threshold', sum.confidence_threshold],
    ['Total frames', sum.total_frames],
    ['Total detections', sum.total_detections],
    ['Total face detections', sum.total_face_detections != null ? sum.total_face_detections : '—'],
    ['Object model', sum.object_model ? formatObjectModelLabel(sum.object_model) : (objectModelSelect ? formatObjectModelLabel(objectModelSelect.value) : '—')],
    ['Face model', sum.face_model ? formatFaceModelLabel(sum.face_model) : (faceModelSelect ? formatFaceModelLabel(faceModelSelect.value) : '—')],
  ];
  if (runStats && typeof runStats === 'object') {
    const timeStr = formatDuration(runStats.total_sec);
    const deviceStr = runStats.device === 'cuda' && runStats.gpu_name
      ? 'GPU (' + runStats.gpu_name + ')'
      : (runStats.device === 'cuda' ? 'GPU (CUDA)' : 'CPU');
    items.push(['Time', timeStr]);
    items.push(['Device', deviceStr]);
  }
  return items;
}

/** Reliable YouTube thumbnail URL (same-origin friendly, no referrer issues). */
function youtubeThumbUrl(videoId) {
  if (!videoId) return '';
  return 'https://img.youtube.com/vi/' + encodeURIComponent(videoId) + '/hqdefault.jpg';
}

/**
 * Parse scan start/end input to total seconds.
 * - "6.30" → 6 min 30 sec = 390; "6.3" → same (3 = 30 sec)
 * - "3.00" → 180, "0" or "0.00" → 0
 * - "180" (no dot) → 180 seconds
 */
function parseTimeToSeconds(value, defaultSeconds) {
  const str = String(value || '').trim();
  if (str === '') return defaultSeconds != null ? defaultSeconds : 0;
  if (str.includes('.')) {
    const parts = str.split('.');
    const min = Math.max(0, parseInt(parts[0], 10) || 0);
    const secStr = (parts[1] || '0').padEnd(2, '0').slice(0, 2);
    const sec = Math.min(59, Math.max(0, parseInt(secStr, 10) || 0));
    return min * 60 + sec;
  }
  return Math.max(0, parseInt(str, 10) || 0);
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  errorEl.hidden = true;
  errorEl.textContent = '';
  statusEl.textContent = 'Starting…';
  processBtn.disabled = true;
  resultsEl.hidden = true;
  if (resultsPlaceholder) resultsPlaceholder.hidden = false;
  if (resultVideoEl) {
    resultVideoEl.removeAttribute('src');
    resultVideoEl.load();
  }
  thumbEl.src = '';
  thumbEl.alt = 'Video thumbnail';

  const scanStart = scanStartInput ? parseTimeToSeconds(scanStartInput.value, 0) : 0;
  const scanEnd = scanEndInput ? parseTimeToSeconds(scanEndInput.value, scanStart + 180) : (scanStart + 180);

  const scanModeEl = document.querySelector('input[name="scan-mode"]:checked');
  const payload = {
    url: urlInput.value.trim(),
    conf_threshold: parseFloat(thresholdInput.value),
    fps: parseInt(fpsInput.value, 10),
    scan_start_seconds: scanStart,
    scan_end_seconds: scanEnd,
    force_rescan: forceRescanInput ? forceRescanInput.checked : false,
    object_model: objectModelSelect ? objectModelSelect.value : 'yolov8n',
    face_model: faceModelSelect ? faceModelSelect.value : 'buffalo_l',
    scan_mode: scanModeEl ? scanModeEl.value : 'both',
  };

  try {
    const res = await fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).catch((networkErr) => {
      const msg = (networkErr && networkErr.message) || '';
      if (msg.toLowerCase().includes('fetch') || msg === 'NetworkError when attempting to fetch resource') {
        throw new Error('Could not reach the server. Start the app (e.g. python web/app.py) and try again. Processing can take several minutes.');
      }
      throw networkErr;
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

    statusEl.textContent = 'Done.';
    const data = await res.json();
    const meta = data.metadata || {};
    const sum = data.summary || {};
    const out = data.results || {};
    const videoId = data.video_id || '';

    titleEl.textContent = meta.title || '—';
    durationEl.textContent = secondsToHms(meta.duration);
    videoIdEl.textContent = videoId || '—';

    // Thumbnail: use YouTube's public URL so it always loads (no CORS/referrer issues)
    const thumbUrl = youtubeThumbUrl(videoId);
    if (thumbUrl) {
      thumbEl.src = thumbUrl;
      thumbEl.alt = meta.title ? 'Thumbnail: ' + meta.title : 'Video thumbnail';
      thumbEl.onerror = function () {
        this.onerror = null;
        this.src = 'https://img.youtube.com/vi/' + encodeURIComponent(videoId) + '/default.jpg';
      };
    }

    // Export link for annotated video (inline player removed)
    if (out.output_video_url) {
      if (resultVideoEl) resultVideoEl.src = out.output_video_url;
      videoLinkEl.href = out.output_video_url;
    }
    if (out.detection_json_url) jsonLinkEl.href = out.detection_json_url;
    if (out.metadata_url) metadataLinkEl.href = out.metadata_url;

    // Summary list (include models and run stats when available)
    summaryListEl.innerHTML = '';
    const items = buildSummaryItems(sum, objectModelSelect, faceModelSelect, sum.run_stats);
    for (const [label, value] of items) {
      const li = document.createElement('li');
      li.textContent = label + ': ' + (value ?? '—');
      summaryListEl.appendChild(li);
    }

    // Output links
    if (out.output_video_url) videoLinkEl.href = out.output_video_url;
    if (out.detection_json_url) jsonLinkEl.href = out.detection_json_url;
    if (out.metadata_url) metadataLinkEl.href = out.metadata_url;

    await renderResultsFromVideoId(data.video_id);
  } catch (err) {
    console.error(err);
    let msg = err && (err.message || err);
    if (String(msg).toLowerCase().includes('failed to fetch')) {
      msg = 'Could not reach the server. Run the app from project root (python web/app.py) and try again. Processing may take several minutes.';
    }
    errorEl.textContent = String(msg);
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
      const faces = Array.isArray(f.faces) ? f.faces : [];
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
      if (faces.length) {
        byClass['face'] = (byClass['face'] || 0) + faces.length;
        if (!objectsIndex['face']) objectsIndex['face'] = {};
        objectsIndex['face'][fname] = faces.map((fa) => ({ bbox: fa.bbox, conf: fa.confidence }));
      }
    }
    summaryListEl.innerHTML = '';
    const totalFaceDetections = (djData.frames || []).reduce((acc, fr) => acc + (Array.isArray(fr.faces) ? fr.faces.length : 0), 0);
    const sumForSummary = {
      confidence_threshold: djData.confidence_threshold,
      total_frames: totalFrames,
      total_detections: totalDetections,
      total_face_detections: totalFaceDetections,
      object_model: djData.object_model || (objectModelSelect ? objectModelSelect.value : 'yolov8n'),
      face_model: djData.face_model || (faceModelSelect ? faceModelSelect.value : 'buffalo_l'),
    };
    const items = buildSummaryItems(sumForSummary, objectModelSelect, faceModelSelect, djData.run_stats);
    for (const [label, value] of items) {
      const li = document.createElement('li');
      li.textContent = `${label}: ${value ?? '—'}`;
      summaryListEl.appendChild(li);
    }
    const entries = Object.entries(objectsIndex)
      .filter(([cls]) => cls !== 'face')
      .map(([cls, framesMap]) => [cls, Object.keys(framesMap).length]);
    entries.sort((a,b)=>b[1]-a[1]);
    objectsListEl.innerHTML = '';
    for (const [cls, count] of entries) {
      const li = document.createElement('li');
      li.className = 'clickable';
      li.setAttribute('role','button');
      li.setAttribute('tabindex','0');
      li.dataset.cls = cls;
      li.textContent = `${displayLabelForClass(cls)} (${count})`;
      li.addEventListener('click', () => openModalForClass(cls));
      li.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); openModalForClass(cls); } });
      objectsListEl.appendChild(li);
    }
    // Detected Faces widget
    if (facesListEl) {
      facesListEl.innerHTML = '';
      const faceCount = totalFaceDetections || 0;
      if (faceCount > 0 && objectsIndex['face']) {
        const li = document.createElement('li');
        li.className = 'clickable';
        li.setAttribute('role','button');
        li.setAttribute('tabindex','0');
        li.dataset.cls = 'face';
        li.textContent = `Face (${faceCount})`;
        li.addEventListener('click', () => openModalForClass('face'));
        li.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); openModalForClass('face'); } });
        facesListEl.appendChild(li);
      } else {
        const li = document.createElement('li');
        li.className = 'faces-empty';
        li.textContent = 'No faces detected';
        facesListEl.appendChild(li);
      }
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

function displayLabelForClass(cls) {
  if (typeof cls !== 'string') return cls;
  if (cls.toLowerCase() === 'face') return 'Face';
  return cls.charAt(0).toUpperCase() + cls.slice(1).toLowerCase();
}

function openModalForClass(cls) {
  if (!modalEl || !modalTitleEl) return;
  currentClass = cls;
  const framesMap = objectsIndex[cls] || {};
  currentFrames = Object.keys(framesMap).sort();
  currentIndex = 0;
  zoomLevel = 1;
  modalTitleEl.textContent = displayLabelForClass(cls);
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
        img.dataset.index = String(i);
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
    frameCounterEl.textContent = `Frame ${currentIndex + 1} of ${total}`;
  }
  const tsLi = document.createElement('li');
  tsLi.innerHTML = '<strong>Timestamp</strong>: ' + formatTimestampFromFrame(fname);
  modalDetailsEl.appendChild(tsLi);
  const dets = (objectsIndex[currentClass] || {})[fname] || [];
  for (const d of dets) {
    const li = document.createElement('li');
    const b = (d.bbox || []).map(v => Math.round(Number(v) || 0));
    const confStr = typeof d.conf === 'number' ? d.conf.toFixed(3) : String(d.conf);
    li.innerHTML = '<strong>bbox</strong>: [' + b.join(', ') + ']<br><strong>conf</strong>: ' + confStr;
    modalDetailsEl.appendChild(li);
  }
  if (thumbStripEl) {
    thumbStripEl.querySelectorAll('img').forEach((img, i) => {
      img.classList.toggle('active', i === currentIndex);
    });
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

// Load system info (Python, CPU, GPU) on page load
async function loadSystemInfo() {
  if (!systemInfoListEl) return;
  try {
    const res = await fetch('/api/system-info');
    const data = await res.json();
    systemInfoListEl.innerHTML = '';
    const entries = [
      ['Python', data.python_version || '—'],
      ['CPU', data.cpu || '—'],
      ['Graphics', data.gpu || '—'],
    ];
    for (const [label, value] of entries) {
      const li = document.createElement('li');
      li.className = 'system-info-item';
      li.innerHTML = '<strong>' + label + '</strong>: ' + (value || '—');
      systemInfoListEl.appendChild(li);
    }
  } catch (err) {
    systemInfoListEl.innerHTML = '<li class="system-info-item">Unable to load system info</li>';
  }
}
loadSystemInfo();

// Keep scan end >= scan start; when start changes, if end <= start set end = start + 180
if (scanStartInput && scanEndInput) {
  scanStartInput.addEventListener('input', function () {
    const s = parseTimeToSeconds(scanStartInput.value, 0);
    const e = parseTimeToSeconds(scanEndInput.value, s + 180);
    if (e <= s) scanEndInput.value = String(s + 180);
  });
  scanEndInput.addEventListener('input', function () {
    const s = parseTimeToSeconds(scanStartInput.value, 0);
    const e = parseTimeToSeconds(scanEndInput.value, s + 180);
    if (e <= s) scanEndInput.value = String(s + 180);
  });
}
