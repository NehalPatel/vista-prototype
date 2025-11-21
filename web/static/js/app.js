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
      const err = await res.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(err.error || `HTTP ${res.status}`);
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

    // Summary list
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
    const byClass = sum.by_class || {};
    for (const [cls, count] of Object.entries(byClass)) {
      const li = document.createElement('li');
      li.textContent = `${cls}: ${count}`;
      summaryListEl.appendChild(li);
    }

    // Output links
    if (out.output_video_url) videoLinkEl.href = out.output_video_url;
    if (out.detection_json_url) jsonLinkEl.href = out.detection_json_url;
    if (out.metadata_url) metadataLinkEl.href = out.metadata_url;

    resultsEl.hidden = false;
  } catch (err) {
    console.error(err);
    errorEl.textContent = String(err.message || err);
    errorEl.hidden = false;
    statusEl.textContent = '';
  } finally {
    processBtn.disabled = false;
  }
});