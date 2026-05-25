(function () {
  'use strict';

  /* Mobile nav */
  const toggle = document.getElementById('navToggle');
  const mobileNav = document.getElementById('mobileNav');
  if (toggle && mobileNav) {
    toggle.addEventListener('click', () => {
      const open = mobileNav.classList.toggle('open');
      document.body.style.overflow = open ? 'hidden' : '';
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    mobileNav.querySelectorAll('a').forEach((a) => {
      a.addEventListener('click', () => {
        mobileNav.classList.remove('open');
        document.body.style.overflow = '';
        toggle.setAttribute('aria-expanded', 'false');
      });
    });
  }

  /* Hero feature pills → mockup panels */
  const pills = document.querySelectorAll('[data-mock]');
  const panels = document.querySelectorAll('.mock-panel');
  if (pills.length && panels.length) {
    pills.forEach((pill) => {
      pill.addEventListener('click', () => {
        const id = pill.getAttribute('data-mock');
        pills.forEach((p) => p.classList.remove('active'));
        pill.classList.add('active');
        panels.forEach((panel) => {
          panel.classList.toggle('active', panel.id === 'mock-' + id);
        });
      });
    });
  }

  /* Screening flow rail */
  const flowBtns = document.querySelectorAll('.flow-rail-btn[data-mastery]');
  const flowStages = document.querySelectorAll('.flow-stage');
  const flowProgress = document.querySelector('.flow-rail-progress');
  if (flowBtns.length && flowStages.length) {
    const stepWidths = { learn: '25%', recognize: '50%', screen: '75%', consult: '100%' };
    flowBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const id = btn.getAttribute('data-mastery');
        flowBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        flowStages.forEach((s) => s.classList.toggle('active', s.id === 'mastery-' + id));
        if (flowProgress && stepWidths[id]) flowProgress.style.width = stepWidths[id];
      });
    });
  }

  /* FAQ accordion */
  document.querySelectorAll('.faq-q').forEach((btn) => {
    btn.addEventListener('click', () => {
      const item = btn.closest('.faq-item');
      const wasOpen = item.classList.contains('open');
      document.querySelectorAll('.faq-item').forEach((i) => i.classList.remove('open'));
      if (!wasOpen) item.classList.add('open');
    });
  });

  /* Upload form (scan section) */
  const uploadForm = document.getElementById('uploadForm');
  if (!uploadForm) return;

  const uploadArea = document.getElementById('uploadArea');
  const fileInput = document.getElementById('file');
  const previewImage = document.getElementById('previewImage');
  const previewContainer = document.getElementById('previewContainer');
  const submitBtn = document.getElementById('submitBtn');
  const loading = document.getElementById('loading');
  const fileName = document.getElementById('fileName');

  function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
      alert('Please select a JPG or PNG image.');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      alert('File must be under 10 MB.');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      if (fileName) fileName.textContent = file.name;
      previewContainer.classList.add('active');
      submitBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  fileInput.addEventListener('change', function () {
    if (this.files[0]) handleFile(this.files[0]);
  });

  uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
  });
  uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
  uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  uploadForm.addEventListener('submit', function (e) {
    e.preventDefault();
    if (!fileInput.files[0]) return;

    submitBtn.disabled = true;
    loading.classList.add('active');

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const backendUrl = (window.BACKEND_URL || '').replace(/\/$/, '');

    fetch(backendUrl + '/predict', { method: 'POST', body: formData })
      .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
      .then(({ ok, data }) => {
        if (!ok || data.error) {
          localStorage.setItem('derma_error', data.error || 'An unexpected error occurred.');
          window.location.href = './error.html';
        } else {
          localStorage.setItem('derma_result', JSON.stringify(data));
          window.location.href = './result.html';
        }
      })
      .catch(() => {
        localStorage.setItem('derma_error', 'Could not reach the analysis server. Please check your connection and try again.');
        window.location.href = './error.html';
      });
  });
})();
