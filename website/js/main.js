async function loadResults() {
  const response = await fetch('./assets/results.json?v=20260726', { cache: 'no-store' });
  if (!response.ok) {
    throw new Error('Unable to load results.json');
  }
  return response.json();
}

function markActiveNav() {
  const links = document.querySelectorAll('a[href^="#"]');
  links.forEach((link) => {
    link.addEventListener('click', () => {
      links.forEach((item) => item.classList.remove('active'));
      link.classList.add('active');
    });
  });
}

function withPct(value) {
  return typeof value === 'number' ? `${value}%` : 'NA';
}

function withScore(value) {
  return typeof value === 'number' ? value.toFixed(2) : 'NA';
}

function renderHero(project) {
  const hero = document.querySelector('[data-hero]');
  if (!hero) return;

  const continentBenchmark =
    project.benchmarks && typeof project.benchmarks.continentModelTestAccuracy === 'number'
      ? `${project.benchmarks.continentModelTestAccuracy}%`
      : 'NA';

  hero.innerHTML = `
    <h1>${project.name}: Design Journey To Better Geographic Predictions</h1>
    <p>
      This site documents the model evolution from a weak CNN baseline to a tuned
      sector-based ResNet34 system. The focus is simple: improve class accuracy while
      reducing geographically severe mistakes.
    </p>
    <div class="hero-meta">
      <div class="metric-card">
        <div class="k">Sector Model Accuracy</div>
        <div class="v">${project.headline.testAccuracy}%</div>
      </div>
      <div class="metric-card">
        <div class="k">Continent Model Accuracy</div>
        <div class="v">${continentBenchmark}</div>
      </div>
    </div>
  `;
}

function renderTimeline(experiments) {
  const timeline = document.querySelector('[data-timeline]');
  if (!timeline) return;

  const ranked = experiments.slice().sort((a, b) => a.order - b.order);

  const html = ranked
    .map((exp) => {
      const final = exp.status === 'final';
      const accuracyWidth = Math.max(0, Math.min(100, exp.testAccuracy));
      const hasDistance = typeof exp.distanceScore === 'number';
      const distanceWidth = hasDistance
        ? Math.max(0, Math.min(100, exp.distanceScore * 100))
        : 0;
      return `
        <article class="step reveal${final ? ' final' : ''}">
          <p class="step-title">${exp.order}. ${exp.name}</p>
          <p>${exp.summary}</p>
          <div class="step-metrics">
            <div class="step-metric-row">
              <div class="step-metric-track">
                <div class="step-metric-fill" style="width: ${accuracyWidth}%"></div>
              </div>
              <div class="step-metric-meta">
                <span class="step-metric-name">Accuracy</span>
                <strong class="step-metric-value">${withPct(exp.testAccuracy)}</strong>
              </div>
            </div>
            <div class="step-metric-row">
              <div class="step-metric-track">
                <div class="step-metric-fill dist" style="width: ${distanceWidth}%"></div>
              </div>
              <div class="step-metric-meta">
                <span class="step-metric-name">Distance</span>
                <strong class="step-metric-value">${hasDistance ? withScore(exp.distanceScore) : 'NA'}</strong>
              </div>
            </div>
          </div>
        </article>
      `;
    })
    .join('');

  timeline.innerHTML = html;
}

function renderFinalModel(data) {
  const panel = document.querySelector('[data-final-model]');
  if (!panel) return;

  const finalExp = data.experiments.find((x) => x.status === 'final');
  if (!finalExp) {
    panel.innerHTML = '<p>No final model is marked in results data.</p>';
    return;
  }

  panel.innerHTML = `
    <div>
      <h3>${finalExp.name}</h3>
      <div class="kpi-grid">
        <div class="kpi">
          <div class="kpi-label">Test Accuracy</div>
          <div class="kpi-value">${withPct(finalExp.testAccuracy)}</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Distance Score</div>
          <div class="kpi-value">${withScore(finalExp.distanceScore)}</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Backbone</div>
          <div class="kpi-value">${finalExp.backbone}</div>
        </div>
      </div>
    </div>
    <div>
      <div class="callout">
        <strong>Why this model was selected</strong>
        <p>
          It preserves the strongest observed accuracy and improves geographic quality,
          which aligns with the project goal of reducing severe location errors.
        </p>
      </div>
    </div>
  `;
}

function renderFinalPerformance(project) {
  const host = document.querySelector('[data-final-performance]');
  if (!host || !project.finalPerformance) return;

  const perf = project.finalPerformance;

  const distanceRows = perf.withinMiles
    .map((row) => `<li><span class="metric-inline">${row.miles} mi</span> ${row.pct.toFixed(2)}%</li>`)
    .join('');

  const classRows = perf.strongPerClass
    .map((row) => `<li>${row.label}: ${row.pct.toFixed(1)}%</li>`)
    .join('');

  host.innerHTML = `
    <div class="split-grid">
      <article class="process-card reveal">
        <h3>Geographic Error Profile</h3>
        <ul class="list-clean">
          <li>Mean error: ${perf.meanErrorMiles.toFixed(1)} miles</li>
          <li>Median error: ${perf.medianErrorMiles.toFixed(1)} miles</li>
          <li>Distance score exp(-d/tau), tau=932: ${perf.distanceScore.toFixed(4)}</li>
        </ul>
      </article>
      <article class="process-card reveal">
        <h3>Within Distance Thresholds</h3>
        <ul class="list-clean">
          ${distanceRows}
        </ul>
      </article>
      <article class="process-card reveal">
        <h3>Strong Per-Sector Accuracies</h3>
        <ul class="list-clean">
          ${classRows}
        </ul>
        <p>${perf.note}</p>
      </article>
    </div>
  `;
}

function renderProcessCards() {
  const host = document.querySelector('[data-process-cards]');
  if (!host) return;

  host.innerHTML = `
    <article class="process-card reveal">
      <h3>1. Label Design</h3>
      <p>Countries were grouped into sectors to reduce sparsity and keep minority regions trainable.</p>
    </article>
    <article class="process-card reveal">
      <h3>2. Backbone Upgrade</h3>
      <p>Transfer learning via ResNet backbones replaced a weak custom CNN baseline.</p>
    </article>
    <article class="process-card reveal">
      <h3>3. Two-Stage Training</h3>
      <p>Head warmup followed by full fine-tuning stabilized optimization and improved score consistency.</p>
    </article>
    <article class="process-card reveal">
      <h3>4. Imbalance Handling</h3>
      <p>Class weighting and weighted sampling were tested to improve low-frequency sector recall.</p>
    </article>
    <article class="process-card reveal">
      <h3>5. Distance-Aware Objective</h3>
      <p>A distance penalty was added to discourage geographically severe classification mistakes.</p>
    </article>
    <article class="process-card reveal">
      <h3>6. Selection Rule</h3>
      <p>The final pick balanced test accuracy, distance score, and model efficiency.</p>
    </article>
  `;
}

function enableReveal() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );

  document.querySelectorAll('.reveal').forEach((el) => observer.observe(el));
}

async function boot() {
  markActiveNav();
  renderProcessCards();

  let data = null;
  try {
    data = await loadResults();
  } catch (error) {
    const errTarget = document.querySelector('[data-errors]');
    if (errTarget) errTarget.textContent = error.message;
    return;
  }

  renderHero(data.project);
  renderTimeline(data.experiments);
  renderFinalModel(data);
  renderFinalPerformance(data.project);
  enableReveal();
}

boot();
