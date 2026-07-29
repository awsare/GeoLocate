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

function withWholePct(value) {
  return typeof value === 'number' ? `${Math.round(value)}%` : 'NA';
}

function renderHero(project) {
  const hero = document.querySelector('[data-hero]');
  if (!hero) return;

  const countryBenchmark =
    project.benchmarks && typeof project.benchmarks.countryLevelTestAccuracy === 'number'
      ? project.benchmarks.countryLevelTestAccuracy
      : project.benchmarks && typeof project.benchmarks.continentLevelTestAccuracy === 'number'
      ? project.benchmarks.continentLevelTestAccuracy
      : project.benchmarks && typeof project.benchmarks.continentModelTestAccuracy === 'number'
      ? project.benchmarks.continentModelTestAccuracy
      : null;
  const meanDistanceMiles =
    project.finalPerformance && typeof project.finalPerformance.meanErrorMiles === 'number'
      ? `${Math.round(project.finalPerformance.meanErrorMiles)} mi`
      : 'NA';

  hero.innerHTML = `
    <h1>${project.name}: The Design Journey</h1>
    <p>
      This site documents the model journey from a weak CNN baseline to a tuned
      sector-based ResNet34 system. The goal is to improve class accuracy while
      reducing geographically severe mistakes.
    </p>
    <div class="hero-meta">
      <div class="metric-card">
        <div class="k">Sector-Level Test Accuracy</div>
        <div class="v">${withWholePct(project.headline.testAccuracy)}</div>
      </div>
      <div class="metric-card">
        <div class="k">Country-Level Test Accuracy</div>
        <div class="v">${withWholePct(countryBenchmark)}</div>
      </div>
      <div class="metric-card">
        <div class="k">Mean Distance Error</div>
        <div class="v">${meanDistanceMiles}</div>
      </div>
    </div>
  `;
}

function renderTimeline(project, experiments) {
  const timeline = document.querySelector('[data-timeline]');
  if (!timeline) return;

  const ranked = experiments.slice().sort((a, b) => a.order - b.order);
  const finalExperimentId = project && typeof project.finalExperimentId === 'string'
    ? project.finalExperimentId
    : null;

  const html = ranked
    .map((exp) => {
      const final = finalExperimentId ? exp.id === finalExperimentId : false;
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
                <span class="step-metric-name">Test Accuracy</span>
                <strong class="step-metric-value">${withPct(exp.testAccuracy)}</strong>
              </div>
            </div>
            <div class="step-metric-row">
              <div class="step-metric-track">
                <div class="step-metric-fill dist" style="width: ${distanceWidth}%"></div>
              </div>
              <div class="step-metric-meta">
                <span class="step-metric-name">Distance Score</span>
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

  const finalExperimentId = data.project && typeof data.project.finalExperimentId === 'string'
    ? data.project.finalExperimentId
    : null;
  const finalExp = finalExperimentId
    ? data.experiments.find((x) => x.id === finalExperimentId)
    : null;
  const countryLevelBenchmark =
    data.project && data.project.benchmarks && typeof data.project.benchmarks.countryLevelTestAccuracy === 'number'
      ? data.project.benchmarks.countryLevelTestAccuracy
      : data.project && data.project.benchmarks && typeof data.project.benchmarks.continentLevelTestAccuracy === 'number'
      ? data.project.benchmarks.continentLevelTestAccuracy
      : null;
  if (!finalExp) {
    panel.innerHTML = '<p>No final model is marked in results data.</p>';
    return;
  }

  panel.innerHTML = `
    <div>
      <div class="kpi">
        <div class="kpi-label">Selected Model</div>
        <div class="model-name-value">${finalExp.name}</div>
      </div>
      <div class="kpi-grid">
        <div class="kpi">
          <div class="kpi-label">Sector-Level Test Accuracy</div>
          <div class="kpi-value">${withWholePct(finalExp.testAccuracy)}</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Distance Score</div>
          <div class="kpi-value">${withScore(finalExp.distanceScore)}</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Country-Level Test Accuracy</div>
          <div class="kpi-value">${withWholePct(countryLevelBenchmark)}</div>
        </div>
      </div>
    </div>
  `;
}

function renderFinalPerformance(project) {
  const host = document.querySelector('[data-final-performance]');
  if (!host || !project.finalPerformance) return;

  const perf = project.finalPerformance;
  const sortedByPct = (perf.strongPerClass || [])
    .slice()
    .sort((a, b) => b.pct - a.pct);
  const topSectors = sortedByPct.slice(0, 5);

  const distanceRows = perf.withinMiles
    .map((row) => `<li>${row.miles} mi: ${row.pct.toFixed(2)}%</li>`)
    .join('');

  const topSectorRows = topSectors
    .map((row) => `<li>${row.label}: ${row.pct.toFixed(1)}%</li>`)
    .join('');

  host.innerHTML = `
    <div class="split-grid">
      <article class="process-card reveal">
        <h3>Geographic Metrics</h3>
        <ul class="list-clean">
          <li>Distance score: ${perf.distanceScore.toFixed(4)}</li>
          <li>Mean error: ${Math.round(perf.meanErrorMiles)} miles</li>
          <li>Median error: ${perf.medianErrorMiles.toFixed(1)} miles</li>
        </ul>
      </article>
      <article class="process-card reveal">
        <h3>Within Distance Thresholds</h3>
        <ul class="list-clean">
          ${distanceRows}
        </ul>
      </article>
      <article class="process-card reveal">
        <h3>Strongest Sectors</h3>
        <ul class="list-clean">
          ${topSectorRows}
        </ul>
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
      <h3>6. Validation-based Selection</h3>
      <p>Final evaluation uses the checkpoint with the strongest validation score to reduce overfitting and preserve generalization.</p>
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
  renderTimeline(data.project, data.experiments);
  renderFinalModel(data);
  renderFinalPerformance(data.project);
  enableReveal();
}

boot();
