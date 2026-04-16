/* ═══════════════════════════════════════════
   App.js — Cascade selectors + model tabs
════════════════════════════════════════════ */

(function () {
  "use strict";

  // ── 1. Parse embedded cascade data ────────────────────────────
  const cascadeEl = document.getElementById("cascadeData");
  const cascade = cascadeEl ? JSON.parse(cascadeEl.textContent) : {};
  const brandSeriesMap    = cascade.brand_series_map    || {};
  const cpuBrandFamilyMap = cascade.cpu_brand_family_map || {};
  const cpuFamilySuffixMap= cascade.cpu_family_suffix_map|| {};
  const cpuBrandGenerationMap = cascade.cpu_brand_generation_map || {};

  // DOM refs
  const brandSel    = document.getElementById("brandSelect");
  const seriesSel   = document.getElementById("seriesSelect");
  const cpuBrandSel = document.getElementById("cpuBrandSelect");
  const cpuFamSel   = document.getElementById("cpuFamilySelect");
  const cpuGenSel   = document.getElementById("cpuGenSelect");
  const cpuSufSel   = document.getElementById("cpuSuffixSelect");
  const gpuModelSel = document.getElementById("gpuModelSelect");
  const gpuSufSel   = document.getElementById("gpuSuffixSelect");
  const gpuSufWrap  = document.getElementById("gpuSuffixWrap");

  // ── Helper: repopulate a <select> ─────────────────────────────
  function repopulate(sel, values, currentValue) {
    if (!sel || !values || values.length === 0) return;
    sel.innerHTML = "";
    values.forEach(function (v) {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      if (v === currentValue) opt.selected = true;
      sel.appendChild(opt);
    });
    // If current value wasn't found, pick first
    if (!sel.value && values.length > 0) sel.value = values[0];
  }

  // ── 2. Brand → Series cascade ──────────────────────────────────
  function updateSeries(keepCurrent) {
    if (!brandSel || !seriesSel) return;
    const brand = brandSel.value;
    const current = keepCurrent ? seriesSel.value : null;
    const options = brandSeriesMap[brand] || [];
    if (options.length > 0) {
      repopulate(seriesSel, options, current);
    }
  }

  // ── 3. CPU Brand → CPU Family cascade ──────────────────────────
  function updateCpuFamily(keepCurrent) {
    if (!cpuBrandSel || !cpuFamSel) return;
    const brand = cpuBrandSel.value;
    const current = keepCurrent ? cpuFamSel.value : null;
    const options = cpuBrandFamilyMap[brand] || [];
    if (options.length > 0) {
      repopulate(cpuFamSel, options, current);
    }
    updateCpuGeneration(keepCurrent);
    updateCpuSuffix(keepCurrent);
  }

  // ── 3.b CPU Brand → CPU Generation cascade ──────────────────────
  function updateCpuGeneration(keepCurrent) {
    if (!cpuBrandSel || !cpuGenSel) return;
    const brand = cpuBrandSel.value;
    const current = keepCurrent ? parseInt(cpuGenSel.value) || null : null;
    let options = cpuBrandGenerationMap[brand] || [];
    
    if (options.length > 0) {
      // Ensure numeric representation
      options = options.map(function(num) { return String(num); });
      repopulate(cpuGenSel, options, current ? String(current) : null);
    }
  }

  // ── 4. CPU Family → CPU Suffix cascade ─────────────────────────
  function updateCpuSuffix(keepCurrent) {
    if (!cpuBrandSel || !cpuFamSel || !cpuSufSel) return;
    const brand = cpuBrandSel.value;
    const fam   = cpuFamSel.value;
    const key   = brand + "|" + fam;
    const current = keepCurrent ? cpuSufSel.value : null;
    const options = cpuFamilySuffixMap[key] || [];
    if (options.length > 0) {
      repopulate(cpuSufSel, options, current);
    }
  }

  // ── 5. GPU Model → hide suffix for integrated ──────────────────
  function updateGpuSuffix() {
    if (!gpuModelSel || !gpuSufSel) return;
    const val = gpuModelSel.value.toLowerCase();
    const isIntegrated = val.includes("integrated") || val.includes("no dedicated");
    if (gpuSufWrap) gpuSufWrap.style.opacity = isIntegrated ? "0.4" : "1";
    if (isIntegrated) gpuSufSel.value = "None/Standard";
  }

  // ── 6. Wire up change listeners ────────────────────────────────
  if (brandSel)    brandSel.addEventListener("change", function () { updateSeries(false); });
  if (cpuBrandSel) cpuBrandSel.addEventListener("change", function () { updateCpuFamily(false); });
  if (cpuFamSel)   cpuFamSel.addEventListener("change", function () { updateCpuSuffix(true); });
  if (gpuModelSel) gpuModelSel.addEventListener("change", updateGpuSuffix);

  // Run once on load with keepCurrent = true (defaults are already set by server)
  updateSeries(true);
  updateCpuFamily(true);
  updateCpuGeneration(true);
  updateCpuSuffix(true);
  updateGpuSuffix();

  // ── 7. Model tab switcher ───────────────────────────────────────
  const tabs = document.querySelectorAll(".model-tab");
  const form = document.getElementById("predictForm");

  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      const key = tab.dataset.key;
      if (!key) return;

      // Update active tab UI immediately
      tabs.forEach(function (t) { t.classList.remove("active"); });
      tab.classList.add("active");

      // Update the form action URL
      if (form) {
        form.action = "/predict?model_key=" + encodeURIComponent(key);
      }

      // Update the GET link (so refreshing the page keeps the model)
      const url = new URL(window.location.href);
      url.searchParams.set("model_key", key);
      window.history.replaceState({}, "", url.toString());
    });
  });

  // ── 8. Loading state on submit ──────────────────────────────────
  const submitBtn = document.getElementById("submitBtn");
  if (form && submitBtn) {
    form.addEventListener("submit", function () {
      const btnText    = submitBtn.querySelector(".btn-text");
      const btnLoading = submitBtn.querySelector(".btn-loading");
      if (btnText)    btnText.hidden = true;
      if (btnLoading) btnLoading.hidden = false;
      submitBtn.disabled = true;
    });
  }

  // ── 9. Scroll to result on load ────────────────────────────────
  const resultBanner = document.getElementById("resultBanner");
  if (resultBanner) {
    setTimeout(function () {
      resultBanner.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }, 150);
  }

})();
