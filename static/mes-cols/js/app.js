(() => {
  "use strict";

  const ASC_MAX = 20;

  const state = {
    all: [],
    filtered: [],
    dept: "", // selected département/country value ("" = all)
    map: null,
    markers: new Map(), // id -> leaflet marker
    markerLayer: null,
    view: "map",
    viewBeforeImport: "map",
    sort: { key: "altitude", dir: "desc" }, // list view sort
  };

  let pendingImport = null; // parsed records waiting for confirmation
  let searchFilterTimer = null;

  const els = {
    stats: document.getElementById("stats"),
    search: document.getElementById("search"),
    searchClear: document.getElementById("search-clear"),
    searchList: document.getElementById("search-list"),
    deptInput: document.getElementById("dept-input"),
    deptClear: document.getElementById("dept-clear"),
    deptList: document.getElementById("dept-list"),
    fPass: document.getElementById("f-pass"),
    fMontee: document.getElementById("f-montee"),
    fHigh: document.getElementById("f-high"),
    fMtb: document.getElementById("f-mtb"),
    fUnclimbed: document.getElementById("f-unclimbed"),
    altMin: document.getElementById("alt-min"),
    altValue: document.getElementById("alt-value"),
    ascMinus: document.getElementById("asc-minus"),
    ascPlus: document.getElementById("asc-plus"),
    ascValue: document.getElementById("asc-value"),
    resetBtn: document.getElementById("reset-filters"),
    emptyState: document.getElementById("empty-state"),
    main: document.getElementById("main"),
    filters: document.getElementById("filters"),
    filtersToggle: document.getElementById("filters-toggle"),
    mapView: document.getElementById("map-view"),
    listView: document.getElementById("list-view"),
    tableHead: document.getElementById("cols-table-head"),
    tableBody: document.getElementById("cols-table-body"),
    drawer: document.getElementById("drawer"),
    drawerContent: document.getElementById("drawer-content"),
    drawerClose: document.getElementById("drawer-close"),
    btnViewMap: document.getElementById("btn-view-map"),
    btnViewList: document.getElementById("btn-view-list"),
    btnDownload: document.getElementById("btn-download"),
    exportOptions: document.getElementById("export-options"),
    btnImport: document.getElementById("btn-import"),
    importView: document.getElementById("import-view"),
    dropzone: document.getElementById("dropzone"),
    fileInput: document.getElementById("file-input"),
    importResult: document.getElementById("import-result"),
    importSummary: document.getElementById("import-summary"),
    importErrors: document.getElementById("import-errors"),
    importWarnings: document.getElementById("import-warnings"),
    btnImportConfirm: document.getElementById("btn-import-confirm"),
    btnImportCancel: document.getElementById("btn-import-cancel"),
    dataSource: document.getElementById("data-source"),
  };

  // ---------------------------------------------------------------
  // Data loading
  // ---------------------------------------------------------------
  async function loadData() {
    // Prefer the geocoded file; fall back to the raw parsed file so the
    // app still runs (list view + non-geocoded filter) before the
    // geocoding script has been run.
    let data = null;
    let usedFallback = false;
    try {
      const res = await fetch("data/cols_geocoded.json");
      if (res.ok) data = await res.json();
    } catch (e) { /* ignore, try fallback */ }

    if (!data) {
      usedFallback = true;
      try {
        const res = await fetch("data/cols.json");
        if (res.ok) data = await res.json();
      } catch (e) {
        console.error("Could not load data/cols.json either.", e);
        data = [];
      }
    }

    const hasMissingCoords = data.some(r => r.lat == null || r.lon == null);
    els.emptyState.hidden = !(usedFallback || hasMissingCoords) || data.length === 0;

    return data;
  }

  // ---------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------
  async function init() {
    state.all = await loadData();
    populateDeptList(state.all);
    initMap();
    wireControls();
    wireImport();
    applyFilters();
  }

  function populateDeptList(rows) {
    const groups = new Map(); // key -> label
    rows.forEach(r => {
      if (r.department_code) {
        groups.set(r.department_code, `${r.department_code} \u2013 ${r.department_name}`);
      } else {
        groups.set(r.country, r.country);
      }
    });
    const entries = Array.from(groups.entries()).sort((a, b) => a[1].localeCompare(b[1]));
    els.deptList.innerHTML = "";
    const all = document.createElement("li");
    all.dataset.value = "";
    all.textContent = "Tous";
    els.deptList.appendChild(all);
    for (const [value, label] of entries) {
      const li = document.createElement("li");
      li.dataset.value = value;
      li.textContent = label;
      els.deptList.appendChild(li);
    }
    markDeptSelected("");
  }

  function markDeptSelected(value) {
    for (const li of els.deptList.children) {
      li.setAttribute("aria-selected", (li.dataset.value || "") === value ? "true" : "false");
    }
  }

  function openDeptList() {
    els.deptList.hidden = false;
    els.deptInput.setAttribute("aria-expanded", "true");
  }

  function closeDeptList() {
    els.deptList.hidden = true;
    els.deptInput.setAttribute("aria-expanded", "false");
  }

  function filterDeptList() {
    const q = els.deptInput.value.trim();
    const matches = Array.from(els.deptList.children)
      .filter(li => !li.classList.contains("dept-empty"))
      .map(li => ({
        li,
        score: q ? fuzzyScore(li.textContent, q) : (li.dataset.value ? 1 : 0),
      }))
      .filter(item => item.score != null)
      .sort((a, b) => a.score - b.score || a.li.textContent.localeCompare(b.li.textContent, "fr"));

    for (const li of els.deptList.children) {
      if (!li.classList.contains("dept-empty")) li.hidden = true;
    }
    for (const item of matches) item.li.hidden = false;

    const visible = matches.length;
    let empty = els.deptList.querySelector(".dept-empty");
    if (visible === 0) {
      if (!empty) {
        empty = document.createElement("li");
        empty.className = "dept-empty";
        empty.textContent = "Aucun r\u00e9sultat";
        els.deptList.appendChild(empty);
      }
      empty.hidden = false;
    } else if (empty) {
      empty.hidden = true;
    }
  }

  function deptVisibleItems() {
    return Array.from(els.deptList.children).filter(li => !li.hidden && !li.classList.contains("dept-empty"));
  }

  function setDeptActive(li) {
    for (const item of els.deptList.children) item.classList.remove("active");
    if (li) {
      li.classList.add("active");
      li.scrollIntoView({ block: "nearest" });
    }
  }

  function selectDept(li) {
    if (!li || li.classList.contains("dept-empty")) return;
    state.dept = li.dataset.value || "";
    els.deptInput.value = state.dept ? li.textContent : "";
    updateDeptClear();
    markDeptSelected(state.dept);
    closeDeptList();
    applyFilters();
  }

  function initMap() {
    state.map = L.map("map", { zoomControl: true, attributionControl: false }).setView([45.5, 5.9], 7);

    // OpenFreeMap: free, no API key, no account, no request limits.
    // "dark" is an unofficial-but-published style (not on their quick-start
    // page yet) built for exactly this use case. If it ever misbehaves,
    // swap the URL below for '.../styles/liberty' or '.../styles/positron'
    // (both fully supported) - everything else in this file stays the same.
    L.maplibreGL({
      style: "https://tiles.openfreemap.org/styles/liberty",
    }).addTo(state.map);

    L.control.attribution({ prefix: false })
      .addAttribution('&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://openfreemap.org">OpenFreeMap</a>')
      .addTo(state.map);

    state.markerLayer = L.layerGroup().addTo(state.map);
  }

  function wireControls() {
    setFiltersOpen(!window.matchMedia("(max-width: 760px)").matches);
    els.main.classList.add("filters-ready");
    els.filtersToggle.addEventListener("click", () => {
      setFiltersOpen(els.filters.classList.contains("collapsed"));
    });
    els.search.addEventListener("input", () => {
      updateSearchClear();
      showSearchSuggestions();
      setSearchActive(searchVisibleItems()[0]);
      scheduleSearchFilter();
    });
    els.search.addEventListener("focus", () => {
      showSearchSuggestions();
      setSearchActive(searchVisibleItems()[0]);
    });
    els.search.addEventListener("keydown", handleSearchKeydown);
    els.searchClear.addEventListener("click", clearSearch);
    els.searchList.addEventListener("click", (e) => {
      const li = e.target instanceof Element ? e.target.closest("li[data-name]") : null;
      if (li) selectSearchSuggestion(li.dataset.name || "");
    });
    els.deptInput.addEventListener("focus", () => {
      updateDeptClear();
      openDeptList();
      filterDeptList();
      setDeptActive(deptVisibleItems()[0]);
    });
    els.deptInput.addEventListener("input", () => {
      updateDeptClear();
      openDeptList();
      filterDeptList();
      setDeptActive(deptVisibleItems()[0]);
    });
    els.deptClear.addEventListener("click", clearDept);
    els.deptInput.addEventListener("keydown", (e) => {
      const open = !els.deptList.hidden;
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        if (!open) openDeptList();
        const items = deptVisibleItems();
        if (!items.length) return;
        const idx = items.indexOf(els.deptList.querySelector("li.active"));
        const next = e.key === "ArrowDown"
          ? items[(idx + 1) % items.length]
          : items[(idx - 1 + items.length) % items.length];
        setDeptActive(next);
      } else if (e.key === "Enter") {
        if (!open) return;
        e.preventDefault();
        selectDept(els.deptList.querySelector("li.active") || deptVisibleItems()[0]);
      } else if (e.key === "Escape") {
        if (open) closeDeptList();
      }
    });
    els.deptList.addEventListener("click", (e) => {
      const li = e.target instanceof Element ? e.target.closest("li") : null;
      if (li) selectDept(li);
    });
    document.addEventListener("mousedown", (e) => {
      if (e.target instanceof Element && !e.target.closest(".combobox")) closeDeptList();
      if (e.target instanceof Element && !e.target.closest(".search-combobox")) closeSearchSuggestions();
    });
    els.fPass.addEventListener("change", applyFilters);
    els.fMontee.addEventListener("change", applyFilters);
    els.fHigh.addEventListener("change", applyFilters);
    els.fMtb.addEventListener("change", applyFilters);
    els.fUnclimbed.addEventListener("change", applyFilters);
    els.altMin.addEventListener("input", () => {
      els.altValue.textContent = `${els.altMin.value} m`;
      applyFilters();
    });
    els.ascMinus.addEventListener("click", () => setAscMin(getAscMin() - 1));
    els.ascPlus.addEventListener("click", () => setAscMin(getAscMin() + 1));
    els.ascValue.addEventListener("input", () => {
      syncAscButtons();
      applyFilters();
    });
    els.ascValue.addEventListener("change", () => setAscMin(getAscMin()));
    setAscMin(0);

    els.resetBtn.addEventListener("click", resetFilters);

    els.drawerClose.addEventListener("click", closeDrawer);

    els.btnViewMap.addEventListener("click", () => switchView("map"));
    els.btnViewList.addEventListener("click", () => switchView("list"));
    els.btnDownload.addEventListener("click", toggleExportOptions);
    for (const option of els.exportOptions.querySelectorAll("[data-export-format]")) {
      option.addEventListener("click", () => {
        downloadData(option.dataset.exportFormat);
        closeExportOptions();
      });
    }
    document.addEventListener("mousedown", (e) => {
      if (e.target instanceof Element && !e.target.closest(".export-menu")) closeExportOptions();
    });

    for (const th of els.tableHead.querySelectorAll("th[data-sort]")) {
      th.addEventListener("click", () => setSort(th.dataset.sort));
    }
    updateSortIndicators();
  }

  function setFiltersOpen(open) {
    els.filters.classList.toggle("collapsed", !open);
    els.filtersToggle.setAttribute("aria-expanded", open);
    els.filtersToggle.setAttribute("aria-label", open ? "Masquer les filtres" : "Afficher les filtres");
    els.filtersToggle.querySelector(".filters-toggle-icon").textContent = open ? "\u2039" : "\u203a";
  }

  function scheduleSearchFilter() {
    clearTimeout(searchFilterTimer);
    searchFilterTimer = setTimeout(() => {
      searchFilterTimer = null;
      applyFilters();
    }, 300);
  }

  function updateSearchClear() {
    els.searchClear.hidden = !els.search.value;
  }

  function updateDeptClear() {
    els.deptClear.hidden = !els.deptInput.value;
  }

  function clearSearch() {
    clearTimeout(searchFilterTimer);
    searchFilterTimer = null;
    els.search.value = "";
    updateSearchClear();
    showSearchSuggestions();
    setSearchActive(searchVisibleItems()[0]);
    applyFilters();
    els.search.focus();
  }

  function clearDept() {
    state.dept = "";
    els.deptInput.value = "";
    updateDeptClear();
    markDeptSelected("");
    closeDeptList();
    applyFilters();
    els.deptInput.focus();
  }

  function normalizeSearch(value) {
    return value
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLowerCase();
  }

  function fuzzyScore(name, query) {
    const candidate = normalizeSearch(name);
    const needle = normalizeSearch(query);
    if (!needle) return 0;
    if (candidate === needle) return 0;
    if (candidate.startsWith(needle)) return 1;
    if (candidate.includes(needle)) return 2 + candidate.indexOf(needle) / 1000;

    let cursor = 0;
    let gaps = 0;
    for (const character of needle) {
      const index = candidate.indexOf(character, cursor);
      if (index === -1) return null;
      gaps += index - cursor;
      cursor = index + 1;
    }
    return 3 + gaps / 1000 + (candidate.length - needle.length) / 10000;
  }

  function searchSuggestions() {
    const query = els.search.value.trim();
    const matches = state.all
      .map(rec => ({ rec, score: fuzzyScore(rec.name, query) }))
      .filter(item => item.score != null)
      .sort((a, b) => a.score - b.score || a.rec.name.localeCompare(b.rec.name, "fr"))
      .map(item => item.rec);
    return query ? matches : state.all.slice().sort((a, b) => a.name.localeCompare(b.name, "fr"));
  }

  function showSearchSuggestions() {
    const query = els.search.value.trim();
    const suggestions = searchSuggestions();
    els.searchList.innerHTML = "";
    if (query && suggestions.length === 0) {
      const empty = document.createElement("li");
      empty.className = "dept-empty";
      empty.setAttribute("role", "option");
      empty.textContent = "Aucun résultat";
      els.searchList.appendChild(empty);
    }
    for (const rec of suggestions) {
      const li = document.createElement("li");
      li.dataset.name = rec.name;
      li.setAttribute("role", "option");
      li.textContent = rec.name;
      els.searchList.appendChild(li);
    }
    els.searchList.hidden = false;
    els.search.setAttribute("aria-expanded", "true");
  }

  function searchVisibleItems() {
    return Array.from(els.searchList.children).filter(li => !li.hidden);
  }

  function setSearchActive(li) {
    for (const item of els.searchList.children) item.classList.remove("active");
    if (li) li.classList.add("active");
  }

  function closeSearchSuggestions() {
    els.searchList.hidden = true;
    els.search.setAttribute("aria-expanded", "false");
  }

  function selectSearchSuggestion(name) {
    clearTimeout(searchFilterTimer);
    searchFilterTimer = null;
    els.search.value = name;
    updateSearchClear();
    closeSearchSuggestions();
    applyFilters();
  }

  function handleSearchKeydown(e) {
    if (e.key === "Escape") {
      closeSearchSuggestions();
      return;
    }
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp" && e.key !== "Enter") return;
    const items = Array.from(els.searchList.querySelectorAll("li[data-name]"));
    if (!items.length || els.searchList.hidden) return;
    e.preventDefault();
    const active = els.searchList.querySelector("li.active");
    const index = items.indexOf(active);
    if (e.key === "Enter") {
      selectSearchSuggestion((active || items[0]).dataset.name);
      return;
    }
    const next = e.key === "ArrowDown"
      ? items[(index + 1) % items.length]
      : items[(index - 1 + items.length) % items.length];
    for (const item of items) item.classList.remove("active");
    next.classList.add("active");
    next.scrollIntoView({ block: "nearest" });
  }

  function switchView(view) {
    state.view = view;
    els.mapView.hidden = view !== "map";
    els.listView.hidden = view !== "list";
    els.importView.hidden = view !== "import";
    els.btnViewMap.classList.toggle("active", view === "map");
    els.btnViewMap.setAttribute("aria-selected", view === "map");
    els.btnViewList.classList.toggle("active", view === "list");
    els.btnViewList.setAttribute("aria-selected", view === "list");
    if (view === "map") setTimeout(() => state.map.invalidateSize(), 50);
  }

  // ---------------------------------------------------------------
  // CSV / JSON export & import
  // ---------------------------------------------------------------
  function resetFilters() {
    clearTimeout(searchFilterTimer);
    searchFilterTimer = null;
    els.search.value = "";
    updateSearchClear();
    closeSearchSuggestions();
    state.dept = "";
    els.deptInput.value = "";
    updateDeptClear();
    markDeptSelected("");
    closeDeptList();
    els.fPass.checked = true;
    els.fMontee.checked = true;
    els.fHigh.checked = false;
    els.fMtb.checked = false;
    els.fUnclimbed.checked = false;
    els.altMin.value = 0;
    els.altValue.textContent = "0 m";
    setAscMin(0);
  }

  function toggleExportOptions() {
    const open = els.exportOptions.hidden;
    els.exportOptions.hidden = !open;
    els.btnDownload.setAttribute("aria-expanded", open);
  }

  function closeExportOptions() {
    els.exportOptions.hidden = true;
    els.btnDownload.setAttribute("aria-expanded", "false");
  }

  function downloadData(format) {
    let url;
    try {
      const isCsv = format === "csv";
      // BOM so Excel/LibreOffice detect the UTF-8 encoding.
      const contents = isCsv
        ? "\uFEFF" + ColCsv.recordsToCsv(state.all)
        : JSON.stringify(state.all, null, 2) + "\n";
      const blob = new Blob([contents], {
        type: isCsv ? "text/csv;charset=utf-8" : "application/json;charset=utf-8",
      });
      url = URL.createObjectURL(blob);
    } catch (e) {
      alert(`Export ${format.toUpperCase()} indisponible : ${e.message}`);
      return;
    }
    const a = document.createElement("a");
    a.href = url;
    a.download = `mes-cols.${format}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // Browser-side equivalent of scripts/parse_csv.py detect_encoding():
  // try UTF-8 (fatal), fall back to Latin-1 (Excel's plain "CSV" flavor).
  function decodeBuffer(buf) {
    const b = new Uint8Array(buf);
    if (b.length >= 2 && b[0] === 0xff && b[1] === 0xfe) return new TextDecoder("utf-16le").decode(b);
    if (b.length >= 2 && b[0] === 0xfe && b[1] === 0xff) return new TextDecoder("utf-16be").decode(b);
    let start = 0;
    if (b.length >= 3 && b[0] === 0xef && b[1] === 0xbb && b[2] === 0xbf) start = 3;
    const sub = b.slice(start);
    try {
      return new TextDecoder("utf-8", { fatal: true }).decode(sub);
    } catch (e) {
      return new TextDecoder("iso-8859-1").decode(sub);
    }
  }

  async function handleFile(file) {
    try {
      const text = decodeBuffer(await file.arrayBuffer());
      const trimmed = text.trim();
      const looksJson = /\.json$/i.test(file.name) || trimmed.startsWith("[") || trimmed.startsWith("{");
      const parsed = looksJson
        ? ColCsv.jsonTextToRecords(trimmed)
        : ColCsv.csvTextToRecords(text);
      showImportPreview(file.name, parsed);
    } catch (e) {
      showImportPreview(file.name, { records: [], errors: [`Échec de la lecture : ${e.message}`], warnings: [] });
    }
  }

  function showImportPreview(fileName, { records, errors, warnings }) {
    pendingImport = records;
    const withCoords = records.filter(r => r.lat != null && r.lon != null).length;
    els.importResult.hidden = false;
    els.importSummary.innerHTML =
      `<b>${escapeHtml(fileName)}</b> &middot; ${records.length} col(s) valide(s)` +
      (withCoords !== records.length ? ` &middot; ${records.length - withCoords} sans coordonn&eacute;es` : "") +
      (errors.length ? ` &middot; ${errors.length} ligne(s) ignor&eacute;e(s)` : "") +
      (warnings.length ? ` &middot; ${warnings.length} avertissement(s)` : "");
    els.importErrors.innerHTML =
      errors.slice(0, 8).map(e => `<li>${escapeHtml(e)}</li>`).join("") +
      (errors.length > 8 ? `<li class="more">&hellip; et ${errors.length - 8} autres</li>` : "");
    els.importErrors.hidden = errors.length === 0;
    els.importWarnings.textContent =
      warnings.slice(0, 5).join(" \u00b7 ") + (warnings.length > 5 ? ` \u2026 (+${warnings.length - 5})` : "");
    els.importWarnings.hidden = warnings.length === 0;
    els.btnImportConfirm.textContent = records.length
      ? `Remplacer les donn\u00e9es (${records.length} cols)`
      : "Aucune donn\u00e9e valide";
    els.btnImportConfirm.disabled = records.length === 0;
  }

  function applyPendingImport() {
    if (!pendingImport || !pendingImport.length) return;
    state.all = pendingImport;
    pendingImport = null;
    closeDrawer();
    populateDeptList(state.all);
    resetFilters(); // ends with applyFilters()
    els.emptyState.hidden = true;
    els.dataSource.hidden = false;
    resetImportView();
    switchView("map");
  }

  function resetImportView() {
    pendingImport = null;
    els.importResult.hidden = true;
    els.fileInput.value = "";
    els.dropzone.classList.remove("drag");
  }

  function wireImport() {
    els.btnImport.addEventListener("click", () => {
      state.viewBeforeImport = state.view;
      resetImportView();
      switchView("import");
    });
    els.dropzone.addEventListener("click", () => els.fileInput.click());
    els.dropzone.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        els.fileInput.click();
      }
    });
    els.fileInput.addEventListener("change", () => {
      if (els.fileInput.files && els.fileInput.files[0]) handleFile(els.fileInput.files[0]);
    });
    ["dragenter", "dragover"].forEach(ev =>
      els.dropzone.addEventListener(ev, e => {
        e.preventDefault();
        els.dropzone.classList.add("drag");
      })
    );
    ["dragleave", "drop"].forEach(ev =>
      els.dropzone.addEventListener(ev, e => {
        e.preventDefault();
        els.dropzone.classList.remove("drag");
      })
    );
    els.dropzone.addEventListener("drop", e => {
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) handleFile(file);
    });
    els.btnImportConfirm.addEventListener("click", applyPendingImport);
    els.btnImportCancel.addEventListener("click", () => {
      resetImportView();
      switchView(state.viewBeforeImport || "map");
    });
  }

  // ---------------------------------------------------------------
  // Filtering
  // ---------------------------------------------------------------
  function getAscMin() {
    const n = parseInt(els.ascValue.value, 10);
    if (Number.isNaN(n)) return 0;
    return Math.max(0, Math.min(ASC_MAX, n));
  }

  function syncAscButtons() {
    const v = getAscMin();
    els.ascMinus.disabled = v <= 0;
    els.ascPlus.disabled = v >= ASC_MAX;
  }

  function setAscMin(value) {
    els.ascValue.value = Math.max(0, Math.min(ASC_MAX, value));
    syncAscButtons();
    applyFilters();
  }

  function applyFilters() {
    const q = els.search.value.trim().toLowerCase();
    const dept = state.dept;
    const showPass = els.fPass.checked;
    const showMontee = els.fMontee.checked;
    const highOnly = els.fHigh.checked;
    const mtbOnly = els.fMtb.checked;
    const unclimbedOnly = els.fUnclimbed.checked;
    const altMin = parseInt(els.altMin.value, 10);
    const ascMin = getAscMin();

    state.filtered = state.all.filter(r => {
      if (q && !r.name.toLowerCase().includes(q)) return false;
      if (dept) {
        const key = r.department_code || r.country;
        if (key !== dept) return false;
      }
      if (r.is_pass && !showPass) return false;
      if (!r.is_pass && !showMontee) return false;
      if (highOnly && !r.is_high_altitude) return false;
      if (mtbOnly && !r.is_mtb) return false;
      if (r.altitude < altMin) return false;
      if ((r.ascents || 0) < ascMin) return false;
      const hasCoords = r.lat != null && r.lon != null;
      if (unclimbedOnly && hasCoords) return false;
      if (!unclimbedOnly && !hasCoords) return false;
      return true;
    });

    renderStats();
    renderMap();
    renderList();
  }

  function renderStats() {
    const total = state.filtered.length;
    const passes = state.filtered.filter(r => r.is_pass).length;
    const ascents = state.filtered.reduce((s, r) => s + (r.ascents || 0), 0);
    els.stats.innerHTML = `
      <span><b>${total}</b> cols</span>
      <span><b>${passes}</b> cols de montagne</span>
      <span><b>${ascents}</b> ascensions</span>
    `;
  }

  // ---------------------------------------------------------------
  // Map rendering
  // ---------------------------------------------------------------
  function markerHtml(rec) {
    const cls = ["col-marker", rec.is_pass ? "pass" : "montee", rec.is_high_altitude ? "high" : ""].join(" ").trim();
    return `<div class="${cls}"><div class="pin">${rec.altitude}</div></div>`;
  }

  function renderMap() {
    state.markerLayer.clearLayers();
    state.markers.clear();

    const bounds = [];
    state.filtered.forEach(rec => {
      if (rec.lat == null || rec.lon == null) return;
      const icon = L.divIcon({
        html: markerHtml(rec),
        className: "",
        iconSize: null,
        iconAnchor: [17, 24],
      });
      const marker = L.marker([rec.lat, rec.lon], { icon }).addTo(state.markerLayer);
      marker.on("click", () => openDrawer(rec));
      marker.bindTooltip(rec.name, { direction: "top", offset: [0, -20] });
      state.markers.set(rec.id, marker);
      bounds.push([rec.lat, rec.lon]);
    });

    if (bounds.length === 1) {
      // Same effect as clicking the climb's row in the list view:
      // switch to the map, fly to it and open the detail panel.
      const rec = state.filtered.find(r => r.lat != null && r.lon != null);
      if (rec) openDrawer(rec, { switchToMap: true });
    } else if (bounds.length > 1) {
      state.map.fitBounds(bounds, { padding: [50, 50], maxZoom: 12 });
    }
  }

  // ---------------------------------------------------------------
  // List rendering
  // ---------------------------------------------------------------
  const SORTS = {
    name: {
      defaultDir: "asc",
      cmp: (a, b) => a.name.localeCompare(b.name, "fr", { sensitivity: "base" }),
    },
    loc: {
      defaultDir: "asc",
      cmp: (a, b) => locOf(a).localeCompare(locOf(b), "fr", { sensitivity: "base" }),
    },
    altitude: { defaultDir: "desc", cmp: (a, b) => a.altitude - b.altitude },
    ascents: { defaultDir: "desc", cmp: (a, b) => (a.ascents || 0) - (b.ascents || 0) },
  };

  function locOf(rec) {
    return rec.department_code ? `${rec.department_code} \u2013 ${rec.department_name}` : rec.country;
  }

  function setSort(key) {
    if (state.sort.key === key) {
      state.sort.dir = state.sort.dir === "asc" ? "desc" : "asc";
    } else {
      state.sort = { key, dir: SORTS[key].defaultDir };
    }
    updateSortIndicators();
    renderList();
  }

  function updateSortIndicators() {
    for (const th of els.tableHead.querySelectorAll("th[data-sort]")) {
      const active = th.dataset.sort === state.sort.key;
      const ind = th.querySelector(".sort-ind");
      if (ind) ind.textContent = active ? (state.sort.dir === "asc" ? "\u2191" : "\u2193") : "";
      th.setAttribute(
        "aria-sort",
        active ? (state.sort.dir === "asc" ? "ascending" : "descending") : "none"
      );
    }
  }

  function renderList() {
    const s = SORTS[state.sort.key];
    const mul = state.sort.dir === "asc" ? 1 : -1;
    const rows = state.filtered
      .slice()
      .sort((a, b) => mul * s.cmp(a, b))
      .map(rec => {
        const loc = locOf(rec);
        const badges = [
          rec.is_pass ? '<span class="badge pass">col</span>' : '<span class="badge montee">mont\u00e9e</span>',
          rec.is_high_altitude ? '<span class="badge high">&gt;2000m</span>' : "",
          rec.is_mtb ? '<span class="badge mtb">vtt</span>' : "",
        ].join("");
        return `<tr data-id="${rec.id}">
          <td>${escapeHtml(rec.name)}</td>
          <td>${escapeHtml(loc)}</td>
          <td class="num">${rec.altitude} m</td>
          <td class="num">${rec.ascents}</td>
          <td>${badges}</td>
        </tr>`;
      })
      .join("");
    els.tableBody.innerHTML = rows;

    els.tableBody.querySelectorAll("tr").forEach(tr => {
      tr.addEventListener("click", () => {
        const rec = state.all.find(r => r.id === tr.dataset.id);
        if (rec) openDrawer(rec, { switchToMap: true });
      });
    });
  }

  function cleanName(rec) {
    return rec.name.replace(/\s*\([^)]*\)\s*$/, "").trim();
  }

  function profileLinks(rec) {
    const saved = rec.profile_links || {};
    const name = cleanName(rec);
    return [
      [
        "cols-cyclisme.com",
        saved.cols_cyclisme ||
          `https://www.cols-cyclisme.com/recherche/${encodeURIComponent(name)}.htm`,
      ],
      [
        "mycols.app",
        saved.mycols || `https://mycols.app/fr/cols/recherche?q=${encodeURIComponent(name)}`,
      ],
    ];
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // ---------------------------------------------------------------
  // Detail drawer
  // ---------------------------------------------------------------
  function openDrawer(rec, opts = {}) {
    const loc = rec.department_code ? `${rec.department_code} \u2013 ${rec.department_name}, France` : rec.country;
    const badges = [
      rec.is_pass ? '<span class="badge pass">col</span>' : '<span class="badge montee">mont\u00e9e (sans issue)</span>',
      rec.is_high_altitude ? '<span class="badge high">plus de 2000 m</span>' : "",
      rec.is_mtb ? '<span class="badge mtb">vtt</span>' : "",
    ].join(" ");

    els.drawerContent.innerHTML = `
      <div class="drawer-eyebrow">${escapeHtml(loc)}</div>
      <div class="drawer-name">${escapeHtml(rec.name)}</div>
      <div class="drawer-alt">${rec.altitude} m</div>
      <div class="drawer-badges">${badges}</div>
      <div class="drawer-stats">
        <div class="drawer-stat-row"><span>Ascensions</span><span>${rec.ascents}</span></div>
        <div class="drawer-stat-row"><span>Type</span><span>${rec.is_pass ? "Col de montagne" : "Mont\u00e9e"}</span></div>
        <div class="drawer-stat-row"><span>Coordonn\u00e9es</span><span>${rec.lat != null ? `${rec.lat.toFixed(4)}, ${rec.lon.toFixed(4)}` : "\u2014"}</span></div>
      </div>
      <div class="drawer-links-label">Voir le profil</div>
      <div class="drawer-links">
        ${profileLinks(rec)
          .map(([label, url]) => `<a href="${url}" target="_blank" rel="noopener"><span>${label}</span><span>\u2197</span></a>`)
          .join("")}
      </div>
      <div class="drawer-note">Distance, d\u00e9nivel\u00e9 et pente : voir la fiche sur cols-cyclisme.com ou mycols.app.</div>
    `;
    els.drawer.classList.add("open");
    els.drawer.setAttribute("aria-hidden", "false");

    if (opts.switchToMap && state.view !== "map") switchView("map");
    if (rec.lat != null && rec.lon != null && state.map) {
      state.map.flyTo([rec.lat, rec.lon], Math.max(state.map.getZoom(), 10), { duration: 0.5 });
      const marker = state.markers.get(rec.id);
      if (marker) marker.openTooltip();
    }
  }

  function closeDrawer() {
    els.drawer.classList.remove("open");
    els.drawer.setAttribute("aria-hidden", "true");
  }

  init();
})();
