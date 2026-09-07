/* JSON <-> CSV converter for the app's data.
 *
 * Pure functions, no DOM: exposed as window.ColCsv in the browser and via
 * module.exports in Node (used by the round-trip test). CSV parsing/writing
 * is delegated to PapaParse (loaded from a CDN in index.html).
 *
 * CSV format ("mes-cols.csv"): ";"-delimited, UTF-8 (a BOM is added at
 * download time so Excel/LibreOffice detect the encoding). One record per
 * line, header:
 *
 *   nom; pays; departement_code; departement; altitude; ascensions;
 *   est_un_col; vtt; latitude; longitude; link_cols_cyclisme; link_mycols
 *
 * The import is lenient on purpose (the file round-trips through Excel /
 * LibreOffice): comma decimals are accepted, booleans as Oui/Non/X/...,
 * a département name can be given instead of its code (and a code without
 * its leading zero, which Excel eats), unknown columns are ignored.
 * Derived fields (id, is_high_altitude, geocode_status, geocode_query)
 * are recomputed on import.
 */
(() => {
  "use strict";

  // PapaParse is global in the browser (CDN script tag); require() in Node.
  // If the CDN is down, the app must keep working (map/list) — only the
  // CSV download/import features then fail with a console warning.
  const Papa =
    typeof window !== "undefined" && window.Papa
      ? window.Papa
      : typeof require === "function"
        ? require("papaparse")
        : ((console.warn("PapaParse manquant (CDN injoignable) : export/import CSV indisponibles."), undefined));

  function needPapa() {
    if (!Papa) throw new Error("PapaParse manquant");
  }

  // French department INSEE code -> name. Kept in sync with
  // scripts/parse_csv.py (DEPARTMENTS).
  const DEPARTMENTS = {
    "01": "Ain",
    "02": "Aisne",
    "03": "Allier",
    "04": "Alpes-de-Haute-Provence",
    "05": "Hautes-Alpes",
    "06": "Alpes-Maritimes",
    "07": "Ardèche",
    "08": "Ardennes",
    "09": "Ariège",
    "10": "Aube",
    "11": "Aude",
    "12": "Aveyron",
    "13": "Bouches-du-Rhône",
    "14": "Calvados",
    "15": "Cantal",
    "16": "Charente",
    "17": "Charente-Maritime",
    "18": "Cher",
    "19": "Corrèze",
    "2A": "Corse-du-Sud",
    "2B": "Haute-Corse",
    "21": "Côte-d'Or",
    "22": "Côtes-d'Armor",
    "23": "Creuse",
    "24": "Dordogne",
    "25": "Doubs",
    "26": "Drôme",
    "27": "Eure",
    "28": "Eure-et-Loir",
    "29": "Finistère",
    "30": "Gard",
    "31": "Haute-Garonne",
    "32": "Gers",
    "33": "Gironde",
    "34": "Hérault",
    "35": "Ille-et-Vilaine",
    "36": "Indre",
    "37": "Indre-et-Loire",
    "38": "Isère",
    "39": "Jura",
    "40": "Landes",
    "41": "Loir-et-Cher",
    "42": "Loire",
    "43": "Haute-Loire",
    "44": "Loire-Atlantique",
    "45": "Loiret",
    "46": "Lot",
    "47": "Lot-et-Garonne",
    "48": "Lozère",
    "49": "Maine-et-Loire",
    "50": "Manche",
    "51": "Marne",
    "52": "Haute-Marne",
    "53": "Mayenne",
    "54": "Meurthe-et-Moselle",
    "55": "Meuse",
    "56": "Morbihan",
    "57": "Moselle",
    "58": "Nièvre",
    "59": "Nord",
    "60": "Oise",
    "61": "Orne",
    "62": "Pas-de-Calais",
    "63": "Puy-de-Dôme",
    "64": "Pyrénées-Atlantiques",
    "65": "Hautes-Pyrénées",
    "66": "Pyrénées-Orientales",
    "67": "Bas-Rhin",
    "68": "Haut-Rhin",
    "69": "Rhône",
    "70": "Haute-Saône",
    "71": "Saône-et-Loire",
    "72": "Sarthe",
    "73": "Savoie",
    "74": "Haute-Savoie",
    "75": "Paris",
    "76": "Seine-Maritime",
    "77": "Seine-et-Marne",
    "78": "Yvelines",
    "79": "Deux-Sèvres",
    "80": "Somme",
    "81": "Tarn",
    "82": "Tarn-et-Garonne",
    "83": "Var",
    "84": "Vaucluse",
    "85": "Vendée",
    "86": "Vienne",
    "87": "Haute-Vienne",
    "88": "Vosges",
    "89": "Yonne",
    "90": "Territoire de Belfort",
    "91": "Essonne",
    "92": "Hauts-de-Seine",
    "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne",
    "95": "Val-d'Oise",
    "971": "Guadeloupe",
    "972": "Martinique",
    "973": "Guyane",
    "974": "La Réunion",
    "976": "Mayotte",
  };

  const DEPT_BY_NAME = {};
  for (const [code, name] of Object.entries(DEPARTMENTS)) {
    DEPT_BY_NAME[stripAccents(name).toLowerCase()] = code;
  }

  const HEADER = [
    "nom",
    "pays",
    "departement_code",
    "departement",
    "altitude",
    "ascensions",
    "est_un_col",
    "vtt",
    "latitude",
    "longitude",
    "link_cols_cyclisme",
    "link_mycols",
  ];

  // Header cell (normalized) -> canonical field. Aliases make the import
  // tolerant of small header tweaks.
  const HEADER_ALIASES = {
    nom: "name",
    name: "name",
    pays: "country",
    country: "country",
    departement_code: "department_code",
    code_departement: "department_code",
    departement: "department_name",
    department: "department_name",
    altitude: "altitude",
    altitude_m: "altitude",
    ascensions: "ascents",
    est_un_col: "is_pass",
    col: "is_pass",
    vtt: "is_mtb",
    mtb: "is_mtb",
    latitude: "lat",
    lat: "lat",
    longitude: "lon",
    lon: "lon",
    link_cols_cyclisme: "cols_cyclisme",
    cols_cyclisme: "cols_cyclisme",
    link_mycols: "mycols",
    mycols: "mycols",
  };

  function stripAccents(s) {
    return String(s).normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
  }

  // Same algorithm as scripts/parse_csv.py slugify().
  function slugify(text) {
    let s = String(text).normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
    s = s.replace(/[^\x00-\x7F]/g, ""); // encode("ascii", "ignore")
    s = s.replace(/[^a-zA-Z0-9]+/g, "-").replace(/^-+|-+$/g, "").toLowerCase();
    return s;
  }

  function parseNumber(v) {
    if (v == null) return null;
    if (typeof v === "number") return Number.isFinite(v) ? v : null;
    let s = String(v).trim();
    if (!s) return null;
    s = s.replace(/\s/g, "").replace(",", "."); // French decimal comma
    if (!/^[+-]?\d+(\.\d+)?$/.test(s)) return null;
    return parseFloat(s);
  }

  function parseBool(v) {
    if (typeof v === "boolean") return v;
    if (v == null) return false;
    const s = String(v).trim().toLowerCase();
    return ["oui", "x", "vrai", "true", "1", "yes", "y", "o"].includes(s);
  }

  // ---------------------------------------------------------------
  // JSON -> CSV (export)
  // ---------------------------------------------------------------
  function recordsToCsv(records) {
    needPapa();
    const rows = [HEADER.slice()];
    for (const r of records) {
      rows.push([
        r.name,
        r.country,
        r.department_code || "",
        r.department_name || "",
        r.altitude,
        r.ascents,
        r.is_pass ? "Oui" : "Non",
        r.is_mtb ? "Oui" : "Non",
        r.lat == null ? "" : r.lat,
        r.lon == null ? "" : r.lon,
        (r.profile_links && r.profile_links.cols_cyclisme) || "",
        (r.profile_links && r.profile_links.mycols) || "",
      ]);
    }
    return Papa.unparse(rows, { delimiter: ";", newline: "\n" });
  }

  // ---------------------------------------------------------------
  // Text -> records (import)
  // ---------------------------------------------------------------
  function headerKey(h) {
    return stripAccents(String(h == null ? "" : h))
      .trim()
      .toLowerCase()
      .replace(/[\s_\-]+/g, "_");
  }

  function csvTextToRecords(text) {
    needPapa();
    const t = String(text).replace(/^\uFEFF/, "");
    const parsed = Papa.parse(t, { skipEmptyLines: true });
    const rows = parsed.data || [];
    if (rows.length === 0) return { records: [], errors: ["fichier vide"], warnings: [] };

    const fields = rows[0].map(h => HEADER_ALIASES[headerKey(h)]);
    if (!fields.includes("name")) {
      return {
        records: [],
        errors: [
          `en-tête non reconnue : colonne « nom » introuvable (première ligne : ${rows[0].join(";")})`,
        ],
        warnings: [],
      };
    }

    const items = [];
    for (let i = 1; i < rows.length; i++) {
      const raw = {};
      for (let j = 0; j < fields.length; j++) {
        if (fields[j]) raw[fields[j]] = rows[i][j] == null ? "" : rows[i][j];
      }
      items.push({ raw, label: `l.${i + 1}` });
    }
    return buildRecords(items);
  }

  function jsonTextToRecords(text) {
    let data;
    try {
      data = JSON.parse(text);
    } catch (e) {
      return { records: [], errors: [`JSON invalide : ${e.message}`], warnings: [] };
    }
    if (!Array.isArray(data)) {
      return {
        records: [],
        errors: ["JSON : attendu un tableau de cols (format data/cols_geocoded.json)"],
        warnings: [],
      };
    }
    const items = data.map((r, i) => ({
      label: `n°${i + 1}`,
      raw: {
        id: r.id,
        name: r.name,
        country: r.country,
        department_code: r.department_code,
        department_name: r.department_name,
        altitude: r.altitude,
        ascents: r.ascents,
        is_pass: r.is_pass,
        is_mtb: r.is_mtb,
        lat: r.lat,
        lon: r.lon,
        cols_cyclisme: r.profile_links ? r.profile_links.cols_cyclisme : "",
        mycols: r.profile_links ? r.profile_links.mycols : "",
      },
    }));
    return buildRecords(items);
  }

  // ---------------------------------------------------------------
  // Normalization (shared by CSV and JSON import)
  // ---------------------------------------------------------------
  function finalizeRecord(raw, label) {
    const name = String(raw.name == null ? "" : raw.name).trim();
    if (!name) return { error: `${label} : nom manquant` };

    const altitude = parseNumber(raw.altitude);
    if (altitude == null) return { error: `${label} (${name}) : altitude absente ou invalide` };

    let ascents = 0;
    if (raw.ascents != null && String(raw.ascents).trim() !== "") {
      ascents = parseNumber(raw.ascents);
      if (ascents == null || ascents < 0) return { error: `${label} (${name}) : ascensions invalides` };
    }

    const lat = raw.lat == null || String(raw.lat).trim() === "" ? null : parseNumber(raw.lat);
    const lon = raw.lon == null || String(raw.lon).trim() === "" ? null : parseNumber(raw.lon);
    if (lat != null && (lat < -90 || lat > 90)) return { error: `${label} (${name}) : latitude hors bornes (${lat})` };
    if (lon != null && (lon < -180 || lon > 180)) return { error: `${label} (${name}) : longitude hors bornes (${lon})` };

    const warnings = [];
    let country = String(raw.country == null ? "" : raw.country).trim();
    let code = raw.department_code == null ? "" : String(raw.department_code).trim();
    let deptName = raw.department_name == null ? "" : String(raw.department_name).trim();
    if (/^\d$/.test(code)) code = `0${code}`; // Excel eats the leading zero

    if (code) {
      if (Object.prototype.hasOwnProperty.call(DEPARTMENTS, code)) {
        if (!deptName) deptName = DEPARTMENTS[code];
        if (!country) country = "France";
      } else {
        warnings.push(`${label} (${name}) : code département inconnu « ${code} »`);
        if (!country) country = "France";
      }
    } else if (deptName) {
      const found = DEPT_BY_NAME[stripAccents(deptName).toLowerCase()];
      if (found) code = found;
      else warnings.push(`${label} (${name}) : département non reconnu « ${deptName} »`);
      if (!country) country = "France";
    }
    if (!country) return { error: `${label} (${name}) : pays manquant` };

    const cc = raw.cols_cyclisme == null ? "" : String(raw.cols_cyclisme).trim();
    const mc = raw.mycols == null ? "" : String(raw.mycols).trim();

    return {
      record: {
        id: raw.id != null && String(raw.id).trim() ? String(raw.id).trim() : null,
        name,
        department_code: code || null,
        department_name: deptName || null,
        country,
        altitude: Math.round(altitude),
        ascents: Math.round(ascents),
        is_pass: parseBool(raw.is_pass),
        is_mtb: parseBool(raw.is_mtb),
        lat,
        lon,
        profile_links: cc || mc ? { cols_cyclisme: cc || null, mycols: mc || null } : undefined,
      },
      warnings,
    };
  }

  // Assign ids, then derive the rest. Ids provided by a JSON import are
  // kept; CSV imports (no id column) re-slug the names with the same
  // collision scheme as scripts/parse_csv.py, so a round-trip keeps the
  // original ids as long as names are unchanged.
  function buildRecords(items) {
    const errors = [];
    const warnings = [];
    const records = [];
    for (const item of items) {
      const res = finalizeRecord(item.raw, item.label);
      if (res.error) {
        errors.push(res.error);
        continue;
      }
      records.push(res.record);
      for (const w of res.warnings) warnings.push(w);
    }

    const used = new Set();
    for (const r of records) {
      let id = r.id;
      if (id && used.has(id)) id = null;
      if (!id) {
        const base = slugify(r.name);
        let n = 1;
        let candidate = base;
        while (used.has(candidate)) {
          n += 1;
          candidate = `${base}-${n}`;
        }
        id = candidate;
      }
      used.add(id);
      r.id = id;
    }

    const nameCount = new Map();
    for (const r of records) {
      const k = r.name.toLowerCase();
      nameCount.set(k, (nameCount.get(k) || 0) + 1);
    }
    for (const [name, n] of nameCount) {
      if (n > 1) warnings.push(`nom en double : « ${name} » (${n} fois)`);
    }

    for (const r of records) {
      r.is_high_altitude = r.altitude >= 2000;
      const hasCoords = r.lat != null && r.lon != null;
      r.geocode_status = hasCoords ? "ok" : "pending";
      const clean = r.name.replace(/\s*\([^)]*\)\s*$/, "").trim();
      const parts = [clean];
      if (r.department_name) parts.push(r.department_name);
      parts.push(r.country);
      r.geocode_query = parts.join(", ");
    }

    return { records, errors, warnings };
  }

  const api = {
    HEADER,
    DEPARTMENTS,
    slugify,
    parseNumber,
    parseBool,
    recordsToCsv,
    csvTextToRecords,
    jsonTextToRecords,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.ColCsv = api;
})();
