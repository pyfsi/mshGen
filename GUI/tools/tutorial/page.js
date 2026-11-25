// page.js: helpers injected into the mshGen GUI page by record.py (tutorial video)
// Widgets are found through Bokeh's view tree (model title / label / button text / plot title / tab or card title),
// values are set on the Bokeh models (synchronised to the server like user input), buttons are clicked in the DOM.
window.T = (function () {
  function roots() { return Object.values(window.Bokeh.index || {}); }
  function walk(view, fn, seen) {
    if (!view || seen.has(view)) return;
    seen.add(view);
    fn(view);
    let kids = [];
    try { kids = view.child_views || []; } catch (e) { kids = []; }
    if (kids && typeof kids[Symbol.iterator] === "function") for (const c of kids) walk(c, fn, seen);
    for (const key of ["header_view", "tab_views", "_child_views"]) {
      const v = view[key];
      if (v && typeof v[Symbol.iterator] === "function") for (const c of v) walk(c, fn, seen);
      else if (v && v.values) for (const c of v.values()) walk(c, fn, seen);
    }
  }
  function views() { const out = [], seen = new Set(); for (const r of roots()) walk(r, (v) => out.push(v), seen); return out; }
  function text(m) {
    let t = "";
    for (const k of ["title", "label", "name", "text", "object"]) {
      const v = m[k];
      if (typeof v === "string") t += " " + v;
      else if (v && typeof v === "object" && typeof v.text === "string") t += " " + v.text;
    }
    if (Array.isArray(m.labels)) t += " " + m.labels.join(" | ");
    if (Array.isArray(m.tabs)) t += " " + m.tabs.map((p) => p.title).join(" | ");
    return t;
  }
  // find a view: kind = "title" (exact model title / label), "has" (substring of the model's text), "type"
  function find(what, kind, n) {
    kind = kind || "title";
    const hits = [];
    for (const v of views()) {
      const m = v.model;
      if (!m) continue;
      let ok = false;
      if (kind === "title") ok = m.title === what || m.label === what || (m.title && m.title.text === what);
      else if (kind === "has") ok = text(m).indexOf(what) >= 0;
      else if (kind === "type") ok = m.type === what;
      else if (kind === "labels") ok = Array.isArray(m.labels)          // a label may carry a status mark
        && m.labels.some((l) => l === what || l.startsWith(what + " "));
      else if (kind === "tabs") ok = Array.isArray(m.tabs) && m.tabs.some((p) => p.title === what);
      if (ok && v.el && v.el.getBoundingClientRect().width > 0) hits.push(v);
    }
    return hits[n || 0] || null;
  }
  function el(v) { return v ? v.el : null; }
  let marked = [];
  function clear() {
    for (const e of marked) { e.style.outline = ""; e.style.outlineOffset = ""; e.style.boxShadow = ""; }
    marked = [];
    for (const id of ["T-pop", "T-dot"]) { const p = document.getElementById(id); if (p) p.remove(); }
  }
  function mark(e) {
    if (!e) return;
    e.style.outline = "4px solid #e6550d";
    e.style.outlineOffset = "3px";
    e.style.boxShadow = "0 0 0 9999px rgba(0,0,0,0.12)";
    marked.push(e);
  }
  function scroll(e, block) { if (e) e.scrollIntoView({ block: block || "center", inline: "nearest" }); }
  // pop-up message next to the element (or centred if none); title in bold
  function pop(title, body, e, where) {
    const p = document.createElement("div");
    p.id = "T-pop";
    p.innerHTML = (title ? "<div style='font-weight:700;font-size:21px;margin-bottom:6px;color:#e6550d'>" + title + "</div>" : "")
      + "<div>" + body + "</div>";
    Object.assign(p.style, {
      position: "fixed", zIndex: 100000, maxWidth: "560px", background: "#ffffff", color: "#1d1d1d",
      border: "3px solid #e6550d", borderRadius: "10px", padding: "14px 18px", font: "18px/1.45 Arial, sans-serif",
      boxShadow: "0 6px 24px rgba(0,0,0,0.35)"
    });
    document.body.appendChild(p);
    const W = window.innerWidth, H = window.innerHeight, pr = p.getBoundingClientRect();
    let x = (W - pr.width) / 2, y = H - pr.height - 30;
    if (e && where !== "bottom") {
      const r = e.getBoundingClientRect();
      if (where === "left" || (where !== "right" && r.right + pr.width + 30 > W && r.left - pr.width - 30 > 0)) x = r.left - pr.width - 24;
      else if (r.right + pr.width + 30 <= W) x = r.right + 24;
      else x = Math.max(10, Math.min(W - pr.width - 10, r.left));
      y = Math.max(70, Math.min(H - pr.height - 10, r.top + Math.min(r.height, 300) / 2 - pr.height / 2));
      if (where === "below") { x = Math.max(10, Math.min(W - pr.width - 10, r.left)); y = Math.min(H - pr.height - 10, r.bottom + 18); }
      if (where === "above") { x = Math.max(10, Math.min(W - pr.width - 10, r.left)); y = Math.max(70, r.top - pr.height - 18); }
      if (x < 10 && where === "left") {     // no room on the left: below the top edge of the element, inside it
        x = Math.max(10, r.left + 20); y = Math.max(70, Math.min(H - pr.height - 10, r.top + 20));
      }
      x = Math.max(10, Math.min(W - pr.width - 10, x));
    }
    p.style.left = x + "px"; p.style.top = y + "px";
  }
  function dot(e) {                       // click marker
    if (!e) return;
    const r = e.getBoundingClientRect(), d = document.createElement("div");
    d.id = "T-dot";
    Object.assign(d.style, { position: "fixed", zIndex: 100001, left: (r.left + r.width / 2 - 14) + "px",
      top: (r.top + r.height / 2 - 14) + "px", width: "28px", height: "28px", borderRadius: "14px",
      background: "rgba(230,85,13,0.55)", border: "3px solid #e6550d" });
    document.body.appendChild(d);
  }
  function button(label, n) {             // the <button> of a Bokeh Button / RadioButtonGroup option
    for (const v of views()) {
      const root = v.shadow_el || (v.el && v.el.shadowRoot);
      if (!root) continue;
      for (const b of root.querySelectorAll("button, .bk-btn")) {
        if ((b.textContent || "").trim() === label && b.getBoundingClientRect().width > 0) { if (!n) return b; n -= 1; }
      }
    }
    return null;
  }
  function click(label, n) { const b = button(label, n); if (b) { b.click(); return true; } return false; }
  function buttonIn(tabTitle, label) {    // the button with this label in the active tab of the Tabs with tabTitle
    const tv = find(tabTitle, "tabs");
    if (!tv) return null;
    const active = tv.model.tabs[tv.model.active].child;
    for (const v of views()) {
      if (v.model !== active) continue;
      const out = [], seen = new Set();
      walk(v, (x) => out.push(x), seen);
      for (const x of out) {
        const root = x.shadow_el || (x.el && x.el.shadowRoot);
        if (!root) continue;
        for (const b of root.querySelectorAll("button, .bk-btn")) if ((b.textContent || "").trim() === label) return b;
      }
    }
    return null;
  }
  function set(title, value, kind) {      // set the value of a widget model found by title
    const v = find(title, kind || "title");
    if (!v) return false;
    v.model.value = value;
    return true;
  }
  function radio(label, kind) {           // select an option of a radio/button group (labels may carry a status mark)
    for (const v of views()) {
      const m = v.model;
      if (!m || !Array.isArray(m.labels) || !v.el || v.el.getBoundingClientRect().width === 0) continue;
      const i = m.labels.findIndex((l) => l === label || l.startsWith(label + " "));
      if (i >= 0) { m.active = i; return true; }
    }
    return false;
  }
  function check(label, on) {             // Checkbox (Bokeh Checkbox: label + active true/false) on/off
    const v = find(label, "title") || find(label, "labels");
    if (!v) return false;
    if (typeof v.model.active === "boolean") v.model.active = !!on;
    else v.model.active = on ? [0] : [];
    return true;
  }
  function checks(labels) {               // CheckBoxGroup: these labels active
    const v = find(labels[0], "labels");
    if (!v) return false;
    v.model.active = labels.map((l) => v.model.labels.indexOf(l)).filter((i) => i >= 0);
    return true;
  }
  function tab(title) {
    const v = find(title, "tabs");
    if (!v) return false;
    v.model.active = v.model.tabs.findIndex((p) => p.title === title);
    return true;
  }
  function deeptext(m, depth) {           // text of a model and its children (Card headers)
    if (!m || depth > 4) return "";
    let t = text(m);
    for (const k of ["children", "header"]) {
      const c = m[k];
      if (Array.isArray(c)) for (const x of c) t += " " + deeptext(Array.isArray(x) ? x[0] : x, depth + 1);
      else if (c && typeof c === "object") t += " " + deeptext(c, depth + 1);
    }
    return t;
  }
  function cardview(title) {
    for (const v of views()) {
      const m = v.model;
      if (m && m.collapsed !== undefined && Array.isArray(m.children) && m.children.length
          && deeptext(Array.isArray(m.children[0]) ? m.children[0][0] : m.children[0], 0).indexOf(title) >= 0) return v;
    }
    return null;
  }
  function card(title, open) {            // Panel Card: expand / collapse
    const v = cardview(title);
    if (!v) return false;
    v.model.collapsed = !open;
    return true;
  }
  function drag(vid, dx, dz) {            // move a draggable (exterior or airfoil) vertex like the point-draw tool
    for (const d of window.Bokeh.documents) for (const m of d._all_models.values()) {
      if (m.data !== undefined && m.data.id !== undefined && m.data.z !== undefined
          && (m.data.id[0] === "22" || m.data.color !== undefined) && m.data.id.indexOf(String(vid)) >= 0) {
        const i = m.data.id.indexOf(String(vid));
        m.data.x[i] += dx; m.data.z[i] += dz;
        m.setv({ data: m.data }, { check_eq: false });
        return true;
      }
    }
    return false;
  }
  function typed(type, n) {               // n-th visible view of a model type (substring match)
    const hits = views().filter((v) => v.model && v.model.type.indexOf(type) >= 0 && v.el
                                       && v.el.getBoundingClientRect().width > 0);
    return hits[n || 0] || null;
  }
  function rect(e) { const r = e.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; }
  function scrollEnd(what) {              // scroll the box holding the view with this text to its end (a log)
    const v = find(what, "has");
    let e = v ? v.el : null;
    while (e) {
      const st = getComputedStyle(e);
      if (e.scrollHeight > e.clientHeight + 5 && (st.overflowY === "auto" || st.overflowY === "scroll")) {
        if (e.clientHeight > 600) return false;          // the page itself: leave it
        e.scrollTop = e.scrollHeight;
        return true;
      }
      e = e.parentElement || (e.getRootNode && e.getRootNode().host) || null;
    }
    return false;
  }
  return { views, find, el, clear, mark, scroll, pop, dot, button, buttonIn, click, set, radio, check, checks, tab,
           card, cardview, drag, text, typed, rect, scrollEnd };
})();
true;
