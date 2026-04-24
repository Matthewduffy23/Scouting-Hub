<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Scout Board Pro</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #07090f;
  --s0: #0b0e18;
  --s1: #0f1420;
  --s2: #141b28;
  --s3: #1a2235;
  --s4: #1f293f;
  --border: #1e2d45;
  --border2: #28395a;
  --text: #e4eaf5;
  --muted: #6b82a0;
  --muted2: #3f5270;
  --accent: #00c8f0;
  --accent2: #0099cc;

  /* Status colours */
  --c-summer: #e53e3e;
  --c-monitor: #3182ce;
  --c-january: #d69e2e;
  --c-signed: #38a169;
  --c-rejected: #718096;

  /* Target move */
  --c-pl: #38a169;
  --c-champ: #d69e2e;
  --c-eb1: #e53e3e;
  --c-eb12: #dd6b20;
  --c-eb23: #d69e2e;
  --c-plmon: #3182ce;
  --c-chmon: #805ad5;
  --c-tbc: #4a5568;

  /* Score bands */
  --score-80: #2b6cb0;
  --score-76: #3182ce;
  --score-72: #63b3ed;
  --score-66: #718096;
  --score-60: #4a5568;

  /* Status column */
  --c-noprog: #c53030;
  --c-relationship: #2f855a;
  --c-agency: #b7791f;
  --c-contact: #6b46c1;

  /* Agency */
  --c-small: #276749;
  --c-medium: #b7791f;
  --c-big: #9b2c2c;

  /* Recent move */
  --c-yes: #276749;
  --c-no: #c53030;
}

* { margin:0; padding:0; box-sizing:border-box; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* Grain */
body::after {
  content:'';
  position:fixed;
  inset:0;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E");
  pointer-events:none;
  z-index:9998;
}

/* ── TOP BAR ── */
.topbar {
  height: 52px;
  background: var(--s1);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 20px;
  gap: 16px;
  flex-shrink: 0;
  z-index: 50;
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-right: 8px;
}

.logo-mark {
  width: 30px;
  height: 30px;
  background: linear-gradient(135deg,var(--accent),var(--accent2));
  border-radius: 7px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 15px;
  flex-shrink: 0;
}

.logo-text {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 19px;
  letter-spacing: 2px;
  color: var(--text);
  line-height: 1;
}

.logo-sub {
  font-size: 9px;
  color: var(--muted2);
  letter-spacing: 1.5px;
  text-transform: uppercase;
}

.topbar-sep { width:1px; height:28px; background:var(--border); }

.board-name {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 17px;
  letter-spacing: 1.5px;
  color: var(--text);
  cursor: pointer;
}

.topbar-spacer { flex:1; }

.btn {
  padding: 6px 14px;
  border-radius: 7px;
  font-family: 'DM Sans', sans-serif;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  border: none;
  transition: all 0.15s;
  white-space: nowrap;
}

.btn-primary {
  background: var(--accent);
  color: #000;
  font-weight: 600;
}
.btn-primary:hover { background: #33d4f5; }

.btn-ghost {
  background: transparent;
  color: var(--muted);
  border: 1px solid var(--border);
}
.btn-ghost:hover { border-color: var(--border2); color: var(--text); }

.btn-sm { padding: 4px 10px; font-size: 11px; }

/* ── TABS ── */
.tabs {
  height: 40px;
  background: var(--s1);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: flex-end;
  padding: 0 20px;
  gap: 2px;
  flex-shrink: 0;
}

.tab {
  padding: 8px 16px;
  font-size: 12px;
  font-weight: 500;
  color: var(--muted);
  cursor: pointer;
  border-radius: 6px 6px 0 0;
  border: 1px solid transparent;
  border-bottom: none;
  transition: all 0.15s;
  white-space: nowrap;
  position: relative;
  bottom: -1px;
}

.tab:hover { color: var(--text); background: var(--s2); }

.tab.active {
  color: var(--text);
  background: var(--s2);
  border-color: var(--border);
  border-bottom-color: var(--s2);
}

.tab-add {
  padding: 6px 10px;
  color: var(--muted2);
  cursor: pointer;
  border-radius: 6px;
  font-size: 16px;
  transition: all 0.15s;
}
.tab-add:hover { color: var(--text); }

/* ── TOOLBAR ── */
.toolbar {
  height: 46px;
  background: var(--s2);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 20px;
  gap: 10px;
  flex-shrink: 0;
}

.search-wrap {
  position: relative;
  width: 260px;
}

.search-icon {
  position: absolute;
  left: 10px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--muted);
  font-size: 12px;
  pointer-events: none;
}

.search-input {
  width: 100%;
  background: var(--s3);
  border: 1px solid var(--border);
  border-radius: 7px;
  padding: 6px 10px 6px 30px;
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
  font-size: 12px;
  outline: none;
  transition: border-color 0.2s;
}
.search-input:focus { border-color: var(--accent); }
.search-input::placeholder { color: var(--muted2); }

.filter-sel {
  background: var(--s3);
  border: 1px solid var(--border);
  border-radius: 7px;
  padding: 6px 10px;
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
  font-size: 12px;
  outline: none;
  cursor: pointer;
}

.toolbar-sep { flex:1; }

.record-count {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--muted2);
  background: var(--s3);
  padding: 4px 10px;
  border-radius: 6px;
  border: 1px solid var(--border);
}

/* ── MAIN AREA ── */
.main {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* ── TABLE ── */
.table-scroll {
  flex: 1;
  overflow: auto;
}

.table-scroll::-webkit-scrollbar { width:6px; height:6px; }
.table-scroll::-webkit-scrollbar-track { background: transparent; }
.table-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius:4px; }

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12.5px;
  min-width: 1600px;
}

thead {
  position: sticky;
  top: 0;
  z-index: 20;
}

thead th {
  background: var(--s1);
  padding: 10px 12px;
  text-align: left;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--muted);
  border-bottom: 2px solid var(--border);
  border-right: 1px solid var(--border);
  white-space: nowrap;
  cursor: pointer;
  user-select: none;
  transition: color 0.15s;
}
thead th:hover { color: var(--text); }
thead th:last-child { border-right: none; }

.th-check { width: 36px; }
.th-player { min-width: 180px; }

tbody tr {
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  transition: background 0.1s;
  animation: rowIn 0.25s ease forwards;
  opacity: 0;
}

@keyframes rowIn {
  to { opacity:1; }
}

tbody tr:hover { background: rgba(0,200,240,0.04); }

tbody td {
  padding: 10px 12px;
  border-right: 1px solid rgba(30,45,69,0.4);
  white-space: nowrap;
  vertical-align: middle;
  color: var(--text);
  font-size: 13px;
}
tbody td:last-child { border-right: none; }

.td-player {
  display: flex;
  align-items: center;
  gap: 9px;
  min-width: 180px;
}

.player-avatar {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  object-fit: cover;
  background: var(--s4);
  border: 1px solid var(--border);
  flex-shrink: 0;
}

.player-avatar-ph {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: var(--s4);
  border: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  flex-shrink: 0;
  color: var(--muted);
}

.player-name {
  font-weight: 600;
  font-size: 13px;
  color: var(--text);
}

.player-pos-tag {
  font-size: 9px;
  font-weight: 700;
  padding: 2px 5px;
  border-radius: 3px;
  background: var(--s4);
  color: var(--muted);
  letter-spacing: 0.5px;
}

/* Coloured pills */
.pill {
  display: inline-flex;
  align-items: center;
  padding: 3px 10px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
}

.pill-summer { background: var(--c-summer); color: #fff; }
.pill-monitor { background: var(--c-monitor); color: #fff; }
.pill-january { background: var(--c-january); color: #fff; }
.pill-signed { background: var(--c-signed); color: #fff; }
.pill-rejected { background: var(--c-rejected); color: #fff; }

.pill-pl { background: var(--c-pl); color: #fff; }
.pill-champ { background: var(--c-champ); color: #fff; }
.pill-eb1 { background: var(--c-eb1); color: #fff; }
.pill-eb12 { background: var(--c-eb12); color: #fff; }
.pill-eb23 { background: var(--c-eb23); color: #fff; }
.pill-plmon { background: var(--c-plmon); color: #fff; }
.pill-chmon { background: var(--c-chmon); color: #fff; }
.pill-tbc { background: var(--c-tbc); color: #aaa; }

.pill-noprog { background: var(--c-noprog); color: #fff; }
.pill-relationship { background: var(--c-relationship); color: #fff; }
.pill-agency { background: var(--c-agency); color: #fff; }
.pill-contact { background: var(--c-contact); color: #fff; }

.pill-small { background: var(--c-small); color: #fff; }
.pill-medium { background: var(--c-medium); color: #fff; }
.pill-big { background: var(--c-big); color: #fff; }

.pill-yes { background: var(--c-yes); color: #fff; }
.pill-no { background: var(--c-no); color: #fff; }

/* Score bands */
.score-band {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 4px 10px;
  border-radius: 5px;
  font-family: 'DM Mono', monospace;
  font-size: 11.5px;
  font-weight: 600;
  min-width: 60px;
  text-align: center;
  letter-spacing: 0.3px;
}

.sb-80 { background: #1a4731; color: #6ee7b7; border: 1px solid rgba(110,231,183,0.3); }
.sb-76 { background: #1a3a4a; color: #7dd3fc; border: 1px solid rgba(125,211,252,0.3); }
.sb-72 { background: #2d3a1a; color: #bef264; border: 1px solid rgba(190,242,100,0.3); }
.sb-66 { background: #3a2a0a; color: #fde68a; border: 1px solid rgba(253,230,138,0.3); }
.sb-60 { background: #3a1a0a; color: #fdba74; border: 1px solid rgba(253,186,116,0.3); }
.sb-55 { background: var(--s3); color: var(--muted); border: 1px solid var(--border); }

/* Stars */
.stars { color: #f6c90e; font-size: 13px; letter-spacing: -1px; }
.stars-empty { color: var(--s4); font-size: 13px; letter-spacing: -1px; }

/* Role badge */
.role-badge {
  display: inline-flex;
  align-items: center;
  padding: 3px 8px;
  border-radius: 4px;
  font-size: 10.5px;
  font-weight: 700;
  background: rgba(0,200,240,0.1);
  color: var(--accent);
  border: 1px solid rgba(0,200,240,0.25);
  margin-right: 3px;
  letter-spacing: 0.3px;
}

/* Style tag */
.style-tag {
  font-size: 11px;
  color: #c4b5fd;
  background: rgba(167,139,250,0.1);
  padding: 3px 9px;
  border-radius: 4px;
  border: 1px solid rgba(167,139,250,0.25);
}

/* Physical tag */
.phys-tag {
  font-size: 11px;
  padding: 3px 9px;
  border-radius: 4px;
  background: rgba(148,163,184,0.1);
  color: #94a3b8;
  border: 1px solid rgba(148,163,184,0.2);
}

/* Mono values */
.mono { font-family: 'DM Mono', monospace; font-size: 12.5px; }
.val-green { color: #4ade80; font-weight: 500; }
.val-gold { color: #f6e05e; }

/* TM Link */
.tm-link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: var(--accent);
  text-decoration: none;
  background: rgba(0,200,240,0.08);
  padding: 3px 8px;
  border-radius: 4px;
  border: 1px solid rgba(0,200,240,0.2);
  transition: all 0.15s;
}
.tm-link:hover { background: rgba(0,200,240,0.15); }

/* Data docs */
.doc-icons { display: flex; gap: 3px; }
.doc-icon {
  width: 20px;
  height: 24px;
  background: var(--s4);
  border: 1px solid var(--border);
  border-radius: 3px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  cursor: pointer;
  transition: all 0.15s;
}
.doc-icon:hover { background: var(--s3); border-color: var(--border2); }
.doc-icon.filled { background: rgba(66,153,225,0.15); border-color: rgba(66,153,225,0.4); color: #63b3ed; }

/* Notes text */
.notes-text {
  max-width: 180px;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--muted);
  font-size: 11px;
}

/* Checkbox */
.cb-wrap { display: flex; justify-content: center; }

.table-checkbox {
  width: 14px;
  height: 14px;
  accent-color: var(--accent);
  cursor: pointer;
}

/* ── MODAL ── */
.overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.75);
  backdrop-filter: blur(6px);
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.2s;
}
.overlay.open { opacity:1; pointer-events:all; }

.modal {
  background: var(--s1);
  border: 1px solid var(--border);
  border-radius: 14px;
  width: 680px;
  max-width: 95vw;
  max-height: 92vh;
  overflow-y: auto;
  transform: translateY(16px) scale(0.98);
  transition: transform 0.2s;
  box-shadow: 0 32px 100px rgba(0,0,0,0.6);
}
.overlay.open .modal { transform: translateY(0) scale(1); }
.modal::-webkit-scrollbar { width:4px; }
.modal::-webkit-scrollbar-thumb { background: var(--border); border-radius:4px; }

.modal-head {
  padding: 18px 22px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  background: var(--s1);
  z-index: 5;
}

.modal-title {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 20px;
  letter-spacing: 1.5px;
}

.modal-x {
  width: 28px;
  height: 28px;
  border-radius: 7px;
  background: var(--s3);
  border: 1px solid var(--border);
  color: var(--muted);
  cursor: pointer;
  font-size: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.15s;
}
.modal-x:hover { color: var(--text); }

.modal-body { padding: 20px 22px; }

.form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.form-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
.form-full { grid-column: 1/-1; }

.form-group { display: flex; flex-direction: column; gap: 5px; }

.form-label {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  color: var(--muted2);
}

.form-input, .form-select, .form-textarea {
  background: var(--s3);
  border: 1px solid var(--border);
  border-radius: 7px;
  padding: 8px 10px;
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
  font-size: 12.5px;
  outline: none;
  transition: border-color 0.2s;
}
.form-input:focus, .form-select:focus, .form-textarea:focus { border-color: var(--accent); }
.form-textarea { resize: vertical; min-height: 70px; }

.section-divider {
  margin: 20px 0 14px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.section-divider-label {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--muted2);
  white-space: nowrap;
}

.section-divider-line {
  flex: 1;
  height: 1px;
  background: var(--border);
}

.modal-foot {
  padding: 14px 22px;
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: sticky;
  bottom: 0;
  background: var(--s1);
}

/* ── DETAIL PANEL ── */
.detail-modal { width: 820px; }

.detail-hero {
  display: flex;
  gap: 20px;
  padding: 22px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(135deg, var(--s2), var(--s1));
}

.detail-photo {
  width: 90px;
  height: 90px;
  border-radius: 12px;
  object-fit: cover;
  background: var(--s4);
  border: 2px solid var(--border);
  flex-shrink: 0;
}

.detail-photo-ph {
  width: 90px;
  height: 90px;
  border-radius: 12px;
  background: var(--s4);
  border: 2px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 36px;
  flex-shrink: 0;
}

.detail-info { flex: 1; }

.detail-name {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 32px;
  letter-spacing: 1px;
  line-height: 1;
  margin-bottom: 6px;
}

.detail-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 10px;
  font-size: 12px;
  color: var(--muted);
}

.detail-meta-chip {
  background: var(--s3);
  padding: 3px 8px;
  border-radius: 4px;
  border: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 4px;
}

.detail-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }

/* Stats grid in detail */
.detail-stats-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
}

.detail-stat {
  padding: 14px 16px;
  text-align: center;
  border-right: 1px solid var(--border);
}
.detail-stat:last-child { border-right: none; }

.ds-val {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 24px;
  letter-spacing: 0.5px;
  color: var(--text);
  display: block;
}

.ds-lbl {
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--muted2);
  margin-top: 3px;
}

/* Score display in detail */
.score-display {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.score-bar-wrap {
  width: 100%;
  height: 3px;
  background: var(--s4);
  border-radius: 2px;
  overflow: hidden;
  margin-top: 4px;
}

.score-bar-fill {
  height: 100%;
  border-radius: 2px;
  background: var(--accent);
}

/* Notes */
.notes-area { padding: 18px 22px; }

.notes-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.notes-title {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--muted2);
}

.note-item {
  background: var(--s3);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 11px 14px;
  margin-bottom: 8px;
}

.note-text { font-size: 13px; color: var(--text); line-height: 1.55; }
.note-date { font-size: 10px; color: var(--muted2); margin-top: 5px; font-family: 'DM Mono', monospace; }

.note-input-row { display: flex; gap: 8px; margin-top: 12px; }
.note-input-row .form-input { flex: 1; }

/* ── TOAST ── */
.toast {
  position: fixed;
  bottom: 20px;
  right: 20px;
  background: var(--s2);
  border: 1px solid var(--border);
  border-radius: 9px;
  padding: 10px 16px;
  font-size: 12.5px;
  color: var(--text);
  z-index: 9000;
  transform: translateY(60px);
  opacity: 0;
  transition: all 0.25s;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  pointer-events: none;
  display: flex;
  align-items: center;
  gap: 8px;
}
.toast.show { transform: translateY(0); opacity: 1; }

/* ── EMPTY STATE ── */
.empty {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: var(--muted2);
  font-size: 13px;
}
.empty-icon { font-size: 40px; opacity: 0.3; }

/* ── MISC ── */
.hidden { display: none !important; }
.text-right { text-align: right; }
.nowrap { white-space: nowrap; }

/* Context actions on row hover */
.row-actions {
  display: none;
  gap: 4px;
}
tbody tr:hover .row-actions { display: flex; }

/* By League view */
.league-group { margin-bottom: 0; }

.league-group-header {
  padding: 10px 20px;
  background: var(--s2);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  position: sticky;
  top: 0;
  z-index: 5;
}

.league-group-name {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 15px;
  letter-spacing: 1.5px;
  color: var(--accent);
}

.league-group-count {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--muted2);
  background: var(--s3);
  padding: 2px 7px;
  border-radius: 4px;
}

/* Sortable header indicator */
th.sort-asc::after { content: ' ↑'; color: var(--accent); }
th.sort-desc::after { content: ' ↓'; color: var(--accent); }

/* View toggle */
.view-toggle-btn {
  padding: 5px 12px;
  background: transparent;
  border: none;
  color: var(--muted);
  cursor: pointer;
  font-size: 11px;
  font-family: 'DM Sans', sans-serif;
  font-weight: 500;
  transition: all 0.15s;
}
.view-toggle-btn.active { background: var(--s4); color: var(--text); }
.view-toggle-btn:hover:not(.active) { color: var(--text); }

/* League group */
.lg-header {
  padding: 10px 20px;
  background: var(--s2);
  border-bottom: 1px solid var(--border);
  border-top: 2px solid var(--accent);
  display: flex;
  align-items: center;
  gap: 10px;
  position: sticky;
  top: 0;
  z-index: 10;
  cursor: pointer;
  user-select: none;
}
.lg-name {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 14px;
  letter-spacing: 2px;
  color: var(--accent);
}
.lg-count {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--muted2);
  background: var(--s3);
  padding: 2px 8px;
  border-radius: 4px;
  border: 1px solid var(--border);
}
.lg-stats {
  margin-left: auto;
  display: flex;
  gap: 16px;
  font-size: 11px;
  color: var(--muted2);
  font-family: 'DM Mono', monospace;
}
/* League group */
.lg-header {
  padding: 11px 20px;
  background: var(--s2);
  border-bottom: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  display: flex;
  align-items: center;
  gap: 12px;
  position: sticky;
  top: 0;
  z-index: 10;
  cursor: pointer;
  user-select: none;
}
.lg-header:hover { background: var(--s3); }
.lg-name {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 15px;
  letter-spacing: 2px;
  color: var(--text);
}
.lg-flag { font-size: 18px; line-height: 1; }
.lg-count {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--accent);
  background: rgba(0,200,240,0.1);
  padding: 2px 8px;
  border-radius: 4px;
  border: 1px solid rgba(0,200,240,0.2);
  font-weight: 600;
}
.lg-stats {
  margin-left: auto;
  display: flex;
  gap: 20px;
  font-size: 11px;
  color: var(--muted);
  font-family: 'DM Mono', monospace;
}
.lg-stats span { display: flex; align-items: center; gap: 4px; }
.lg-body { overflow: hidden; transition: max-height 0.3s ease; }
.lg-body.collapsed { max-height: 0 !important; }
.lg-table { width: 100%; border-collapse: collapse; }
.lg-table th {
  background: var(--s1);
  padding: 8px 12px;
  text-align: left;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--muted);
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
.lg-table td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
  color: var(--text);
  white-space: nowrap;
  vertical-align: middle;
}
.lg-table tr:hover td { background: rgba(0,200,240,0.03); cursor: pointer; }
.lg-table tr:last-child td { border-bottom: none; }

/* ── LIGHT MODE ── */
body.light-mode {
  --bg:     #f0f4f8;
  --s1:     #ffffff;
  --s2:     #f7f9fc;
  --s3:     #edf2f7;
  --s4:     #e2e8f0;
  --border: #d1dbe8;
  --border2:#b8c8da;
  --text:   #1a2535;
  --muted:  #4a5568;
  --muted2: #718096;
  --accent: #0891b2;
}
body.light-mode .topbar { background: #fff; border-color: #d1dbe8; }
body.light-mode thead th { background: #edf2f7; color: #4a5568; }
body.light-mode tbody tr:hover { background: rgba(8,145,178,0.04); }
body.light-mode .pill-summer { background: #e53e3e; }
body.light-mode .pill-monitor { background: #3182ce; }
body.light-mode .pill-january { background: #6b46c1; }
body.light-mode .pill-signed { background: #38a169; }
body.light-mode .pill-rejected { background: #718096; }
body.light-mode .tab.active { background: #fff; color: #0891b2; border-color: #d1dbe8; }
body.light-mode .tab { color: #718096; }
body.light-mode .tab:hover { background: #edf2f7; color: #1a2535; }
body.light-mode .modal { background: #fff; border-color: #d1dbe8; }
body.light-mode .form-input { background: #f7f9fc; border-color: #d1dbe8; color: #1a2535; }
body.light-mode .form-input:focus { border-color: #0891b2; }
body.light-mode .filter-sel { background: #fff; border-color: #d1dbe8; color: #1a2535; }
body.light-mode .overlay { background: rgba(0,0,0,0.4); }
body.light-mode .lg-header { background: #f7f9fc; border-left-color: #0891b2; }
body.light-mode .lg-header:hover { background: #edf2f7; }
body.light-mode .detail-stat { border-color: #d1dbe8; }
body.light-mode .val-green { color: #276749; }
body.light-mode .score-band { filter: brightness(1.1) saturate(0.9); }
/* ── MOBILE ────────────────────────────────────────────────────────────────── */
@media (max-width: 768px) {
  /* Show only essential buttons */
  #mobile-btn { display: inline-flex !important; }
  #theme-btn  { display: inline-flex !important; }
  button[onclick="exportCSV()"] { display: none !important; }
  button[onclick="syncAll()"]   { display: none !important; }
  button[onclick="openImport()"]{ display: none !important; }
  button[onclick="openAdd()"]   { display: none !important; }
  /* Topbar */
  .topbar { padding: 0 10px; gap: 6px; height: auto; min-height: 52px; }
  .topbar-spacer, .topbar-sep { display: none; }
  .logo-sub { display: none; }
  .logo-text { font-size: 13px; }
  /* Tabs */
  .tabs-bar { padding: 0 8px; overflow-x: auto; -webkit-overflow-scrolling: touch; flex-wrap: nowrap; }
  .tab { padding: 8px 10px; font-size: 11px; white-space: nowrap; flex-shrink: 0; }
  /* Toolbar */
  .toolbar { padding: 8px 10px; gap: 6px; flex-wrap: wrap; }
  .search-wrap { width: 100%; order: -1; }
  .search-input { width: 100%; }
  .filter-sel { font-size: 11px; max-width: 110px; }
  #extra-filter-btn { display: none !important; }
  #extra-filters { display: none !important; }
  /* Table */
  .table-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
  /* Detail modal — bottom sheet */
  .overlay { align-items: flex-end; padding: 0; }
  .detail-modal { width: 100% !important; max-width: 100% !important; max-height: 92vh; border-radius: 18px 18px 0 0; overflow-y: auto; }
  .modal { width: calc(100vw - 24px) !important; max-height: 88vh; overflow-y: auto; }
  /* Detail layout */
  .detail-hero { flex-direction: column; align-items: center; text-align: center; padding: 14px 16px; }
  .detail-meta { justify-content: center; flex-wrap: wrap; }
  .detail-tags { justify-content: center; flex-wrap: wrap; }
  .detail-stats-grid { grid-template-columns: repeat(3, 1fr); }
  /* Form */
  .form-grid { grid-template-columns: 1fr !important; }
  .form-full { grid-column: 1 !important; }
}
@media (max-width: 480px) {
  .detail-stats-grid { grid-template-columns: repeat(2, 1fr); }
  .filter-sel { max-width: 88px; font-size: 10px; }
}

/* Mobile view mode — activated manually on desktop too */
body.mobile-view {
  --mobile-pad: 12px;
}
body.mobile-view .table-scroll { display: none; }
body.mobile-view .lg-wrap { display: none; }
body.mobile-view .perf-wrap { display: none; }
body.mobile-view #mobile-cards { display: block; }

#mobile-cards {
  display: none;
  padding: 10px 12px;
  overflow-y: auto;
}
.mobile-card {
  background: var(--s2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px 14px;
  margin-bottom: 10px;
  cursor: pointer;
  transition: border-color 0.15s;
}
.mobile-card:hover { border-color: var(--accent); }
.mobile-card-top { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.mobile-card-photo { width: 42px; height: 42px; border-radius: 50%; overflow: hidden; background: var(--s3); display:flex; align-items:center; justify-content:center; font-size:18px; flex-shrink:0; }
.mobile-card-photo img { width:100%; height:100%; object-fit:cover; }
.mobile-card-name { font-size: 14px; font-weight: 700; color: var(--text); }
.mobile-card-sub { font-size: 11px; color: var(--muted2); margin-top: 1px; }
.mobile-card-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
.mobile-card-stats { display: grid; grid-template-columns: repeat(4,1fr); gap: 4px; margin-top: 8px; border-top: 1px solid var(--border); padding-top: 8px; }
.mobile-card-stat { text-align: center; }
.mobile-card-stat .v { font-size: 14px; font-weight: 700; color: var(--text); font-family: 'DM Mono', monospace; }
.mobile-card-stat .l { font-size: 9px; color: var(--muted2); text-transform: uppercase; letter-spacing: .04em; margin-top: 1px; }</style>
</head>
<body>

<!-- TOP BAR -->
<div class="topbar">
  <div class="logo">
    <div class="logo-mark">⚽</div>
    <div>
      <div class="logo-text">Database</div>
      <div class="logo-sub">Pro Edition</div>
    </div>
  </div>
  <div class="topbar-sep"></div>
  <div class="board-name" id="board-title">CB List</div>
  <div class="topbar-spacer"></div>
  <button class="btn btn-ghost" onclick="toggleLightMode()" id="theme-btn" style="font-size:11px">☀️ Light</button>
  <button class="btn btn-ghost" onclick="toggleMobileView()" id="mobile-btn" style="font-size:11px;display:none">📱 Mobile</button>
  <button class="btn btn-ghost" onclick="exportCSV()" style="font-size:11px">📊 Export CSV</button>
  <button class="btn btn-ghost" onclick="syncAll()" id="sync-all-btn" style="font-size:11px">🔄 Sync All TM</button>
  <button class="btn btn-ghost" onclick="openImport()" style="font-size:11px">📥 Import</button>
  <button class="btn btn-primary" onclick="openAdd()">＋ New Player</button>
</div>

<!-- TABS -->
<div class="tabs" id="tabs-bar">
  <!-- Generated dynamically -->
  <div class="tab-add" onclick="addTab()" title="New board">＋</div>
</div>

<!-- TOOLBAR -->
<div class="toolbar">
  <div class="search-wrap">
    <span class="search-icon">🔍</span>
    <input class="search-input" id="searchInput" placeholder="Search players, teams, leagues..." oninput="renderAll()">
  </div>
  <select class="filter-sel" id="windowFilter" onchange="renderAll()">
    <option value="">All Windows</option>
    <option value="Summer">Summer</option>
    <option value="Monitor">Monitor</option>
    <option value="January">January</option>
    <option value="Signed">Signed</option>
  </select>
  <select class="filter-sel" id="targetFilter" onchange="renderAll()">
    <option value="">All Targets</option>
    <option value="Premier League">Premier League</option>
    <option value="Championship">Championship</option>
    <option value="Europe Band 1">Europe Band 1</option>
    <option value="Europe Band 1-2">Europe Band 1-2</option>
    <option value="Europe Band 2-3">Europe Band 2-3</option>
    <option value="PL Monitor">PL Monitor</option>
    <option value="CH Monitor">CH Monitor</option>
    <option value="TBC">TBC</option>
  </select>
  <select class="filter-sel" id="statusFilter" onchange="renderAll()">
    <option value="">All Statuses</option>
    <option value="No Progress">No Progress</option>
    <option value="Relationship">Relationship</option>
    <option value="Agency Link">Agency Link</option>
    <option value="Contact Made">Contact Made</option>
  </select>
  <select class="filter-sel" id="agencyFilter" onchange="renderAll()">
    <option value="">All Agencies</option>
    <option value="Small">Small</option>
    <option value="Medium">Medium</option>
    <option value="Big">Big</option>
  </select>
  <button id="extra-filter-btn" onclick="toggleExtraFilters()" style="background:var(--s3);border:1px solid var(--border);border-radius:7px;padding:6px 11px;font-size:12px;font-weight:600;color:var(--muted);cursor:pointer;white-space:nowrap;flex-shrink:0">⊕ Filters</button>
  <div class="topbar-spacer"></div>
  <span class="record-count" id="rec-count">0 players</span>
  <div style="display:flex;background:var(--s3);border:1px solid var(--border);border-radius:7px;overflow:hidden;margin-left:8px">
    <button id="vbtn-table" class="view-toggle-btn active" onclick="setView('table')" title="Table view">≡ Table</button>
    <button id="vbtn-league" class="view-toggle-btn" onclick="setView('league')" title="Group by League">🌍 By League</button>
    <button id="vbtn-perf" class="view-toggle-btn" onclick="setView('perf')" title="Performance scores">📊 Performance</button>
  </div>
  <div style="position:relative;margin-left:8px">
    <button class="view-toggle-btn" onclick="toggleColPicker()" title="Show/hide columns" style="background:var(--s3);border:1px solid var(--border);border-radius:7px;padding:6px 12px;font-size:12px;font-weight:600;color:var(--muted);cursor:pointer;white-space:nowrap">⚙ Columns</button>
    <div id="col-picker" style="display:none;position:absolute;right:0;top:calc(100% + 6px);background:var(--s2);border:1px solid var(--border);border-radius:10px;padding:12px;z-index:500;min-width:220px;box-shadow:0 8px 32px rgba(0,0,0,0.4)">
      <div style="font-size:10px;font-weight:700;color:var(--muted2);letter-spacing:.08em;margin-bottom:8px">VISIBLE COLUMNS</div>
      <div id="col-picker-list" style="display:flex;flex-direction:column;gap:4px"></div>
      <div style="margin-top:10px;display:flex;gap:6px">
        <button onclick="colPickerAll(true)" style="flex:1;font-size:11px;padding:4px 0;background:var(--s3);border:1px solid var(--border);border-radius:5px;color:var(--muted);cursor:pointer">All</button>
        <button onclick="colPickerAll(false)" style="flex:1;font-size:11px;padding:4px 0;background:var(--s3);border:1px solid var(--border);border-radius:5px;color:var(--muted);cursor:pointer">None</button>
        <button onclick="colPickerReset()" style="flex:1;font-size:11px;padding:4px 0;background:var(--s3);border:1px solid var(--border);border-radius:5px;color:var(--muted);cursor:pointer">Reset</button>
      </div>
    </div>
  </div>
</div>

<!-- EXTRA FILTERS ROW -->
<div id="extra-filters" style="display:none;background:var(--s2);border-bottom:1px solid var(--border);padding:8px 20px;display:none;align-items:center;gap:10px;flex-wrap:wrap">
  <span style="font-size:10px;font-weight:700;color:var(--muted2);letter-spacing:.08em;margin-right:4px">FILTERS</span>

  <div style="display:flex;align-items:center;gap:6px">
    <label style="font-size:11px;color:var(--muted);white-space:nowrap">Foot</label>
    <select class="filter-sel" id="footFilter" onchange="renderAll()">
      <option value="">Any</option>
      <option value="Left">Left</option>
      <option value="Right">Right</option>
      <option value="Both">Both</option>
    </select>
  </div>

  <div style="display:flex;align-items:center;gap:6px">
    <label style="font-size:11px;color:var(--muted);white-space:nowrap">Style</label>
    <select class="filter-sel" id="styleFilter" onchange="renderAll()">
      <option value="">Any</option>
      <option value="Wide CB">Wide CB</option>
      <option value="Rounded CB">Rounded CB</option>
      <option value="Complete CB">Complete CB</option>
      <option value="Ball Playing CB">Ball Playing CB</option>
      <option value="Box Defender">Box Defender</option>
      <option value="Attacking FB">Attacking FB</option>
      <option value="Defensive FB">Defensive FB</option>
      <option value="Inverted FB">Inverted FB</option>
      <option value="Box-to-Box">Box-to-Box</option>
      <option value="Defensive CM">Defensive CM</option>
      <option value="Playmaker">Playmaker</option>
      <option value="Wide">Wide</option>
      <option value="Inside Forward">Inside Forward</option>
      <option value="Target Man">Target Man</option>
      <option value="Press &amp; Run">Press &amp; Run</option>
    </select>
  </div>

  <div style="display:flex;align-items:center;gap:6px">
    <label style="font-size:11px;color:var(--muted);white-space:nowrap">Value ≤</label>
    <select class="filter-sel" id="valueFilter" onchange="renderAll()">
      <option value="">Any</option>
      <option value="1">€1M</option>
      <option value="2">€2M</option>
      <option value="3">€3M</option>
      <option value="5">€5M</option>
      <option value="8">€8M</option>
      <option value="10">€10M</option>
      <option value="15">€15M</option>
      <option value="20">€20M</option>
      <option value="30">€30M</option>
      <option value="50">€50M</option>
    </select>
  </div>

  <div style="display:flex;align-items:center;gap:6px">
    <label style="font-size:11px;color:var(--muted);white-space:nowrap">Contract ≤</label>
    <select class="filter-sel" id="contractFilter" onchange="renderAll()">
      <option value="">Any</option>
      <option value="2025">2025</option>
      <option value="2026">2026</option>
      <option value="2027">2027</option>
      <option value="2028">2028</option>
      <option value="2029">2029</option>
    </select>
  </div>

  <div style="display:flex;align-items:center;gap:6px">
    <label style="font-size:11px;color:var(--muted);white-space:nowrap">GBE Band</label>
    <div style="position:relative">
      <button id="gbe-band-btn" onclick="toggleGBEBandPicker()" style="background:var(--s3);border:1px solid var(--border);border-radius:6px;padding:5px 10px;font-size:11px;color:var(--muted);cursor:pointer;white-space:nowrap;min-width:80px;text-align:left">All Bands ▾</button>
      <div id="gbe-band-picker" style="display:none;position:absolute;top:calc(100% + 4px);left:0;background:var(--s2);border:1px solid var(--border);border-radius:8px;padding:8px 10px;z-index:500;min-width:160px;box-shadow:0 8px 24px rgba(0,0,0,0.4)">
        <div style="font-size:10px;color:var(--muted2);margin-bottom:6px;font-weight:700;letter-spacing:.06em">GBE BAND (FA 2025/26)</div>
        <label style="display:flex;align-items:center;gap:7px;font-size:11px;color:var(--fg);cursor:pointer;padding:2px 0"><input type="checkbox" class="gbe-band-cb" value="1" checked onchange="applyGBEBandFilter()" style="accent-color:var(--accent)"> Band 1 – Big 5</label>
        <label style="display:flex;align-items:center;gap:7px;font-size:11px;color:var(--fg);cursor:pointer;padding:2px 0"><input type="checkbox" class="gbe-band-cb" value="2" checked onchange="applyGBEBandFilter()" style="accent-color:var(--accent)"> Band 2</label>
        <label style="display:flex;align-items:center;gap:7px;font-size:11px;color:var(--fg);cursor:pointer;padding:2px 0"><input type="checkbox" class="gbe-band-cb" value="3" checked onchange="applyGBEBandFilter()" style="accent-color:var(--accent)"> Band 3</label>
        <label style="display:flex;align-items:center;gap:7px;font-size:11px;color:var(--fg);cursor:pointer;padding:2px 0"><input type="checkbox" class="gbe-band-cb" value="4" checked onchange="applyGBEBandFilter()" style="accent-color:var(--accent)"> Band 4</label>
        <label style="display:flex;align-items:center;gap:7px;font-size:11px;color:var(--fg);cursor:pointer;padding:2px 0"><input type="checkbox" class="gbe-band-cb" value="5" checked onchange="applyGBEBandFilter()" style="accent-color:var(--accent)"> Band 5</label>
        <label style="display:flex;align-items:center;gap:7px;font-size:11px;color:var(--fg);cursor:pointer;padding:2px 0"><input type="checkbox" class="gbe-band-cb" value="6" checked onchange="applyGBEBandFilter()" style="accent-color:var(--accent)"> Band 6 / Other</label>
        <div style="margin-top:6px;display:flex;gap:6px">
          <button onclick="setAllGBEBands(true)" style="flex:1;font-size:10px;padding:3px 0;background:var(--s3);border:1px solid var(--border);border-radius:4px;color:var(--muted);cursor:pointer">All</button>
          <button onclick="setAllGBEBands(false)" style="flex:1;font-size:10px;padding:3px 0;background:var(--s3);border:1px solid var(--border);border-radius:4px;color:var(--muted);cursor:pointer">None</button>
        </div>
      </div>
    </div>
  </div>

  <button onclick="clearExtraFilters()" style="background:transparent;border:1px solid var(--border);border-radius:6px;padding:4px 10px;font-size:11px;color:var(--muted2);cursor:pointer;margin-left:4px">✕ Clear</button>
</div>

<!-- MAIN -->
<div class="main">
  <!-- MOBILE CARDS VIEW -->
  <div id="mobile-cards"></div>

  <!-- TABLE VIEW -->
  <div class="table-scroll" id="table-area">
    <table id="main-table">
      <thead id="thead"></thead>
      <tbody id="tbody"></tbody>
    </table>
    <div class="empty hidden" id="empty-state">
      <div class="empty-icon">📋</div>
      <div>No players found</div>
      <div style="font-size:11px;color:var(--muted2)">Add your first player or adjust filters</div>
    </div>
  </div>
  <!-- LEAGUE VIEW -->
  <div class="table-scroll hidden" id="league-area"></div>
  <div class="table-scroll hidden" id="perf-area" style="padding:20px"></div>
</div>

<!-- ADD / EDIT MODAL -->
<div class="overlay" id="add-overlay">
<div class="modal" id="add-modal">
  <div class="modal-head">
    <div class="modal-title" id="add-title">NEW PLAYER</div>
    <button class="modal-x" onclick="closeAdd()">✕</button>
  </div>
  <div class="modal-body">
    <div class="form-grid">
      <div class="form-group">
        <label class="form-label">Player Name *</label>
        <input class="form-input" id="f-name" placeholder="e.g. V. van Dijk">
      </div>
      <div class="form-group">
        <label class="form-label">Full Name</label>
        <input class="form-input" id="f-fullname" placeholder="e.g. Virgil van Dijk">
      </div>
      <div class="form-group">
        <label class="form-label">Team</label>
        <input class="form-input" id="f-team" placeholder="e.g. Liverpool">
      </div>
      <div class="form-group">
        <label class="form-label">League</label>
        <input class="form-input" id="f-league" placeholder="e.g. England 1.">
      </div>
      <div class="form-group">
        <label class="form-label">Age</label>
        <input class="form-input" id="f-age" type="number" placeholder="24">
      </div>
      <div class="form-group">
        <label class="form-label">Foot</label>
        <select class="form-select" id="f-foot">
          <option value="">—</option>
          <option value="Right">Right</option>
          <option value="Left">Left</option>
          <option value="Both">Both</option>
        </select>
      </div>
    </div>

    <div class="section-divider"><div class="section-divider-label">Scouting</div><div class="section-divider-line"></div></div>
    <div class="form-grid">
      <div class="form-group">
        <label class="form-label">Window</label>
        <select class="form-select" id="f-window">
          <option value="Monitor">Monitor</option>
          <option value="Summer">Summer</option>
          <option value="January">January</option>
          <option value="Signed">Signed</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Target Move</label>
        <select class="form-select" id="f-target">
          <option value="Premier League">Premier League</option>
          <option value="Championship">Championship</option>
          <option value="Europe Band 1">Europe Band 1</option>
          <option value="Europe Band 1-2">Europe Band 1-2</option>
          <option value="Europe Band 2-3">Europe Band 2-3</option>
          <option value="PL Monitor">PL Monitor</option>
          <option value="CH Monitor">CH Monitor</option>
          <option value="TBC">TBC</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Status</label>
        <select class="form-select" id="f-status">
          <option value="No Progress">No Progress</option>
          <option value="Relationship">Relationship</option>
          <option value="Agency Link">Agency Link</option>
          <option value="Contact Made">Contact Made</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Agency Size</label>
        <select class="form-select" id="f-agency">
          <option value="Small">Small</option>
          <option value="Medium">Medium</option>
          <option value="Big">Big</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Recent Move</label>
        <select class="form-select" id="f-recentmove">
          <option value="No">No</option>
          <option value="Yes">Yes</option>
          <option value="N/A">N/A</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Priority (1–5 ★)</label>
        <select class="form-select" id="f-priority">
          <option value="1">★ 1</option>
          <option value="2">★★ 2</option>
          <option value="3" selected>★★★ 3</option>
          <option value="4">★★★★ 4</option>
          <option value="5">★★★★★ 5</option>
        </select>
      </div>
    </div>

    <div class="section-divider"><div class="section-divider-label">Profile</div><div class="section-divider-line"></div></div>
    <div class="form-grid">
      <div class="form-group">
        <label class="form-label">Style</label>
        <select class="form-select" id="f-style">
          <option value="">—</option>
          <option value="Ball Playing CB">Ball Playing CB</option>
          <option value="Wide CB">Wide CB</option>
          <option value="Complete CB">Complete CB</option>
          <option value="Box Defender">Box Defender</option>
          <option value="Ball Playing GK">Ball Playing GK</option>
          <option value="Sweeper GK">Sweeper GK</option>
          <option value="Box-to-Box">Box-to-Box</option>
          <option value="Deep-Lying PM">Deep-Lying PM</option>
          <option value="Progressive FB">Progressive FB</option>
          <option value="Defensive FB">Defensive FB</option>
          <option value="Inside Forward">Inside Forward</option>
          <option value="Traditional Winger">Traditional Winger</option>
          <option value="Target Man">Target Man</option>
          <option value="Advanced Forward">Advanced Forward</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Primary Role</label>
        <select class="form-select" id="f-role1">
          <option value="">—</option>
          <option value="CB2">CB2</option>
          <option value="CB3">CB3</option>
          <option value="GK1">GK1</option>
          <option value="GK2">GK2</option>
          <option value="RB2">RB2</option>
          <option value="LB2">LB2</option>
          <option value="CM6">CM6</option>
          <option value="CM8">CM8</option>
          <option value="CM10">CM10</option>
          <option value="WG">WG</option>
          <option value="ST">ST</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Score* (Current)</label>
        <select class="form-select" id="f-score">
          <option value="">—</option>
          <option value="80-84">80-84</option>
          <option value="76-79">76-79</option>
          <option value="72-76">72-76</option>
          <option value="66-72">66-72</option>
          <option value="60-65">60-65</option>
          <option value="55-60">55-60</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Potential*</label>
        <select class="form-select" id="f-potential">
          <option value="">—</option>
          <option value="80-84">80-84</option>
          <option value="76-79">76-79</option>
          <option value="72-76">72-76</option>
          <option value="66-72">66-72</option>
          <option value="60-65">60-65</option>
          <option value="55-60">55-60</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Physical Notes</label>
        <select class="form-select" id="f-physical">
          <option value="">—</option>
          <option value="Athletic">Athletic</option>
          <option value="Physical">Physical</option>
          <option value="Athletic, Physical">Athletic + Physical</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Nationality</label>
        <input class="form-input" id="f-nationality" placeholder="e.g. Netherlands">
      </div>
    </div>

    <div class="section-divider"><div class="section-divider-label">Financial</div><div class="section-divider-line"></div></div>
    <div class="form-grid">
      <div class="form-group">
        <label class="form-label">Market Value (€)</label>
        <input class="form-input" id="f-value" placeholder="e.g. 5000000">
      </div>
      <div class="form-group">
        <label class="form-label">Contract Expires</label>
        <input class="form-input" id="f-contract" placeholder="e.g. 2027">
      </div>
    </div>

    <div class="section-divider"><div class="section-divider-label">Stats</div><div class="section-divider-line"></div></div>
    <div class="form-grid-3">
      <div class="form-group">
        <label class="form-label">Games</label>
        <input class="form-input" id="f-games" type="number" placeholder="0">
      </div>
      <div class="form-group">
        <label class="form-label">Goals</label>
        <input class="form-input" id="f-goals" type="number" placeholder="0">
      </div>
      <div class="form-group">
        <label class="form-label">Assists</label>
        <input class="form-input" id="f-assists" type="number" placeholder="0">
      </div>
      <div class="form-group">
        <label class="form-label">Minutes</label>
        <input class="form-input" id="f-minutes" type="number" placeholder="0">
      </div>
      <div class="form-group">
        <label class="form-label">Goals Conceded</label>
        <input class="form-input" id="f-ga" type="number" placeholder="0">
      </div>
    </div>

    <div class="section-divider"><div class="section-divider-label">Links & Notes</div><div class="section-divider-line"></div></div>
    <div class="form-grid">
      <div class="form-group">
        <label class="form-label">Transfermarkt Link</label>
        <input class="form-input" id="f-tm" placeholder="https://www.transfermarkt.com/...">
      </div>
      <div class="form-group">
        <label class="form-label">Video Link</label>
        <input class="form-input" id="f-video" placeholder="https://wyscout.com/...">
      </div>
      <div class="form-group">
        <label class="form-label">Photo URL</label>
        <input class="form-input" id="f-photo" placeholder="https://images.fotmob.com/...">
      </div>
    </div>

    <div class="form-group form-full" style="margin-top:4px">
      <label class="form-label">Data Profile (PNG)</label>
      <div style="display:flex;flex-direction:column;gap:6px">
        <input type="file" id="f-dataprofile-file" accept="image/png,image/jpeg,image/webp" style="display:none" onchange="loadDataProfileFile(this)">
        <div style="display:flex;gap:8px;align-items:center">
          <button type="button" onclick="document.getElementById('f-dataprofile-file').click()" style="background:var(--s3);border:1px solid var(--border);border-radius:6px;padding:7px 14px;font-size:11px;color:var(--muted);cursor:pointer;white-space:nowrap">📎 Upload PNG…</button>
          <span id="f-dataprofile-name" style="font-size:11px;color:var(--muted2);font-style:italic">No file attached</span>
        </div>
        <div id="f-dataprofile-preview" style="display:none;position:relative">
          <img id="f-dataprofile-img" style="width:100%;border-radius:6px;border:1px solid var(--border);max-height:160px;object-fit:contain;background:var(--s2)">
          <button type="button" onclick="clearDataProfile()" style="position:absolute;top:4px;right:4px;background:rgba(0,0,0,0.7);border:none;border-radius:50%;width:22px;height:22px;font-size:12px;color:#fff;cursor:pointer;line-height:22px;text-align:center">✕</button>
        </div>
        <input type="hidden" id="f-dataprofile">
      </div>
    </div>

    <div class="form-grid">
      <div class="form-group">
        <label class="form-label">Position</label>
        <select class="form-select" id="f-pos">
          <option value="CB">CB</option><option value="LCB">LCB</option><option value="RCB">RCB</option>
          <option value="GK">GK</option><option value="LB">LB</option><option value="RB">RB</option>
          <option value="LWB">LWB</option><option value="RWB">RWB</option><option value="DMF">DMF</option>
          <option value="CMF">CMF</option><option value="AMF">AMF</option><option value="LW">LW</option>
          <option value="RW">RW</option><option value="CF">CF</option><option value="SS">SS</option>
        </select>
      </div>
    </div>
    <div class="form-group form-full" style="margin-top:4px">
      <label class="form-label">Extra Key Notes</label>
      <textarea class="form-textarea" id="f-keynotes" placeholder="e.g. Same agent as..., New contract likely..."></textarea>
    </div>
    <div class="form-group form-full">
      <label class="form-label">Scout Note (added to history)</label>
      <textarea class="form-textarea" id="f-note" placeholder="Initial scouting observation..."></textarea>
    </div>
  </div>
  <div class="modal-foot">
    <button class="btn btn-ghost btn-danger-soft" id="delete-btn" onclick="deleteEditing()" style="display:none;color:#f87171;border-color:rgba(248,113,113,0.3)">🗑 Delete</button>
    <div style="display:flex;gap:8px;margin-left:auto">
      <button class="btn btn-ghost" onclick="closeAdd()">Cancel</button>
      <button class="btn btn-primary" onclick="savePlayer()">Save Player</button>
    </div>
  </div>
</div>
</div>

<!-- DETAIL OVERLAY -->
<div class="overlay" id="detail-overlay">
<div class="modal detail-modal" id="detail-modal">
  <div class="modal-head">
    <div class="modal-title" id="detail-title">PLAYER PROFILE</div>
    <div style="display:flex;gap:8px;align-items:center">
      <button class="btn btn-ghost btn-sm" onclick="editFromDetail()">✏ Edit</button>
      <button class="modal-x" onclick="closeDetail()">✕</button>
    </div>
  </div>
  <div id="detail-body"></div>
</div>
</div>

<!-- TOAST -->
<div class="toast" id="toast"></div>

<script>
// ── DATA ──
const BOARDS_KEY = 'scoutpro_boards';
const DATA_KEY = 'scoutpro_data';
const CORRECT_BOARDS = ["GK","CB","RB","LB","CM6","CM8","CM10","WNG","ST"];

let boards = CORRECT_BOARDS.slice();
let currentView = 'table';
let allPlayers = {};
let currentBoard = boards[0] || 'GK';
let editingId = null;
let viewingId = null;
let sortKey = 'priority';
let sortAsc = false;

// Ensure each board has an array
CORRECT_BOARDS.forEach(b => { if (!allPlayers[b]) allPlayers[b] = []; });

async function save() {
  // Save to server if available, always save to localStorage as backup
  localStorage.setItem(DATA_KEY, JSON.stringify(allPlayers));
  if (serverOnline) {
    try {
      await fetch(SERVER + '/api/boards', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(allPlayers)
      });
    } catch(e) { console.warn('Server save failed, localStorage only'); }
  }
}

async function loadFromServer() {
  try {
    const r = await fetch(SERVER + '/api/boards', {signal: AbortSignal.timeout(5000)});
    if (r.ok) {
      const data = await r.json();
      if (data && typeof data === 'object') {
        CORRECT_BOARDS.forEach(b => { allPlayers[b] = data[b] || []; });
        serverOnline = true;
        return true;
      }
    }
  } catch(e) {}
  serverOnline = false;
  return false;
}

const JS_GBE_BAND = {
  'England 1.':1,'Germany 1.':1,'Spain 1.':1,'Italy 1.':1,'France 1.':1,
  'Portugal 1.':2,'Netherlands 1.':2,'Belgium 1.':2,'Turkey 1.':2,'England 2.':2,
  'USA 1.':3,'Brazil 1.':3,'Argentina 1.':3,'Mexico 1.':3,'Scotland 1.':3,
  'Czech 1.':4,'Czech Republic 1.':4,'Croatia 1.':4,'Switzerland 1.':4,'Spain 2.':4,'Germany 2.':4,
  'Ukraine 1.':4,'Greece 1.':4,'Colombia 1.':4,'Austria 1.':4,'Denmark 1.':4,
  'France 2.':4,'Russia 1.':4,
  'Serbia 1.':5,'Poland 1.':5,'Slovenia 1.':5,'Chile 1.':5,'Uruguay 1.':5,
  'Sweden 1.':5,'Norway 1.':5,'Italy 2.':5,'Hungary 1.':5,'Japan 1.':5,
  'Korea 1.':5,'Australia 1.':5,'England 3.':5,
};

const PHOTO_BASE_URL = 'https://raw.githubusercontent.com/Matthewduffy23/scouting-photos/main/photos/';

function uid() { return Date.now().toString(36) + Math.random().toString(36).slice(2); }
function players() { return allPlayers[currentBoard] || []; }

function normPhoto(s) {
  if (!s) return '';
  return s
    .normalize('NFD')                        // decompose accents: ň → n + ̌
    .replace(/[\u0300-\u036f]/g, '')         // strip combining diacritics
    .replace(/[łŁ]/g, 'l')                  // ł → l (Polish)
    .replace(/[øØ]/g, 'o')                  // ø → o (Nordic)
    .replace(/[ðÐ]/g, 'd')                  // ð → d (Icelandic)
    .replace(/[þÞ]/g, 'th')                 // þ → th
    .replace(/[ß]/g, 'ss')                  // ß → ss
    .replace(/[æÆ]/g, 'ae')                 // æ → ae
    .replace(/[œŒ]/g, 'oe')                 // œ → oe
    .replace(/[^a-z0-9 ]/gi, '')            // strip anything else non-ASCII
    .toLowerCase()
    .trim()
    .replace(/\s+/g, '_');
}

function getPhotoUrls(name, team) {
  const pn = normPhoto(name);
  if (!pn) return [];

  const urls = new Set();

  function addUrl(t) {
    const norm = normPhoto(t);
    if (norm) urls.add(`${PHOTO_BASE_URL}${pn}__${norm}.png`);
  }

  const t = (team || '').trim();
  const tl = t.toLowerCase();

  // 1. Full team name as-is
  addUrl(t);

  // Words in the team name
  const words = t.split(/[\s\-]+/).filter(w => w.length > 1);

  // Common club keywords — if present, try using just them
  const keywords = ['fc','afc','sc','sk','bk','fk','nk','hnk','ac','as','us','ss',
                    'cf','cd','ud','rc','rb','psv','vfl','vfb','tsv','sv','bv',
                    'bsc','msv','fsv','pfc','sporting','athletic','atletico',
                    'united','city','real','stade','club','dynamo','lokomotiv',
                    'shakhtar','besiktas','fenerbahce','galatasaray'];

  // 2. Strip leading keyword prefix (e.g. "FC Midtjylland" → "Midtjylland")
  if (words.length >= 2 && keywords.includes(words[0].toLowerCase())) {
    const withoutFirst = words.slice(1).join(' ');
    addUrl(withoutFirst);
    // Also try stripping second word if it's also a keyword
    if (words.length >= 3 && keywords.includes(words[1].toLowerCase())) {
      addUrl(words.slice(2).join(' '));
    }
  }

  // 3. Strip trailing keyword suffix (e.g. "Borussia Dortmund FC" → "Borussia Dortmund")
  if (words.length >= 2 && keywords.includes(words[words.length-1].toLowerCase())) {
    addUrl(words.slice(0, -1).join(' '));
  }

  // 4. Try every word individually if 2-word team (e.g. "Khor Fakkan" → "fakkan")
  if (words.length === 2) {
    words.forEach(w => addUrl(w));
  }

  // 5. Strip hyphens → spaces
  if (tl.includes('-')) {
    addUrl(t.replace(/-/g, ' '));
  }

  // 6. Strip dots and apostrophes
  const nodots = t.replace(/['.]/g, '').replace(/\s+/g, ' ').trim();
  if (nodots !== t) addUrl(nodots);

  // 7. Numbers: "1. FC Köln" → try "koln", "fc koln", "1 fc koln"
  const nonum = t.replace(/^\d+\.\s*/, '').trim();
  if (nonum !== t) {
    addUrl(nonum);
    if (words.length >= 3 && keywords.includes(words[1]?.toLowerCase())) {
      addUrl(t.replace(/^\d+\.\s*/, '').replace(/^[a-z]+\s+/i, '').trim());
    }
  }

  return [...urls].filter((v,i,a) => a.indexOf(v) === i);
}

// Photo URL cache — avoids re-fetching on re-renders
const _photoCache = {};

function loadTablePhotos(list) {
  if (!list.length) return;

  // Apply cache immediately
  list.filter(p => _photoCache[p.id]).forEach(p => {
    const el = document.getElementById(`avatar-${p.id}`);
    if (el && el.tagName !== 'IMG') replaceWithPhoto(el, _photoCache[p.id]);
  });

  const uncached = list.filter(p => !_photoCache[p.id]);
  if (!uncached.length) return;

  if (serverOnline) {
    fetch(`${SERVER}/api/photos/batch`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(uncached.map(p => ({id:p.id, name:p.name||'', team:p.team||'', league:p.league||''})))
    })
    .then(r => r.json())
    .then(urlMap => {
      // Try server-returned URLs
      Object.entries(urlMap).forEach(([pid, urls]) => {
        if (urls?.length) tryPhotoUrls(urls, pid, `avatar-${pid}`);
      });
      // After a delay, fall back to direct for anything still not loaded
      setTimeout(() => {
        const stillMissing = uncached.filter(p => !_photoCache[p.id]);
        if (stillMissing.length) loadTablePhotosDirect(stillMissing, false);
      }, 1500);
    })
    .catch(() => loadTablePhotosDirect(uncached, false));
  } else {
    loadTablePhotosDirect(uncached, false);
  }
}

function loadLeaguePhotos(list) {
  if (!list.length) return;

  list.filter(p => _photoCache[p.id]).forEach(p => {
    const el = document.getElementById(`avatar-lg-${p.id}`);
    if (el && el.tagName !== 'IMG') replaceWithPhoto(el, _photoCache[p.id]);
  });

  const uncached = list.filter(p => !_photoCache[p.id]);
  if (!uncached.length) return;

  if (serverOnline) {
    fetch(`${SERVER}/api/photos/batch`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(uncached.map(p => ({id:p.id, name:p.name||'', team:p.team||'', league:p.league||''})))
    })
    .then(r => r.json())
    .then(urlMap => {
      Object.entries(urlMap).forEach(([pid, urls]) => {
        if (urls?.length) tryPhotoUrls(urls, pid, `avatar-lg-${pid}`);
      });
      setTimeout(() => {
        const stillMissing = uncached.filter(p => !_photoCache[p.id]);
        if (stillMissing.length) loadTablePhotosDirect(stillMissing, true);
      }, 1500);
    })
    .catch(() => loadTablePhotosDirect(uncached, true));
  } else {
    loadTablePhotosDirect(uncached, true);
  }
}

function tryPhotoUrls(urls, pid, elementId) {
  let idx = 0;
  function next() {
    if (idx >= urls.length) return;
    const url = urls[idx++];
    const img = new Image();
    img.onload = () => {
      _photoCache[pid] = url;
      const el = document.getElementById(elementId);
      if (el && el.tagName !== 'IMG') replaceWithPhoto(el, url);
    };
    img.onerror = next;
    img.src = url;
  }
  next();
}

function replaceWithPhoto(el, url) {
  const imgEl = document.createElement('img');
  imgEl.src = url;
  imgEl.className = 'player-avatar';
  imgEl.style.cssText = 'width:28px;height:28px;border-radius:50%;object-fit:cover;flex-shrink:0';
  el.replaceWith(imgEl);
}

function loadTablePhotosDirect(list, league) {
  // Fallback: direct URL guess in batches
  const BATCH = 10;
  let i = 0;
  const prefix = league ? 'avatar-lg-' : 'avatar-';
  function nextBatch() {
    const batch = list.slice(i, i + BATCH);
    if (!batch.length) return;
    i += BATCH;
    batch.forEach(p => {
      if (_photoCache[p.id]) {
        const el = document.getElementById(`${prefix}${p.id}`);
        if (el && el.tagName !== 'IMG') replaceWithPhoto(el, _photoCache[p.id]);
        return;
      }
      tryPhotoUrls(getPhotoUrls(p.name, p.team), p.id, `${prefix}${p.id}`);
    });
    setTimeout(nextBatch, 80);
  }
  nextBatch();
}

function loadPhoto(name, team, elementId, league) {
  function tryUrls(urls) {
    let idx = 0;
    function next() {
      if (idx >= urls.length) return;
      const url = urls[idx++];
      const img = new Image();
      img.onload = () => {
        const el = document.getElementById(elementId);
        if (el) el.innerHTML = `<img class="detail-photo" src="${url}">`;
      };
      img.onerror = next;
      img.src = url;
    }
    next();
  }
  // Use server for best URL match, fall back to local
  if (serverOnline) {
    fetch(`${SERVER}/api/player/photo?player=${encodeURIComponent(name||'')}&team=${encodeURIComponent(team||'')}&league=${encodeURIComponent(league||'')}`)
      .then(r => r.json())
      .then(data => tryUrls(data.urls || getPhotoUrls(name, team)))
      .catch(() => tryUrls(getPhotoUrls(name, team)));
  } else {
    tryUrls(getPhotoUrls(name, team));
  }
}

function toggleMobileView() {
  const isMobile = document.body.classList.toggle('mobile-view');
  const btn = document.getElementById('mobile-btn');
  if (btn) btn.textContent = isMobile ? '🖥 Desktop' : '📱 Mobile';
  localStorage.setItem('scoutpro_mobile', isMobile ? '1' : '0');
  if (isMobile) renderMobileCards();
}

function renderMobileCards() {
  const container = document.getElementById('mobile-cards');
  if (!container) return;
  const list = getFiltered();
  if (!list.length) {
    container.innerHTML = '<div class="empty"><div class="empty-icon">📋</div><div>No players found</div></div>';
    return;
  }
  container.innerHTML = list.map(p => {
    const photoId = `mc-${p.id}`;
    return `<div class="mobile-card" onclick="openDetail('${p.id}')">
      <div class="mobile-card-top">
        <div class="mobile-card-photo" id="${photoId}">👤</div>
        <div style="flex:1;min-width:0">
          <div class="mobile-card-name">${p.name||'—'}</div>
          <div class="mobile-card-sub">${p.team||'—'} · ${p.league||'—'}</div>
        </div>
        <div style="text-align:right;flex-shrink:0">
          ${p.score ? `<span class="score-band ${scoreCls(p.score)}" style="font-size:12px">${p.score}</span>` : ''}
          <div style="font-size:10px;color:var(--muted2);margin-top:2px">${p.age ? p.age+'y' : ''}</div>
        </div>
      </div>
      <div class="mobile-card-chips">
        ${p.window ? `<span class="${windowCls(p.window)}" style="font-size:10px">${p.window}</span>` : ''}
        ${p.target ? `<span class="${targetCls(p.target)}" style="font-size:10px">${p.target}</span>` : ''}
        ${p.status ? `<span class="${statusCls(p.status)}" style="font-size:10px">${p.status}</span>` : ''}
        ${p.priority ? `<span style="font-size:12px">${stars(p.priority)}</span>` : ''}
      </div>
      <div class="mobile-card-stats">
        <div class="mobile-card-stat"><div class="v">${fmtMV(p.marketValue)}</div><div class="l">Value</div></div>
        <div class="mobile-card-stat"><div class="v">${p.games||'—'}</div><div class="l">Games</div></div>
        <div class="mobile-card-stat"><div class="v">${p.goals||'—'}</div><div class="l">Goals</div></div>
        <div class="mobile-card-stat"><div class="v">${p.minutes||'—'}</div><div class="l">Mins</div></div>
      </div>
    </div>`;
  }).join('');
  // Load photos
  setTimeout(() => list.forEach(p => {
    tryPhotoUrls(getPhotoUrls(p.name, p.team), p.id, `mc-${p.id}`);
  }), 50);
}

// Apply saved mobile view on load
if (localStorage.getItem('scoutpro_mobile') === '1') {
  document.body.classList.add('mobile-view');
  const btn = document.getElementById('mobile-btn');
  if (btn) btn.textContent = '🖥 Desktop';
}

function toggleLightMode() {
  document.body.classList.toggle('light-mode');
  const isLight = document.body.classList.contains('light-mode');
  document.getElementById('theme-btn').textContent = isLight ? '🌙 Dark' : '☀️ Light';
  localStorage.setItem('scoutpro_theme', isLight ? 'light' : 'dark');
}

// Apply saved theme on load — default is LIGHT
const savedTheme = localStorage.getItem('scoutpro_theme');
if (savedTheme === 'dark') {
  document.body.classList.remove('light-mode');
  document.getElementById('theme-btn').textContent = '☀️ Light';
} else {
  // Default to light
  document.body.classList.add('light-mode');
  document.getElementById('theme-btn').textContent = '🌙 Dark';
}

const LEAGUE_FLAGS = {
  'England':'ENG','Scotland':'SCO','Wales':'WAL','Spain':'ESP','Germany':'GER',
  'Italy':'ITA','France':'FRA','Portugal':'POR','Netherlands':'NED','Belgium':'BEL',
  'Turkey':'TUR','Brazil':'BRA','Argentina':'ARG','USA':'USA','Mexico':'MEX',
  'Japan':'JPN','Korea':'KOR','Denmark':'DEN','Sweden':'SWE','Norway':'NOR',
  'Switzerland':'SUI','Austria':'AUT','Poland':'POL','Czech':'CZE','Croatia':'CRO',
  'Serbia':'SRB','Greece':'GRE','Ukraine':'UKR','Russia':'RUS','Romania':'ROU',
  'Hungary':'HUN','Bulgaria':'BUL','Slovakia':'SVK','Slovenia':'SVN','Israel':'ISR',
  'Ireland':'IRL','Northern Ireland':'NIR','Saudi':'KSA','UAE':'UAE','Qatar':'QAT',
  'Morocco':'MAR','Algeria':'ALG','Egypt':'EGY','South Africa':'RSA','Nigeria':'NGA',
  'Colombia':'COL','Chile':'CHI','Uruguay':'URU','Ecuador':'ECU','Peru':'PER',
  'Bolivia':'BOL','Paraguay':'PAR','Venezuela':'VEN','Costa Rica':'CRC','Australia':'AUS',
  'China':'CHN','Canada':'CAN','Finland':'FIN','Bosnia':'BIH','Albania':'ALB',
  'Kosovo':'KOS','Montenegro':'MNE','North Macedonia':'MKD','Moldova':'MDA',
  'Georgia':'GEO','Armenia':'ARM','Kazakhstan':'KAZ','Faroe Islands':'FRO',
  'Estonia':'EST','Latvia':'LAT','Lithuania':'LTU','Malta':'MLT','Cyprus':'CYP',
  'Iceland':'ISL','Tunisia':'TUN','Azerbaijan':'AZE','Belarus':'BLR',
  'United Arab Emirates':'UAE','Panama':'PAN','Luxembourg':'LUX',
};

// Real flag image URLs via flagcdn
const FLAG_URLS = {
  'England':'https://flagcdn.com/w20/gb-eng.png',
  'Scotland':'https://flagcdn.com/w20/gb-sct.png',
  'Wales':'https://flagcdn.com/w20/gb-wls.png',
  'Northern Ireland':'https://flagcdn.com/w20/gb-nir.png',
  'Spain':'https://flagcdn.com/w20/es.png',
  'Germany':'https://flagcdn.com/w20/de.png',
  'Italy':'https://flagcdn.com/w20/it.png',
  'France':'https://flagcdn.com/w20/fr.png',
  'Portugal':'https://flagcdn.com/w20/pt.png',
  'Netherlands':'https://flagcdn.com/w20/nl.png',
  'Belgium':'https://flagcdn.com/w20/be.png',
  'Turkey':'https://flagcdn.com/w20/tr.png',
  'Brazil':'https://flagcdn.com/w20/br.png',
  'Argentina':'https://flagcdn.com/w20/ar.png',
  'USA':'https://flagcdn.com/w20/us.png',
  'Mexico':'https://flagcdn.com/w20/mx.png',
  'Japan':'https://flagcdn.com/w20/jp.png',
  'Korea':'https://flagcdn.com/w20/kr.png',
  'Denmark':'https://flagcdn.com/w20/dk.png',
  'Sweden':'https://flagcdn.com/w20/se.png',
  'Norway':'https://flagcdn.com/w20/no.png',
  'Switzerland':'https://flagcdn.com/w20/ch.png',
  'Austria':'https://flagcdn.com/w20/at.png',
  'Poland':'https://flagcdn.com/w20/pl.png',
  'Czech':'https://flagcdn.com/w20/cz.png',
  'Croatia':'https://flagcdn.com/w20/hr.png',
  'Serbia':'https://flagcdn.com/w20/rs.png',
  'Greece':'https://flagcdn.com/w20/gr.png',
  'Ukraine':'https://flagcdn.com/w20/ua.png',
  'Russia':'https://flagcdn.com/w20/ru.png',
  'Romania':'https://flagcdn.com/w20/ro.png',
  'Hungary':'https://flagcdn.com/w20/hu.png',
  'Bulgaria':'https://flagcdn.com/w20/bg.png',
  'Slovakia':'https://flagcdn.com/w20/sk.png',
  'Slovenia':'https://flagcdn.com/w20/si.png',
  'Israel':'https://flagcdn.com/w20/il.png',
  'Ireland':'https://flagcdn.com/w20/ie.png',
  'Saudi':'https://flagcdn.com/w20/sa.png',
  'Morocco':'https://flagcdn.com/w20/ma.png',
  'Algeria':'https://flagcdn.com/w20/dz.png',
  'Egypt':'https://flagcdn.com/w20/eg.png',
  'South Africa':'https://flagcdn.com/w20/za.png',
  'Nigeria':'https://flagcdn.com/w20/ng.png',
  'Colombia':'https://flagcdn.com/w20/co.png',
  'Chile':'https://flagcdn.com/w20/cl.png',
  'Uruguay':'https://flagcdn.com/w20/uy.png',
  'Ecuador':'https://flagcdn.com/w20/ec.png',
  'Peru':'https://flagcdn.com/w20/pe.png',
  'Bolivia':'https://flagcdn.com/w20/bo.png',
  'Paraguay':'https://flagcdn.com/w20/py.png',
  'Venezuela':'https://flagcdn.com/w20/ve.png',
  'Costa Rica':'https://flagcdn.com/w20/cr.png',
  'Australia':'https://flagcdn.com/w20/au.png',
  'China':'https://flagcdn.com/w20/cn.png',
  'Canada':'https://flagcdn.com/w20/ca.png',
  'Finland':'https://flagcdn.com/w20/fi.png',
  'Bosnia':'https://flagcdn.com/w20/ba.png',
  'Albania':'https://flagcdn.com/w20/al.png',
  'Kosovo':'https://flagcdn.com/w20/xk.png',
  'Montenegro':'https://flagcdn.com/w20/me.png',
  'North Macedonia':'https://flagcdn.com/w20/mk.png',
  'Moldova':'https://flagcdn.com/w20/md.png',
  'Georgia':'https://flagcdn.com/w20/ge.png',
  'Armenia':'https://flagcdn.com/w20/am.png',
  'Kazakhstan':'https://flagcdn.com/w20/kz.png',
  'Faroe Islands':'https://flagcdn.com/w20/fo.png',
  'Estonia':'https://flagcdn.com/w20/ee.png',
  'Latvia':'https://flagcdn.com/w20/lv.png',
  'Lithuania':'https://flagcdn.com/w20/lt.png',
  'Malta':'https://flagcdn.com/w20/mt.png',
  'Cyprus':'https://flagcdn.com/w20/cy.png',
  'Iceland':'https://flagcdn.com/w20/is.png',
  'Tunisia':'https://flagcdn.com/w20/tn.png',
  'Azerbaijan':'https://flagcdn.com/w20/az.png',
  'Belarus':'https://flagcdn.com/w20/by.png',
  'Qatar':'https://flagcdn.com/w20/qa.png',
  'UAE':'https://flagcdn.com/w20/ae.png',
  'Panama':'https://flagcdn.com/w20/pa.png',
  'Luxembourg':'https://flagcdn.com/w20/lu.png',
};

// FotMob league logo URLs
const LEAGUE_LOGOS = {
  'England 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/47.png',
  'England 2.':'https://images.fotmob.com/image_resources/logo/leaguelogo/48.png',
  'England 3.':'https://images.fotmob.com/image_resources/logo/leaguelogo/108.png',
  'England 4.':'https://images.fotmob.com/image_resources/logo/leaguelogo/109.png',
  'Spain 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/87.png',
  'Spain 2.':'https://images.fotmob.com/image_resources/logo/leaguelogo/140.png',
  'Germany 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/54.png',
  'Germany 2.':'https://images.fotmob.com/image_resources/logo/leaguelogo/146.png',
  'Germany 3.':'https://images.fotmob.com/image_resources/logo/leaguelogo/208.png',
  'Italy 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/55.png',
  'Italy 2.':'https://images.fotmob.com/image_resources/logo/leaguelogo/86.png',
  'France 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/53.png',
  'France 2.':'https://images.fotmob.com/image_resources/logo/leaguelogo/110.png',
  'Portugal 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/61.png',
  'Portugal 2.':'https://images.fotmob.com/image_resources/logo/leaguelogo/185.png',
  'Netherlands 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/57.png',
  'Netherlands 2.':'https://images.fotmob.com/image_resources/logo/leaguelogo/111.png',
  'Belgium 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/40.png',
  'Turkey 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/71.png',
  'Turkey 2.':'https://images.fotmob.com/image_resources/logo/leaguelogo/165.png',
  'Scotland 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/64.png',
  'Scotland 2.':'https://images.fotmob.com/image_resources/logo/leaguelogo/123.png',
  'Brazil 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/268.png',
  'Brazil 2.':'https://images.fotmob.com/image_resources/logo/leaguelogo/8814.png',
  'Argentina 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/112.png',
  'USA 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/130.png',
  'Mexico 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/230.png',
  'Denmark 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/46.png',
  'Denmark 2.':'https://images.fotmob.com/image_resources/logo/leaguelogo/85.png',
  'Sweden 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/67.png',
  'Norway 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/59.png',
  'Switzerland 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/69.png',
  'Austria 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/38.png',
  'Poland 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/196.png',
  'Czech 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/122.png',
  'Croatia 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/252.png',
  'Serbia 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/182.png',
  'Greece 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/135.png',
  'Ukraine 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/441.png',
  'Russia 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/63.png',
  'Romania 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/189.png',
  'Japan 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/223.png',
  'Korea 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/9080.png',
  'Saudi 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/536.png',
  'Morocco 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/530.png',
  'Ireland 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/185.png',
  'Colombia 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/274.png',
  'Chile 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/273.png',
  'Hungary 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/212.png',
  'Israel 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/127.png',
  'Finland 1.':'https://images.fotmob.com/image_resources/logo/leaguelogo/51.png',
  'Norway 2.':'https://images.fotmob.com/image_resources/logo/leaguelogo/203.png',
  'Sweden 2.':'https://images.fotmob.com/image_resources/logo/leaguelogo/168.png',
};

// Regional colour coding for league headers
function leagueRegionColor(league) {
  if (!league) return 'var(--accent)';
  const l = league.toLowerCase();
  if (l.startsWith('england') || l.startsWith('scotland') || l.startsWith('wales') || l.startsWith('northern ireland') || l.startsWith('ireland')) return '#3b82f6';
  if (l.startsWith('spain') || l.startsWith('portugal')) return '#f59e0b';
  if (l.startsWith('germany') || l.startsWith('austria') || l.startsWith('switzerland')) return '#ef4444';
  if (l.startsWith('italy')) return '#10b981';
  if (l.startsWith('france')) return '#8b5cf6';
  if (l.startsWith('netherlands') || l.startsWith('belgium')) return '#f97316';
  if (l.startsWith('turkey')) return '#dc2626';
  if (l.startsWith('brazil') || l.startsWith('argentina') || l.startsWith('colombia') || l.startsWith('chile') || l.startsWith('uruguay') || l.startsWith('ecuador') || l.startsWith('peru') || l.startsWith('bolivia') || l.startsWith('paraguay') || l.startsWith('venezuela')) return '#22c55e';
  if (l.startsWith('usa') || l.startsWith('mexico') || l.startsWith('canada') || l.startsWith('costa rica')) return '#06b6d4';
  if (l.startsWith('japan') || l.startsWith('korea') || l.startsWith('china') || l.startsWith('australia')) return '#ec4899';
  if (l.startsWith('saudi') || l.startsWith('qatar') || l.startsWith('uae') || l.startsWith('morocco') || l.startsWith('egypt') || l.startsWith('algeria') || l.startsWith('nigeria') || l.startsWith('south africa') || l.startsWith('tunisia')) return '#d97706';
  return '#94a3b8';
}

function countryFlagHtml(country) {
  const CC = {
    'england':'gb-eng','scotland':'gb-sct','wales':'gb-wls','northern ireland':'gb-nir',
    'ireland':'ie','republic of ireland':'ie','spain':'es','germany':'de','italy':'it',
    'france':'fr','belgium':'be','denmark':'dk','poland':'pl','turkey':'tr',
    'netherlands':'nl','croatia':'hr','switzerland':'ch','norway':'no','sweden':'se',
    'czech republic':'cz','czech':'cz','greece':'gr','austria':'at','romania':'ro',
    'serbia':'rs','portugal':'pt','hungary':'hu','ukraine':'ua','slovakia':'sk',
    'slovenia':'si','bulgaria':'bg','finland':'fi','albania':'al','armenia':'am',
    'georgia':'ge','iceland':'is','north macedonia':'mk','lithuania':'lt','latvia':'lv',
    'estonia':'ee','montenegro':'me','moldova':'md','kosovo':'xk','bosnia':'ba',
    'brazil':'br','argentina':'ar','mexico':'mx','colombia':'co','uruguay':'uy',
    'chile':'cl','ecuador':'ec','peru':'pe','venezuela':'ve','paraguay':'py',
    'usa':'us','united states':'us','canada':'ca','costa rica':'cr',
    'australia':'au','japan':'jp','korea':'kr','south korea':'kr','china':'cn',
    'saudi arabia':'sa','uae':'ae','united arab emirates':'ae','qatar':'qa',
    'morocco':'ma','algeria':'dz','egypt':'eg','nigeria':'ng','senegal':'sn',
    'cameroon':'cm','ghana':'gh','ivory coast':'ci','mali':'ml','tunisia':'tn',
    'south africa':'za','russia':'ru','kazakhstan':'kz','israel':'il',
    'uzbekistan':'uz','azerbaijan':'az','sweden':'se',
  };
  const key = (country||'').toLowerCase().trim();
  const cc = CC[key];
  if (!cc) return '';
  // Special subdivision tags
  const SPECIAL = {'gb-eng':'1f3f4-e0067-e0062-e0065-e006e-e0067-e007f','gb-sct':'1f3f4-e0067-e0062-e0073-e0063-e0074-e007f','gb-wls':'1f3f4-e0067-e0062-e006c-e0073-e007f','gb-nir':'1f3f4-e0067-e0062-e006e-e0069-e0072-e007f'};
  let code;
  if (SPECIAL[cc]) {
    code = SPECIAL[cc];
  } else {
    const base = 0x1F1E6;
    const upper = cc.toUpperCase();
    const c1 = (base + upper.charCodeAt(0) - 65).toString(16);
    const c2 = (base + upper.charCodeAt(1) - 65).toString(16);
    code = c1 + '-' + c2;
  }
  return '<img src="https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/' + code + '.svg" style="height:13px;vertical-align:middle;margin-right:2px" />';
}

function getFlagUrl(league) {
  if (!league) return '';
  for (const [country, url] of Object.entries(FLAG_URLS)) {
    if (league.startsWith(country)) return url;
  }
  return '';
}

function leagueFlag(league) {
  if (!league) return '';
  const flagUrl = getFlagUrl(league);
  const code = Object.entries(LEAGUE_FLAGS).find(([c]) => league.startsWith(c))?.[1] || '';
  if (flagUrl) {
    return `<img src="${flagUrl}" style="width:18px;height:13px;border-radius:2px;object-fit:cover;flex-shrink:0;border:1px solid rgba(255,255,255,0.1)" onerror="this.outerHTML='<span style=\'font-size:9px;font-weight:800;color:var(--accent);background:rgba(0,200,240,0.1);border:1px solid rgba(0,200,240,0.2);padding:2px 5px;border-radius:3px\'>${code}</span>'" />`;
  }
  if (code) return `<span style="font-size:9px;font-weight:800;color:var(--accent);background:rgba(0,200,240,0.1);border:1px solid rgba(0,200,240,0.2);padding:2px 5px;border-radius:3px;letter-spacing:0.5px;flex-shrink:0">${code}</span>`;
  return '';
}

function leagueLogo(league) {
  const url = LEAGUE_LOGOS[league];
  if (!url) return '';
  return `<img src="${url}" style="width:22px;height:22px;object-fit:contain;flex-shrink:0;opacity:0.9" onerror="this.style.display='none'" />`;
}

// ── FORMAT ──
function fmtMV(v) {
  if (!v && v !== 0) return '—';
  const s = String(v).replace(/[£€m,k\s]/gi, '');
  const raw = String(v).toLowerCase();
  const n = parseFloat(s);
  if (isNaN(n)) return v || '—';
  // Handle already-abbreviated values like "4m" or "£4m"
  if (raw.includes('m')) return '€' + (n % 1 === 0 ? n : n.toFixed(1)) + 'm';
  if (raw.includes('k')) return '€' + Math.round(n) + 'k';
  // Raw number
  if (n >= 1000000) return '€' + (n/1000000).toFixed(n % 1000000 === 0 ? 0 : 1) + 'm';
  if (n >= 1000) return '€' + Math.round(n/1000) + 'k';
  return '€' + n;
}

// Returns market value in millions (for filtering)
function parseMV(v) {
  if (!v && v !== 0) return 0;
  const s = String(v).replace(/[£€,\s]/g,'').toLowerCase();
  const n = parseFloat(s);
  if (isNaN(n)) return 0;
  if (s.includes('m')) return n;
  if (s.includes('k')) return n / 1000;
  if (n >= 1000000) return n / 1000000;
  if (n >= 1000) return n / 1000;
  return n;
}

function scoreCls(s) {
  if (!s) return '';
  if (s.startsWith('80')) return 'sb-80';
  if (s.startsWith('76')) return 'sb-76';
  if (s.startsWith('72')) return 'sb-72';
  if (s.startsWith('66')) return 'sb-66';
  if (s.startsWith('60')) return 'sb-60';
  return 'sb-55';
}

function windowCls(w) {
  const m = { Summer:'summer', Monitor:'monitor', January:'january', Signed:'signed', Rejected:'rejected' };
  return 'pill pill-' + (m[w] || 'monitor');
}

function targetCls(t) {
  const m = {
    'Premier League':'pl','Championship':'champ','Europe Band 1':'eb1',
    'Europe Band 1-2':'eb12','Europe Band 2-3':'eb23',
    'PL Monitor':'plmon','CH Monitor':'chmon','TBC':'tbc'
  };
  return 'pill pill-' + (m[t] || 'tbc');
}

function statusCls(s) {
  const m = {'No Progress':'noprog','Relationship':'relationship','Agency Link':'agency','Contact Made':'contact'};
  return 'pill pill-' + (m[s] || 'noprog');
}

function agencyCls(a) {
  const m = { Small:'small', Medium:'medium', Big:'big' };
  return 'pill pill-' + (m[a] || 'small');
}

function moveCls(m) { return m === 'Yes' ? 'pill pill-yes' : m === 'No' ? 'pill pill-no' : ''; }

function stars(n, total=5) {
  const r = parseInt(n)||0;
  return '<span class="stars">' + '★'.repeat(r) + '</span><span class="stars-empty">' + '★'.repeat(total-r) + '</span>';
}

function photoEl(url, size=28) {
  if (url) return `<img class="player-avatar" src="${url}" style="width:${size}px;height:${size}px" onerror="this.parentNode.innerHTML='<div class=player-avatar-ph style=width:${size}px;height:${size}px>👤</div>'">`;
  return `<div class="player-avatar-ph" style="width:${size}px;height:${size}px">👤</div>`;
}

// ── TABS ──
function renderTabs() {
  const bar = document.getElementById('tabs-bar');
  const add = bar.querySelector('.tab-add');
  bar.querySelectorAll('.tab').forEach(t => t.remove());
  CORRECT_BOARDS.forEach(b => {
    const t = document.createElement('div');
    t.className = 'tab' + (b === currentBoard ? ' active' : '');
    t.textContent = b;
    t.onclick = () => { currentBoard = b; document.getElementById('board-title').textContent = b; renderTabs(); renderAll(); };
    t.ondblclick = () => renameTab(b, t);
    bar.insertBefore(t, add);
  });
}

function addTab() {
  const name = prompt('Board name:');
  if (!name) return;
  boards.push(name);
  allPlayers[name] = [];
  save();
  currentBoard = name;
  document.getElementById('board-title').textContent = name;
  renderTabs();
  renderAll();
}

function renameTab(old, el) {
  const name = prompt('Rename board:', old);
  if (!name || name === old) return;
  allPlayers[name] = allPlayers[old];
  delete allPlayers[old];
  boards = boards.map(b => b === old ? name : b);
  if (currentBoard === old) currentBoard = name;
  save();
  renderTabs();
}

// ── FILTER + RENDER ──
function getFiltered() {
  const q = document.getElementById('searchInput').value.toLowerCase();
  const wf = document.getElementById('windowFilter').value;
  const tf = document.getElementById('targetFilter').value;
  const sf = document.getElementById('statusFilter').value;
  const af = document.getElementById('agencyFilter').value;
  const footF     = document.getElementById('footFilter')?.value || '';
  const styleF    = document.getElementById('styleFilter')?.value || '';
  const valueF    = parseFloat(document.getElementById('valueFilter')?.value) || 0;
  const contractF = parseInt(document.getElementById('contractFilter')?.value) || 0;

  let list = [...players()];

  if (q) list = list.filter(p =>
    (p.name||'').toLowerCase().includes(q) ||
    (p.fullname||'').toLowerCase().includes(q) ||
    (p.team||'').toLowerCase().includes(q) ||
    (p.league||'').toLowerCase().includes(q) ||
    (p.nationality||'').toLowerCase().includes(q)
  );
  if (wf) list = list.filter(p => p.window === wf);
  if (tf) list = list.filter(p => p.target === tf);
  if (sf) list = list.filter(p => p.status === sf);
  if (af) list = list.filter(p => p.agency === af);
  if (footF)  list = list.filter(p => (p.foot||'').toLowerCase() === footF.toLowerCase());
  if (styleF) list = list.filter(p => (p.style||'') === styleF);
  if (valueF) list = list.filter(p => {
    const mv = parseMV(p.marketValue);
    return mv > 0 && mv <= valueF;
  });
  if (contractF) list = list.filter(p => {
    const yr = parseInt((p.contract||'').toString().slice(0,4));
    return yr > 0 && yr <= contractF;
  });

  const gbeBands = getSelectedGBEBands();
  if (gbeBands) {
    list = list.filter(p => {
      const band = JS_GBE_BAND[p.league] || 6;
      return gbeBands.includes(band);
    });
  }

  // Sort
  list.sort((a,b) => {
    let av = a[sortKey]||'', bv = b[sortKey]||'';
    if (['age','priority','games','goals','assists','minutes'].includes(sortKey)) {
      av = parseFloat(av)||0; bv = parseFloat(bv)||0;
      return sortAsc ? av-bv : bv-av;
    }
    return sortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
  });

  return list;
}

function setView(v) {
  currentView = v;
  document.getElementById('table-area').classList.toggle('hidden', v !== 'table');
  document.getElementById('league-area').classList.toggle('hidden', v !== 'league');
  document.getElementById('perf-area').classList.toggle('hidden', v !== 'perf');
  document.getElementById('vbtn-table').classList.toggle('active', v === 'table');
  document.getElementById('vbtn-league').classList.toggle('active', v === 'league');
  document.getElementById('vbtn-perf').classList.toggle('active', v === 'perf');
  renderAll();
}

function renderAll() {
  if (document.body.classList.contains('mobile-view')) {
    renderMobileCards();
    return;
  }
  if (currentView === 'table') renderTable();
  else if (currentView === 'league') renderLeague();
  else if (currentView === 'perf') renderPerformance();
}

function renderPerformance() {
  const list = getFiltered();
  const area = document.getElementById('perf-area');
  if (list.length === 0) {
    area.innerHTML = '<div class="empty"><div class="empty-icon">📊</div><div>No players found</div></div>';
    return;
  }

  area.innerHTML = `
    <div style="margin-bottom:16px;display:flex;align-items:center;justify-content:space-between">
      <div>
        <div style="font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted2);margin-bottom:4px">📊 Performance Rankings</div>
        <div style="font-size:12px;color:var(--muted2)">Complete Score — weighted metric blend vs league peers. Role bars show fit scores.</div>
      </div>
      <div id="perf-progress" style="font-size:11px;color:var(--muted2)">Loading 0/${list.length}...</div>
    </div>
    <div id="perf-content" style="display:grid;gap:6px"></div>`;

  if (!serverOnline) {
    document.getElementById('perf-progress').textContent = 'Server offline';
    return;
  }

  // Helper: map 0-100 to red→amber→green like the Scouting Hub
  function perfColor(v) {
    if (v == null || isNaN(v)) return '#6b7280';
    v = Math.max(0, Math.min(100, v));
    if (v >= 80) return '#10b981';
    if (v >= 65) return '#22c55e';
    if (v >= 50) return '#f59e0b';
    if (v >= 35) return '#f97316';
    return '#ef4444';
  }

  // Stagger fetches to avoid hammering server — 5 at a time
  const results = new Array(list.length).fill(null);
  let completed = 0;
  const CONCURRENCY = 5;
  let nextIndex = 0;

  function fetchNext() {
    if (nextIndex >= list.length) return;
    const i = nextIndex++;
    const p = list[i];
    fetch(`${SERVER}/api/player/profile?player=${encodeURIComponent(p.name||'')}&team=${encodeURIComponent(p.team||'')}&league=${encodeURIComponent(p.league||'')}&pos=${encodeURIComponent(currentBoard||'')}`)
      .then(r => r.json())
      .then(data => { results[i] = { player: p, roles: data.roles || {}, complete: data.complete_score ?? null }; })
      .catch(() => { results[i] = { player: p, roles: {}, complete: null }; })
      .finally(() => {
        completed++;
        document.getElementById('perf-progress').textContent = `Loading ${completed}/${list.length}...`;
        if (completed === list.length) renderPerfResults();
        else fetchNext();
      });
  }

  // Start initial batch
  for (let i = 0; i < Math.min(CONCURRENCY, list.length); i++) fetchNext();

  function renderPerfResults() {
    document.getElementById('perf-progress').textContent = `${list.length} players`;

    // Sort by Complete Score descending, fall back to best role score
    const sorted = [...results].sort((a, b) => {
      const aScore = a.complete ?? (Object.values(a.roles).length ? Math.max(...Object.values(a.roles)) : 0);
      const bScore = b.complete ?? (Object.values(b.roles).length ? Math.max(...Object.values(b.roles)) : 0);
      return bScore - aScore;
    });

    const content = document.getElementById('perf-content');
    if (!content) return;

    content.innerHTML = sorted.map((r, i) => {
      const p = r.player;
      const completeScore = r.complete != null ? Math.round(r.complete) : null;
      const roleScores = r.roles;

      // Display score: prefer Complete Score
      const displayScore = completeScore ?? (Object.values(roleScores).length ? Math.round(Math.max(...Object.values(roleScores))) : null);
      const scoreColor = displayScore != null ? perfColor(displayScore) : '#6b7280';

      const roleHtml = Object.entries(roleScores)
        .sort((a,b) => b[1]-a[1])
        .slice(0,3)
        .map(([role, score]) => {
          const s = Math.round(score);
          const c = perfColor(s);
          return `<div style="display:flex;align-items:center;gap:8px;margin-bottom:3px">
            <div style="font-size:11px;color:var(--muted2);width:120px;flex-shrink:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${role}</div>
            <div style="flex:1;height:5px;background:var(--s3);border-radius:3px;overflow:hidden;min-width:60px">
              <div style="height:100%;width:${s}%;background:${c};border-radius:3px;transition:width 0.4s ease"></div>
            </div>
            <div style="font-size:11px;font-weight:700;color:${c};width:26px;text-align:right;flex-shrink:0">${s}</div>
          </div>`;
        }).join('');

      // Use cached photo if available
      const cachedPhoto = _photoCache[p.id];
      const photoHtml = cachedPhoto
        ? `<img src="${cachedPhoto}" style="width:100%;height:100%;object-fit:cover;border-radius:8px">`
        : '👤';

      return `
        <div style="background:var(--s2);border:1px solid var(--border);border-radius:10px;padding:12px 16px;display:grid;grid-template-columns:28px 56px 1fr auto;gap:14px;align-items:center;cursor:pointer;transition:background 0.15s" onclick="openDetail('${p.id}')" onmouseenter="this.style.background='var(--s3)'" onmouseleave="this.style.background='var(--s2)'">
          <div style="font-size:16px;font-weight:900;color:var(--muted2);text-align:center">${i+1}</div>
          <div id="perf-photo-${p.id}" style="width:56px;height:56px;border-radius:8px;overflow:hidden;background:var(--s3);display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0">${photoHtml}</div>
          <div style="min-width:0">
            <div style="font-weight:700;font-size:13px;margin-bottom:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${p.name||'—'}</div>
            <div style="font-size:11px;color:var(--muted2);margin-bottom:7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${p.team||'—'} · ${p.league||'—'}</div>
            ${roleHtml || '<div style="font-size:11px;color:var(--muted2)">No data</div>'}
          </div>
          ${displayScore != null ? `
            <div style="text-align:center;flex-shrink:0;min-width:52px">
              <div style="font-size:26px;font-weight:900;color:${scoreColor};line-height:1">${displayScore}</div>
              <div style="font-size:8px;color:var(--muted2);letter-spacing:0.5px;text-transform:uppercase;margin-top:2px">${completeScore != null ? 'complete' : 'role fit'}</div>
            </div>` : '<div></div>'}
        </div>`;
    }).join('');

    // Load photos for perf cards using batch endpoint
    const uncachedPerf = sorted.filter(r => !_photoCache[r.player.id]);
    if (uncachedPerf.length > 0 && serverOnline) {
      fetch(`${SERVER}/api/photos/batch`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(uncachedPerf.map(r => ({id:r.player.id, name:r.player.name||'', team:r.player.team||'', league:r.player.league||''})))
      })
      .then(res => res.json())
      .then(urlMap => {
        Object.entries(urlMap).forEach(([pid, urls]) => {
          if (!urls?.length) return;
          let idx = 0;
          function tryNext() {
            if (idx >= urls.length) return;
            const url = urls[idx++];
            const img = new Image();
            img.onload = () => {
              _photoCache[pid] = url;
              const el = document.getElementById(`perf-photo-${pid}`);
              if (el) el.innerHTML = `<img src="${url}" style="width:100%;height:100%;object-fit:cover;border-radius:8px">`;
            };
            img.onerror = tryNext;
            img.src = url;
          }
          tryNext();
        });
      })
      .catch(() => {
        // Fallback to direct URL
        sorted.forEach(r => {
          if (_photoCache[r.player.id]) return;
          tryPhotoUrls(getPhotoUrls(r.player.name, r.player.team), r.player.id, `perf-photo-${r.player.id}`);
        });
      });
    }
  }
}

function renderLeague() {
  const list = getFiltered();
  document.getElementById('rec-count').textContent = list.length + ' player' + (list.length !== 1 ? 's' : '');
  const area = document.getElementById('league-area');

  if (list.length === 0) {
    area.innerHTML = '<div class="empty"><div class="empty-icon">📋</div><div>No players found</div><div style="font-size:13px;color:var(--muted2);margin-top:6px">Add your first player or adjust filters</div></div>';
    return;
  }

  // Group by league, sort by league strength then name
  const groups = {};
  list.forEach(p => {
    const lg = p.league || 'Unknown';
    if (!groups[lg]) groups[lg] = [];
    groups[lg].push(p);
  });

  const sorted = Object.keys(groups).sort();

  const COLS = [
    {k:'name',        l:'Player'},
    {k:'team',        l:'Team'},
    {k:'age',         l:'Age'},
    {k:'marketValue', l:'Value'},
    {k:'contract',    l:'Contract'},
    {k:'style',       l:'Style'},
    {k:'foot',        l:'Foot'},
    {k:'score',       l:'Score*'},
    {k:'potential',   l:'Potential*'},
    {k:'games',       l:'Games'},
    {k:'goals',       l:'Goals'},
    {k:'assists',     l:'Assists'},
    {k:'minutes',     l:'Mins'},
  ];

  const theadHtml = `<tr>
    ${COLS.map(c => `<th onclick="setSort('${c.k}')" style="cursor:pointer" class="${sortKey===c.k?(sortAsc?'sort-asc':'sort-desc'):''}">${c.l}</th>`).join('')}
  </tr>`;

  area.innerHTML = sorted.map(lg => {
    const rows = groups[lg];
    const totalGames   = rows.reduce((s,p) => s+(parseFloat(p.games)||0), 0);
    const totalGoals   = rows.reduce((s,p) => s+(parseFloat(p.goals)||0), 0);
    const totalAssists = rows.reduce((s,p) => s+(parseFloat(p.assists)||0), 0);
    const flag = leagueFlag(lg);

    const bodyRows = rows.map(p => `
      <tr onclick="openDetail('${p.id}')">
        <td>
          <div class="td-player" style="min-width:160px">
            <div id="avatar-lg-${p.id}" class="player-avatar-ph" style="width:28px;height:28px;font-size:12px">👤</div>
            <div>
              <div class="player-name">${p.name||'—'}</div>
              ${p.nationality ? `<div style="font-size:10px;color:var(--muted2)">${p.nationality}</div>` : ''}
            </div>
            ${p.pos ? `<span class="player-pos-tag">${p.pos}</span>` : ''}
          </div>
        </td>
        <td style="color:var(--muted)">${p.team||'—'}</td>
        <td class="mono" style="text-align:center">${p.age||'—'}</td>
        <td class="mono val-green">${fmtMV(p.marketValue)}</td>
        <td class="mono">${p.contract||'—'}</td>
        <td>${p.style ? `<span class="style-tag">${p.style}</span>` : '—'}</td>
        <td style="color:var(--muted);font-style:italic">${p.foot||'—'}</td>
        <td>${p.score ? `<span class="score-band ${scoreCls(p.score)}">${p.score}</span>` : '—'}</td>
        <td>${p.potential ? `<span class="score-band ${scoreCls(p.potential)}">${p.potential}</span>` : '—'}</td>
        <td class="mono" style="text-align:right">${p.games||'—'}</td>
        <td class="mono" style="text-align:right">${p.goals||'—'}</td>
        <td class="mono" style="text-align:right">${p.assists||'—'}</td>
        <td class="mono" style="text-align:right">${p.minutes||'—'}</td>
      </tr>`).join('');

    const gid = 'lg-' + lg.replace(/[^a-z0-9]/gi,'_');
    const regionColor = leagueRegionColor(lg);
    return `
      <div class="league-group">
        <div class="lg-header" onclick="toggleLg('${gid}')" style="border-left:4px solid ${regionColor}">
          <span style="color:var(--muted2);font-size:11px;width:14px">▼</span>
          ${leagueFlag(lg)}
          ${leagueLogo(lg)}
          <div class="lg-name" style="color:${regionColor}">${lg}</div>
          <div class="lg-count">${rows.length}</div>
          <div class="lg-stats">
            <span>⚽ ${totalGoals}G</span>
            <span>🅰 ${totalAssists}A</span>
            <span>🎮 ${totalGames} apps</span>
          </div>
        </div>
        <div class="lg-body" id="${gid}" style="max-height:${rows.length * 52 + 44}px">
          <div style="overflow-x:auto">
            <table class="lg-table" style="min-width:900px">
              <thead>${theadHtml}</thead>
              <tbody>${bodyRows}</tbody>
            </table>
          </div>
        </div>
      </div>`;
  }).join('');

  // Load photos for league view rows
  setTimeout(() => loadLeaguePhotos(list), 50);
}


function toggleLg(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.toggle('collapsed');
  const arrow = el.previousElementSibling.querySelector('span');
  if (arrow) arrow.textContent = el.classList.contains('collapsed') ? '▶' : '▼';
}


function renderTable() {
  const list = getFiltered();
  document.getElementById('rec-count').textContent = list.length + ' player' + (list.length !== 1 ? 's' : '');

  const thead = document.getElementById('thead');
  const cols = TABLE_COLS.filter(c => c.always || visibleCols.has(c.k));

  thead.innerHTML = '<tr>' + cols.map(c => {
    const sc = sortKey === c.k ? (sortAsc ? 'sort-asc' : 'sort-desc') : '';
    return `<th class="${c.cls||''} ${sc}" onclick="${c.k ? `setSort('${c.k}')` : ''}">${c.l}</th>`;
  }).join('') + '</tr>';

  const tbody = document.getElementById('tbody');

  if (list.length === 0) {
    tbody.innerHTML = '';
    document.getElementById('empty-state').classList.remove('hidden');
    document.getElementById('main-table').classList.add('hidden');
    return;
  }
  document.getElementById('empty-state').classList.add('hidden');
  document.getElementById('main-table').classList.remove('hidden');

  const show = k => visibleCols.has(k);
  tbody.innerHTML = list.map((p,i) => {
    const cells = [];
    // Always: checkbox
    cells.push(`<td onclick="event.stopPropagation()" class="cb-wrap"><input type="checkbox" class="table-checkbox"></td>`);
    // Always: player
    cells.push(`<td><div class="td-player"><div id="avatar-${p.id}" class="player-avatar-ph" style="width:28px;height:28px;font-size:12px">👤</div><div><div class="player-name">${p.name||'—'}</div>${p.nationality?`<div style="font-size:10px;color:var(--muted2)">${p.nationality}</div>`:''}</div>${p.pos?`<span class="player-pos-tag">${p.pos}</span>`:''}</div></td>`);
    if (show('team'))        cells.push(`<td>${p.team||'—'}</td>`);
    if (show('league'))      cells.push(`<td style="color:var(--muted)">${p.league||'—'}</td>`);
    if (show('age'))         cells.push(`<td class="mono" style="text-align:center">${p.age||'—'}</td>`);
    if (show('window'))      cells.push(`<td>${p.window?`<span class="${windowCls(p.window)}">${p.window}</span>`:'—'}</td>`);
    if (show('target'))      cells.push(`<td>${p.target?`<span class="${targetCls(p.target)}">${p.target}</span>`:'—'}</td>`);
    if (show('tm'))          cells.push(`<td onclick="event.stopPropagation()">${p.tm?`<a class="tm-link" href="${p.tm}" target="_blank">🔗 TM Link</a>`:'<span style="color:var(--muted2);font-size:11px">—</span>'}</td>`);
    if (show('style'))       cells.push(`<td>${p.style?`<span class="style-tag">${p.style}</span>`:'—'}</td>`);
    if (show('role1'))       cells.push(`<td>${p.role1?`<span class="role-badge">${p.role1}</span>`:'—'}</td>`);
    if (show('priority'))    cells.push(`<td style="text-align:center">${p.priority?stars(p.priority):'—'}</td>`);
    if (show('status'))      cells.push(`<td>${p.status?`<span class="${statusCls(p.status)}">${p.status}</span>`:'—'}</td>`);
    if (show('agency'))      cells.push(`<td>${p.agency?`<span class="${agencyCls(p.agency)}">${p.agency}</span>`:'—'}</td>`);
    if (show('recentmove'))  cells.push(`<td>${p.recentmove?`<span class="${moveCls(p.recentmove)}">${p.recentmove}</span>`:'—'}</td>`);
    if (show('marketValue')) cells.push(`<td class="mono val-green">${fmtMV(p.marketValue)}</td>`);
    if (show('contract'))    cells.push(`<td class="mono">${p.contract||'—'}</td>`);
    if (show('foot'))        cells.push(`<td style="color:var(--muted);font-style:italic">${p.foot||'—'}</td>`);
    if (show('score'))       cells.push(`<td>${p.score?`<span class="score-band ${scoreCls(p.score)}">${p.score}</span>`:'—'}</td>`);
    if (show('potential'))   cells.push(`<td>${p.potential?`<span class="score-band ${scoreCls(p.potential)}">${p.potential}</span>`:'—'}</td>`);
    if (show('physical'))    cells.push(`<td>${p.physical?`<span class="phys-tag">${p.physical}</span>`:'—'}</td>`);
    if (show('keynotes'))    cells.push(`<td><div class="notes-text">${p.keynotes||'—'}</div></td>`);
    if (show('games'))       cells.push(`<td class="mono text-right">${p.games||'—'}</td>`);
    if (show('goals'))       cells.push(`<td class="mono text-right">${p.goals||'—'}</td>`);
    if (show('assists'))     cells.push(`<td class="mono text-right">${p.assists||'—'}</td>`);
    if (show('minutes'))     cells.push(`<td class="mono text-right">${p.minutes||'—'}</td>`);
    return `<tr onclick="openDetail('${p.id}')" style="animation-delay:${i*0.02}s">${cells.join('')}</tr>`;
  }).join('');

  // Load photos for visible rows
  setTimeout(() => loadTablePhotos(list), 50);
}


// ── COLUMN PICKER ──
const TABLE_COLS = [
  {k:null,  l:'',              cls:'th-check', always:true},
  {k:'name',l:'Player',        cls:'th-player', always:true},
  {k:'team',       l:'Team'},
  {k:'league',     l:'League'},
  {k:'age',        l:'Age'},
  {k:'window',     l:'Window'},
  {k:'target',     l:'Target Move'},
  {k:'tm',         l:'Transfermarkt'},
  {k:'style',      l:'Style'},
  {k:'role1',      l:'Roles'},
  {k:'priority',   l:'Priority'},
  {k:'status',     l:'Status'},
  {k:'agency',     l:'Agency'},
  {k:'recentmove', l:'Recent Move'},
  {k:'marketValue',l:'Value'},
  {k:'contract',   l:'Contract'},
  {k:'foot',       l:'Foot'},
  {k:'score',      l:'Score*'},
  {k:'potential',  l:'Potential*'},
  {k:'physical',   l:'Physical'},
  {k:'keynotes',   l:'Key Notes'},
  {k:'games',      l:'Games'},
  {k:'goals',      l:'Goals'},
  {k:'assists',    l:'Assists'},
  {k:'minutes',    l:'Mins'},
];

const DEFAULT_VISIBLE = new Set(['team','league','age','window','target','tm','style','role1','priority','status','agency','recentmove','marketValue','contract','foot','score','potential','physical','keynotes','games','goals','assists','minutes']);
let visibleCols = new Set(DEFAULT_VISIBLE);

function toggleColPicker() {
  const el = document.getElementById('col-picker');
  const open = el.style.display !== 'none';
  if (open) { el.style.display = 'none'; return; }
  // Build list
  const list = document.getElementById('col-picker-list');
  list.innerHTML = TABLE_COLS.filter(c => !c.always).map(c => `
    <label style="display:flex;align-items:center;gap:8px;cursor:pointer;padding:3px 4px;border-radius:5px;transition:background .15s" onmouseover="this.style.background='var(--s3)'" onmouseout="this.style.background=''">
      <input type="checkbox" ${visibleCols.has(c.k) ? 'checked' : ''} onchange="toggleCol('${c.k}',this.checked)" style="accent-color:var(--accent);width:14px;height:14px;cursor:pointer">
      <span style="font-size:12px;color:var(--fg)">${c.l}</span>
    </label>`).join('');
  el.style.display = 'block';
  // Close on outside click
  setTimeout(() => document.addEventListener('click', colPickerOutside, {once:true}), 0);
}
function colPickerOutside(e) {
  if (!document.getElementById('col-picker').contains(e.target)) {
    document.getElementById('col-picker').style.display = 'none';
  } else {
    setTimeout(() => document.addEventListener('click', colPickerOutside, {once:true}), 0);
  }
}
function toggleCol(k, on) {
  if (on) visibleCols.add(k); else visibleCols.delete(k);
  renderTable();
}
function colPickerAll(on) {
  TABLE_COLS.filter(c => !c.always && c.k).forEach(c => on ? visibleCols.add(c.k) : visibleCols.delete(c.k));
  document.getElementById('col-picker').style.display = 'none';
  renderTable();
}
function colPickerReset() {
  visibleCols = new Set(DEFAULT_VISIBLE);
  document.getElementById('col-picker').style.display = 'none';
  renderTable();
}

function toggleExtraFilters() {
  const el = document.getElementById('extra-filters');
  const btn = document.getElementById('extra-filter-btn');
  const open = el.style.display === 'flex';
  el.style.display = open ? 'none' : 'flex';
  btn.style.color = open ? 'var(--muted)' : 'var(--accent)';
  btn.style.borderColor = open ? 'var(--border)' : 'var(--accent)';
}

function clearExtraFilters() {
  ['footFilter','styleFilter','valueFilter','contractFilter'].forEach(id => {
    document.getElementById(id).value = '';
  });
  setAllGBEBands(true);
  renderAll();
}

function toggleGBEBandPicker() {
  const el = document.getElementById('gbe-band-picker');
  const open = el.style.display !== 'none';
  el.style.display = open ? 'none' : 'block';
  if (!open) setTimeout(() => document.addEventListener('click', gbePickerOutside, {once:true}), 0);
}
function gbePickerOutside(e) {
  const picker = document.getElementById('gbe-band-picker');
  const btn = document.getElementById('gbe-band-btn');
  if (!picker?.contains(e.target) && e.target !== btn) {
    picker.style.display = 'none';
  } else {
    setTimeout(() => document.addEventListener('click', gbePickerOutside, {once:true}), 0);
  }
}
function setAllGBEBands(on) {
  document.querySelectorAll('.gbe-band-cb').forEach(cb => cb.checked = on);
  applyGBEBandFilter();
}
function applyGBEBandFilter() {
  const checked = [...document.querySelectorAll('.gbe-band-cb:checked')].map(cb => cb.value);
  const btn = document.getElementById('gbe-band-btn');
  if (btn) btn.textContent = checked.length === 6 ? 'All Bands ▾' : `Band ${checked.join(',')} ▾`;
  renderAll();
}
function getSelectedGBEBands() {
  const cbs = [...document.querySelectorAll('.gbe-band-cb')];
  if (!cbs.length) return null; // picker not open yet — no filter
  const checked = cbs.filter(cb => cb.checked).map(cb => parseInt(cb.value));
  if (checked.length === 0 || checked.length === cbs.length) return null; // all or none = no filter
  return checked;
}

function setSort(k) {
  if (sortKey === k) sortAsc = !sortAsc;
  else { sortKey = k; sortAsc = true; }
  renderAll();
}

// ── ADD / EDIT ──
function openAdd() {
  editingId = null;
  document.getElementById('add-title').textContent = 'NEW PLAYER';
  document.getElementById('delete-btn').style.display = 'none';
  const fields = ['name','fullname','team','league','age','foot','window','target','status','agency',
    'recentmove','priority','style','role1','score','potential','physical','nationality',
    'value','contract','games','goals','assists','minutes','ga','tm','video','photo','pos','keynotes','note'];
  fields.forEach(f => {
    const el = document.getElementById('f-'+f);
    if (!el) return;
    if (el.tagName === 'SELECT') el.selectedIndex = 0;
    else el.value = '';
  });
  document.getElementById('f-priority').value = '3';
  document.getElementById('f-dataprofile').value = '';
  document.getElementById('f-dataprofile-preview').style.display = 'none';
  document.getElementById('f-dataprofile-img').src = '';
  document.getElementById('add-overlay').classList.add('open');
}

function openEdit(id) {
  const p = players().find(x => x.id === id);
  if (!p) return;
  editingId = id;
  document.getElementById('add-title').textContent = 'EDIT PLAYER';
  document.getElementById('delete-btn').style.display = 'block';

  const map = {
    'f-name':'name','f-fullname':'fullname','f-team':'team','f-league':'league',
    'f-age':'age','f-foot':'foot','f-window':'window','f-target':'target',
    'f-status':'status','f-agency':'agency','f-recentmove':'recentmove',
    'f-priority':'priority','f-style':'style','f-role1':'role1',
    'f-score':'score','f-potential':'potential','f-physical':'physical',
    'f-nationality':'nationality','f-value':'marketValue','f-contract':'contract',
    'f-games':'games','f-goals':'goals','f-assists':'assists','f-minutes':'minutes',
    'f-ga':'goalsAgainst','f-tm':'tm','f-video':'video','f-photo':'photo',
    'f-pos':'pos','f-keynotes':'keynotes'
  };
  Object.entries(map).forEach(([fid, pk]) => {
    const el = document.getElementById(fid);
    if (el) el.value = p[pk] || '';
  });
  document.getElementById('f-note').value = '';

  // Repopulate data profile preview
  const dp = document.getElementById('f-dataprofile');
  const dpPreview = document.getElementById('f-dataprofile-preview');
  const dpImg = document.getElementById('f-dataprofile-img');
  if (p.dataProfile) {
    dp.value = p.dataProfile;
    dpImg.src = p.dataProfile;
    dpPreview.style.display = 'block';
    document.getElementById('f-dataprofile-name').textContent = 'Profile attached';
  } else {
    dp.value = '';
    dpPreview.style.display = 'none';
    dpImg.src = '';
    document.getElementById('f-dataprofile-name').textContent = 'No file attached';
  }

  document.getElementById('add-overlay').classList.add('open');
}

function closeAdd() { document.getElementById('add-overlay').classList.remove('open'); }

function loadDataProfileFile(input) {
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    const b64 = e.target.result;
    document.getElementById('f-dataprofile').value = b64;
    document.getElementById('f-dataprofile-img').src = b64;
    document.getElementById('f-dataprofile-preview').style.display = 'block';
    document.getElementById('f-dataprofile-name').textContent = file.name;
  };
  reader.readAsDataURL(file);
  input.value = '';
}

function clearDataProfile() {
  document.getElementById('f-dataprofile').value = '';
  document.getElementById('f-dataprofile-img').src = '';
  document.getElementById('f-dataprofile-preview').style.display = 'none';
  document.getElementById('f-dataprofile-name').textContent = 'No file attached';
}

function savePlayer() {
  const name = document.getElementById('f-name').value.trim();
  if (!name) { showToast('❌ Player name required'); return; }

  const noteText = document.getElementById('f-note').value.trim();
  const noteEntry = noteText ? [{ text: noteText, date: new Date().toLocaleString('en-GB', {day:'2-digit',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit'}) }] : [];

  const data = {
    name, fullname: document.getElementById('f-fullname').value.trim(),
    team: document.getElementById('f-team').value.trim(),
    league: document.getElementById('f-league').value.trim(),
    age: document.getElementById('f-age').value,
    foot: document.getElementById('f-foot').value,
    pos: document.getElementById('f-pos').value,
    window: document.getElementById('f-window').value,
    target: document.getElementById('f-target').value,
    status: document.getElementById('f-status').value,
    agency: document.getElementById('f-agency').value,
    recentmove: document.getElementById('f-recentmove').value,
    priority: document.getElementById('f-priority').value,
    style: document.getElementById('f-style').value,
    role1: document.getElementById('f-role1').value,
    score: document.getElementById('f-score').value,
    potential: document.getElementById('f-potential').value,
    physical: document.getElementById('f-physical').value,
    nationality: document.getElementById('f-nationality').value.trim(),
    marketValue: document.getElementById('f-value').value,
    contract: document.getElementById('f-contract').value.trim(),
    games: document.getElementById('f-games').value,
    goals: document.getElementById('f-goals').value,
    assists: document.getElementById('f-assists').value,
    minutes: document.getElementById('f-minutes').value,
    goalsAgainst: document.getElementById('f-ga').value,
    tm: document.getElementById('f-tm').value.trim(),
    video: document.getElementById('f-video').value.trim(),
    photo: document.getElementById('f-photo').value.trim(),
    dataProfile: document.getElementById('f-dataprofile').value || undefined,
    keynotes: document.getElementById('f-keynotes').value.trim(),
    updatedAt: new Date().toISOString()
  };

  if (editingId) {
    const idx = allPlayers[currentBoard].findIndex(p => p.id === editingId);
    if (idx !== -1) {
      const existing = allPlayers[currentBoard][idx];
      allPlayers[currentBoard][idx] = { ...existing, ...data, notes: [...(existing.notes||[]), ...noteEntry] };
      showToast('✅ Player updated');
    }
  } else {
    allPlayers[currentBoard].push({ id: uid(), ...data, notes: noteEntry, createdAt: new Date().toISOString() });
    showToast('✅ Player added to ' + currentBoard);
  }

  save();
  closeAdd();
  renderAll();
}

function deleteEditing() {
  if (!editingId) return;
  if (!confirm('Delete this player?')) return;
  allPlayers[currentBoard] = allPlayers[currentBoard].filter(p => p.id !== editingId);
  save();
  closeAdd();
  renderAll();
  showToast('🗑 Player deleted');
}

// FotMob team IDs for badge images
const TEAM_FOTMOB_IDS = {
  'Liverpool':8650,'Arsenal':9825,'Manchester City':8456,'Manchester United':10260,
  'Chelsea':8455,'Tottenham Hotspur':8586,'Newcastle United':10261,'Aston Villa':10252,
  'Brighton':10204,'West Ham United':8654,'Fulham':9879,'Brentford':9937,
  'Crystal Palace':9826,'Everton':8668,'Nottingham Forest':10203,'Wolverhampton Wanderers':8602,
  'Bournemouth':8678,'Ipswich Town':9902,'Leicester City':8197,'Southampton':8466,
  'Leeds United':8463,'Sunderland':8472,'Derby County':10170,'Burnley':8191,
  'Sheffield United':8657,'Luton Town':8346,'Middlesbrough':8549,'Norwich City':9850,
  'Swansea City':10003,'Watford':9817,'Coventry City':8669,'Cardiff City':8344,
  'Real Madrid':8633,'Barcelona':8634,'Atletico Madrid':9906,'Sevilla':8302,
  'Real Betis':8603,'Villarreal':10205,'Real Sociedad':8560,'Athletic Club':8315,
  'Valencia':10267,'Osasuna':8371,'Celta de Vigo':9910,'Getafe':8305,
  'Rayo Vallecano':8370,'Mallorca':8661,'Las Palmas':8306,'Girona':7732,
  'Espanyol':8558,'Deportivo Alaves':9866,'Bayern München':9823,'Borussia Dortmund':9789,
  'RB Leipzig':178475,'Bayer Leverkusen':8178,'Eintracht Frankfurt':9810,
  'Freiburg':8358,'Wolfsburg':8721,'Hoffenheim':8226,'Borussia M\'gladbach':9788,
  'Union Berlin':8149,'Augsburg':8406,'Mainz 05':9905,'Werder Bremen':8697,
  'Stuttgart':10269,'St. Pauli':8152,'Heidenheim':94937,'Holstein Kiel':8150,
  'Hamburger SV':9790,'Schalke 04':10189,'Hannover 96':9904,'Kaiserslautern':8350,
  'PSG':9847,'Olympique Marseille':8592,'Olympique Lyonnais':9748,'Monaco':9829,
  'Lille':8639,'Nice':9831,'Rennes':9851,'Lens':8588,'Strasbourg':9848,
  'Brest':8521,'Nantes':9830,'Toulouse':9941,'Reims':9837,'Angers SCO':8121,
  'Le Havre':9746,'Saint-Etienne':9853,'Montpellier':10249,'Metz':8550,
  'Lorient':8689,'Juventus':9885,'Inter':8636,'Milan':8564,'Napoli':9875,
  'Roma':8686,'Lazio':8543,'Atalanta':8524,'Fiorentina':8535,'Torino':9804,
  'Bologna':9857,'Udinese':8600,'Genoa':10233,'Cagliari':8529,'Lecce':9888,
  'Hellas Verona':9876,'Como':10171,'Parma':10167,'Empoli':8534,'Venezia':7881,
  'Monza':6504,'Sassuolo':7943,'Benfica':9772,'Porto':9773,'Sporting CP':9768,
  'Braga':10264,'Vitoria Guimaraes':7844,'Rio Ave':7841,'Gil Vicente':9764,
  'Famalicao':1634,'Estoril':7842,'Arouca':158085,'Casa Pia AC':212821,
  'Ajax':8593,'PSV':8640,'Feyenoord':10235,'AZ':10229,'Twente':8611,
  'Utrecht':9908,'Heerenveen':10228,'NEC':8464,'Go Ahead Eagles':6433,
  'Celtic':9925,'Rangers':8548,'Hearts':9860,'Hibernian':10251,'Aberdeen':8485,
  'Dundee':8284,'Motherwell':9927,'St. Mirren':9800,'Ross County':8649,
  'Club Brugge':8342,'Anderlecht':8635,'Gent':9991,'Genk':9987,'Antwerp':9988,
  'Galatasaray':8637,'Fenerbahce':8695,'Besiktas':10188,'Trabzonspor':9752,
  'Flamengo':9770,'Palmeiras':10283,'Atletico Mineiro':10272,'Fluminense':9863,
  'Internacional':8702,'Cruzeiro':9781,'Botafogo':8517,'Sao Paulo':10277,
  'Corinthians':9808,'Fortaleza':8287,'Bahia':7877,'Atletico GO':165545,
  'Vasco da Gama':10276,'Gremio':9769,'RB Leipzig':178475,
  'Zrinjski':10107,'UD Las Palmas':8306,'Mönchengladbach':9788,
};

function getTeamFotmobId(teamName) {
  if (!teamName) return null;
  // Direct match
  if (TEAM_FOTMOB_IDS[teamName]) return TEAM_FOTMOB_IDS[teamName];
  // Partial match
  for (const [name, id] of Object.entries(TEAM_FOTMOB_IDS)) {
    if (teamName.toLowerCase().includes(name.toLowerCase()) || name.toLowerCase().includes(teamName.toLowerCase())) {
      return id;
    }
  }
  return null;
}
const SERVER = window.location.hostname === 'localhost' ? 'http://localhost:5000' : window.location.origin;
// On Railway the server IS the app — always online. Only false for local without server.py
let serverOnline = window.location.hostname !== 'localhost';

async function checkServer() {
  try {
    const r = await fetch(SERVER + '/api/health', {signal: AbortSignal.timeout(5000)});
    serverOnline = r.ok;
  } catch {
    serverOnline = window.location.hostname !== 'localhost';
  }
  return serverOnline;
}
checkServer();

function roleScoreColor(v) {
  if (v == null) return '#1a2235';
  if (v >= 75) return '#065f46';
  if (v >= 60) return '#166534';
  if (v >= 50) return '#854d0e';
  if (v >= 35) return '#7f1d1d';
  return '#1f2937';
}

function roleScoreTextColor(v) {
  if (v == null) return '#6b82a0';
  if (v >= 60) return '#6ee7b7';
  if (v >= 50) return '#fde68a';
  return '#fca5a5';
}

function pctColor(v) {
  if (v == null) return '#3f5270';
  if (v >= 80) return '#6ee7b7';
  if (v >= 65) return '#86efac';
  if (v >= 50) return '#fde68a';
  if (v >= 35) return '#fdba74';
  return '#fca5a5';
}

function renderRadarSVG(scores) {
  const keys = Object.keys(scores);
  if (keys.length < 3) return '';
  const n = keys.length;
  const cx = 150, cy = 150, r = 85;
  const isDark = !document.body.classList.contains('light-mode');
  const angles = keys.map((_, i) => (i / n) * 2 * Math.PI - Math.PI / 2);

  const gridStroke  = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
  const gridOuter   = isDark ? 'rgba(255,255,255,0.22)' : 'rgba(0,0,0,0.18)';
  const spokeStroke = isDark ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.10)';
  const labelColor  = isDark ? '#94a3b8' : '#64748b';

  function pt(pct, angle) {
    const rad = (pct / 100) * r;
    return [cx + rad * Math.cos(angle), cy + rad * Math.sin(angle)];
  }

  // Grid rings
  const rings = [25,50,75,100].map(pct => {
    const pts = angles.map(a => pt(pct, a).join(',')).join(' ');
    const isOuter = pct === 100;
    return `<polygon points="${pts}" fill="none" stroke="${isOuter?gridOuter:gridStroke}" stroke-width="${isOuter?1.5:0.7}" stroke-dasharray="${pct===50?'4,3':'none'}"/>`;
  }).join('');

  // Spokes
  const spokes = angles.map(a => {
    const [x,y] = pt(100, a);
    return `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="${spokeStroke}" stroke-width="1"/>`;
  }).join('');

  // Fotmob-style: each sector is a filled wedge from centre to the data value
  // Colour of each wedge depends on score: green ≥70, amber 45-69, red <45
  function sectorColor(score, alpha) {
    if (score >= 70) return isDark ? `rgba(74,222,128,${alpha})` : `rgba(22,163,74,${alpha})`;
    if (score >= 45) return isDark ? `rgba(251,191,36,${alpha})` : `rgba(180,83,9,${alpha})`;
    return isDark ? `rgba(248,113,113,${alpha})` : `rgba(220,38,38,${alpha})`;
  }

  // Each sector: triangle from centre → spoke i data point → spoke i+1 data point
  const sectors = keys.map((k, i) => {
    const score = scores[k] ?? 0;
    const a1 = angles[i];
    const a2 = angles[(i + 1) % n];
    const [x1, y1] = pt(score, a1);
    const [x2, y2] = pt(score, a2);
    // Arc through intermediate angles for curved sector edge
    const steps = 8;
    let arcPts = [];
    for (let s = 0; s <= steps; s++) {
      const a = a1 + (a2 - a1) * (s / steps);
      arcPts.push(pt(score, a).join(','));
    }
    const pts = `${cx},${cy} ${arcPts.join(' ')}`;
    return `<polygon points="${pts}" fill="${sectorColor(score, isDark?0.28:0.22)}" stroke="${sectorColor(score, isDark?0.70:0.65)}" stroke-width="0.5" stroke-linejoin="round"/>`;
  }).join('');

  // Outer boundary line connecting all data points (smooth)
  const boundaryPts = [];
  keys.forEach((k, i) => {
    const score = scores[k] ?? 0;
    const a1 = angles[i];
    const a2 = angles[(i + 1) % n];
    const steps = 8;
    for (let s = 0; s <= steps; s++) {
      const a = a1 + (a2 - a1) * (s / steps);
      boundaryPts.push(pt(score, a).join(','));
    }
  });

  // Labels at bisector between spokes, outside the ring
  const labels = keys.map((k, i) => {
    const a1 = angles[i];
    const a2 = angles[(i + 1) % n];
    let diff = a2 - a1;
    if (diff < 0) diff += 2 * Math.PI;
    const bisect = a1 + diff / 2;
    const labelR = r + 42;
    const lx = cx + labelR * Math.cos(bisect);
    const ly = cy + labelR * Math.sin(bisect);
    const anchor = lx < cx - 10 ? 'end' : lx > cx + 10 ? 'start' : 'middle';
    const score = scores[k] ?? 0;
    const sColor = sectorColor(score, 1.0).replace(/[\d.]+\)$/, '1)');

    return `<text x="${lx.toFixed(1)}" y="${(ly-6).toFixed(1)}" text-anchor="${anchor}" dominant-baseline="middle" font-size="8.5" font-weight="600" fill="${labelColor}" font-family="DM Sans,sans-serif">${k}</text>
<text x="${lx.toFixed(1)}" y="${(ly+7).toFixed(1)}" text-anchor="${anchor}" dominant-baseline="middle" font-size="10" font-weight="700" fill="${sColor}" font-family="DM Mono,monospace">${score}%</text>`;
  }).join('');

  return `<svg viewBox="0 0 300 300" width="240" height="240" style="flex-shrink:0;overflow:visible">
    ${rings}${spokes}${sectors}
    <polyline points="${boundaryPts.join(' ')}" fill="none" stroke="${isDark?'rgba(255,255,255,0.25)':'rgba(0,0,0,0.18)'}" stroke-width="1" stroke-linejoin="round"/>
    ${labels}
  </svg>`;
}

function renderDetailStatic(p) {
  const scoreVal = p.score ? parseInt(p.score.split('-')[0]) : 0;
  const potVal = p.potential ? parseInt(p.potential.split('-')[0]) : 0;
  const flagUrl = getFlagUrl(p.league || '');
  const flagHtml = flagUrl ? `<img src="${flagUrl}" style="width:16px;height:11px;border-radius:2px;object-fit:cover;border:1px solid rgba(255,255,255,0.15);vertical-align:middle;margin-right:4px" />` : '';
  return `
    <div class="detail-hero">
      <div id="detail-photo-wrap"><div class="detail-photo-ph">👤</div></div>
      <div class="detail-info">
        <div class="detail-name">${p.name||'—'}</div>
        <div class="detail-meta">
          ${p.team ? '<div class="detail-meta-chip" style="display:flex;align-items:center;gap:6px"><img id="detail-team-badge" src="" style="width:18px;height:18px;object-fit:contain;display:none" /><span id="detail-team-emoji">🏟</span> ' + p.team + '</div>' : ''}
          ${p.league ? '<div class="detail-meta-chip">' + flagHtml + p.league + '</div>' : ''}
          ${p.pos ? '<div class="detail-meta-chip">📍 ' + p.pos + '</div>' : ''}
          ${p.age ? '<div class="detail-meta-chip">🎂 ' + p.age + 'y</div>' : ''}
          ${p.foot ? '<div class="detail-meta-chip">🦶 ' + p.foot + '</div>' : ''}
          <div id="detail-height-chip"></div>
          ${p.nationality ? '<div class="detail-meta-chip" id="detail-nat-chip">🏳 ' + p.nationality + '</div>' : '<div id="detail-nat-chip"></div>'}
        </div>
        <div class="detail-tags">
          ${p.window ? `<span class="${windowCls(p.window)}">${p.window}</span>` : ''}
          ${p.target ? `<span class="${targetCls(p.target)}">${p.target}</span>` : ''}
          ${p.status ? `<span class="${statusCls(p.status)}">${p.status}</span>` : ''}
          ${p.agency ? `<span class="${agencyCls(p.agency)}">${p.agency}</span>` : ''}
          ${p.recentmove ? '<span class="' + moveCls(p.recentmove) + '">Recent Move: ' + p.recentmove + '</span>' : ''}
          ${p.role1 ? `<span class="role-badge">${p.role1}</span>` : ''}
          ${p.style ? `<span class="style-tag">${p.style}</span>` : ''}
          ${p.priority ? `<span style="font-size:13px">${stars(p.priority)}</span>` : ''}
        </div>
        <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">
          ${p.tm ? `<a class="tm-link" href="${p.tm}" target="_blank">🔗 Transfermarkt</a>` : ''}
          ${p.video ? `<a class="tm-link" href="${p.video}" target="_blank">🎬 Video</a>` : ''}
          ${p.tm ? `<button class="btn btn-ghost btn-sm" onclick="tmUpdate('${p.id}')">🔄 TM Sync</button>` : ''}
        </div>
      </div>
    </div>
    <div class="detail-stats-grid">
      <div class="detail-stat"><span class="ds-val val-green">${fmtMV(p.marketValue)}</span><div class="ds-lbl">Value</div></div>
      <div class="detail-stat"><span class="ds-val">${p.contract||'—'}</span><div class="ds-lbl">Contract</div></div>
      <div class="detail-stat"><span class="${'score-band '+scoreCls(p.score)}" style="font-size:15px">${p.score||'—'}</span><div class="ds-lbl" style="margin-top:6px">Score*</div></div>
      <div class="detail-stat"><span class="${'score-band '+scoreCls(p.potential)}" style="font-size:15px">${p.potential||'—'}</span><div class="ds-lbl" style="margin-top:6px">Potential*</div></div>
      <div class="detail-stat"><span class="ds-val">${p.physical||'—'}</span><div class="ds-lbl">Physical</div></div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);border-bottom:1px solid var(--border)">
      ${[['Games',p.games],['Goals',p.goals],['Assists',p.assists],['Minutes',p.minutes]].map(([l,v])=>
        '<div class="detail-stat" style="border-right:1px solid var(--border)">'
        + '<span class="ds-val mono">' + (v??'—') + '</span><div class="ds-lbl">' + l + '</div>'
        + '</div>').join('')}
    </div>

    <!-- DATA PROFILE SECTION -->
    <div id="data-profile-section" style="padding:18px 22px;border-bottom:1px solid var(--border)">
      <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted2);margin-bottom:14px">
        📊 Data Profile
        ${!serverOnline ? '<span style="color:var(--muted2);font-size:10px;font-weight:400;margin-left:8px">(Start server.py for live data)</span>' : ''}
      </div>
      <div id="profile-loading" style="color:var(--muted2);font-size:12px">${serverOnline ? '⏳ Loading...' : '—'}</div>
    </div>

    ${p.keynotes ? `<div style="padding:14px 22px;border-bottom:1px solid var(--border);background:var(--s2)">
      <div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--muted2);margin-bottom:6px">Key Notes</div>
      <div style="font-size:13px;color:var(--text)">${p.keynotes}</div>
    </div>` : ''}

    <!-- SIMILAR PLAYERS + GBE SECTION -->
    <!-- SIMILAR PLAYERS + GBE SECTION -->
    <div id="similar-gbe-section" style="border-bottom:1px solid var(--border)">
    <div id="similar-gbe-section" style="border-bottom:1px solid var(--border)">

      <!-- Similar Players -->
      <div style="padding:16px 22px 0">
        <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted2);margin-bottom:12px">🔍 Similar Players</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
          <div>
            <div style="font-size:10px;font-weight:600;color:var(--muted);margin-bottom:6px;letter-spacing:.04em">GLOBAL</div>
            <div id="similar-global" style="color:var(--muted2);font-size:11px">⏳ Loading...</div>
          </div>
          <div>
            <div style="font-size:10px;font-weight:600;color:var(--muted);margin-bottom:6px;letter-spacing:.04em">UK (ENG 1-3 · SCO 1)</div>
            <div id="similar-uk" style="color:var(--muted2);font-size:11px">⏳ Loading...</div>
          </div>
        </div>
      </div>

      <!-- GBE Calculator -->
      <div style="padding:16px 22px">
        <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted2);margin-bottom:10px">🧮 GBE Points (FA 2025/26)</div>
        <div id="gbe-display" style="color:var(--muted2);font-size:11px">⏳ Loading...</div>

        <details style="margin-top:10px">
          <summary style="font-size:11px;color:var(--muted);cursor:pointer;list-style:none;display:flex;align-items:center;gap:6px">
            <span style="background:var(--s3);border:1px solid var(--border);border-radius:5px;padding:3px 10px;font-size:10px">⚙ Adjust inputs</span>
          </summary>
          <div style="margin-top:10px;display:flex;flex-direction:column;gap:10px">

            <!-- Domestic -->
            <div style="font-size:10px;font-weight:700;color:var(--muted2);text-transform:uppercase;letter-spacing:.06em">Domestic</div>
            <label style="font-size:11px;color:var(--muted);display:flex;align-items:center;gap:6px;cursor:pointer">
              <input type="checkbox" id="gbe-is-youth" onchange="refreshGBE()" style="accent-color:var(--accent)"> Youth Player (U21)
            </label>
            <label id="gbe-debut-wrap" style="font-size:11px;color:var(--muted);display:none;align-items:center;gap:6px;cursor:pointer">
              <input type="checkbox" id="gbe-youth-debut" onchange="refreshGBE()" style="accent-color:var(--accent)"> Made first senior debut this period
            </label>

            <!-- International -->
            <div style="font-size:10px;font-weight:700;color:var(--muted2);text-transform:uppercase;letter-spacing:.06em;margin-top:4px">International (Table 1)</div>
            <label style="font-size:11px;color:var(--muted);display:flex;align-items:center;gap:6px;cursor:pointer">
              <input type="checkbox" id="gbe-use-intl" onchange="refreshGBE()" style="accent-color:var(--accent)"> Include senior international appearances
            </label>
            <div id="gbe-intl-inputs" style="display:none;grid-template-columns:1fr 1fr;gap:8px">
              <div>
                <div style="font-size:10px;color:var(--muted2);margin-bottom:3px">FIFA Ranking</div>
                <input type="number" id="gbe-intl-rank" value="50" min="1" max="200" onchange="refreshGBE()" class="form-input" style="font-size:11px;padding:4px 8px;height:28px">
              </div>
              <div>
                <div style="font-size:10px;color:var(--muted2);margin-bottom:3px">Apps %</div>
                <input type="number" id="gbe-intl-pct" value="0" min="0" max="100" step="5" onchange="refreshGBE()" class="form-input" style="font-size:11px;padding:4px 8px;height:28px">
              </div>
            </div>

            <!-- Continental mins -->
            <div style="font-size:10px;font-weight:700;color:var(--muted2);text-transform:uppercase;letter-spacing:.06em;margin-top:4px">Continental Minutes (Table 3)</div>
            <label style="font-size:11px;color:var(--muted);display:flex;align-items:center;gap:6px;cursor:pointer">
              <input type="checkbox" id="gbe-use-cont" onchange="refreshGBE()" style="accent-color:var(--accent)"> Add continental minutes
            </label>
            <div id="gbe-cont-inputs" style="display:none;grid-template-columns:1fr 1fr;gap:8px">
              <div>
                <div style="font-size:10px;color:var(--muted2);margin-bottom:3px">Competition Band</div>
                <select id="gbe-cont-band" onchange="refreshGBE()" class="filter-sel" style="font-size:11px">
                  <option value="1">Band 1 (UCL/CL)</option>
                  <option value="2">Band 2 (UEL/CWC)</option>
                  <option value="3">Band 3 (Other)</option>
                </select>
              </div>
              <div>
                <div style="font-size:10px;color:var(--muted2);margin-bottom:3px">Minutes %</div>
                <input type="number" id="gbe-cont-pct" value="0" min="0" max="100" step="5" onchange="refreshGBE()" class="form-input" style="font-size:11px;padding:4px 8px;height:28px">
              </div>
            </div>

            <!-- League position -->
            <div style="font-size:10px;font-weight:700;color:var(--muted2);text-transform:uppercase;letter-spacing:.06em;margin-top:4px">Final League Position (Table 4)</div>
            <label style="font-size:11px;color:var(--muted);display:flex;align-items:center;gap:6px;cursor:pointer">
              <input type="checkbox" id="gbe-use-finish" onchange="refreshGBE()" style="accent-color:var(--accent)"> Add final position points
            </label>
            <div id="gbe-finish-inputs" style="display:none">
              <select id="gbe-finish-cat" onchange="refreshGBE()" class="filter-sel" style="font-size:11px;width:100%">
                <option>Title winner</option><option>Band1 group / conf winner</option>
                <option>Band1 qualifiers</option><option>Band2 group</option>
                <option>Band2 qualifiers</option><option selected>Mid-table</option>
                <option>Relegation</option><option>Promotion</option>
              </select>
            </div>

            <!-- Continental progression -->
            <div style="font-size:10px;font-weight:700;color:var(--muted2);text-transform:uppercase;letter-spacing:.06em;margin-top:4px">Continental Progression (Table 5)</div>
            <label style="font-size:11px;color:var(--muted);display:flex;align-items:center;gap:6px;cursor:pointer">
              <input type="checkbox" id="gbe-use-cprog" onchange="refreshGBE()" style="accent-color:var(--accent)"> Add continental progression
            </label>
            <div id="gbe-cprog-inputs" style="display:none;grid-template-columns:1fr 1fr;gap:8px">
              <div>
                <div style="font-size:10px;color:var(--muted2);margin-bottom:3px">Competition Band</div>
                <select id="gbe-cprog-band" onchange="refreshGBE()" class="filter-sel" style="font-size:11px">
                  <option value="1">Band 1</option><option value="2">Band 2</option><option value="3">Band 3</option>
                </select>
              </div>
              <div>
                <div style="font-size:10px;color:var(--muted2);margin-bottom:3px">Stage reached</div>
                <select id="gbe-cprog-stage" onchange="refreshGBE()" class="filter-sel" style="font-size:11px">
                  <option>Final</option><option>Semi-final</option><option>Quarter-final</option>
                  <option>Round of 16</option><option>Round of 32 / KO PO</option>
                  <option>Group / league phase</option><option>Other</option>
                </select>
              </div>
            </div>

            <!-- League quality -->
            <div style="font-size:10px;font-weight:700;color:var(--muted2);text-transform:uppercase;letter-spacing:.06em;margin-top:4px">League Quality (Table 6)</div>
            <label style="font-size:11px;color:var(--muted);display:flex;align-items:center;gap:6px;cursor:pointer">
              <input type="checkbox" id="gbe-use-lq" checked onchange="refreshGBE()" style="accent-color:var(--accent)"> Include league quality points
            </label>

            <!-- ESC criteria -->
            <div id="gbe-esc-section" style="margin-top:4px">
              <div style="font-size:10px;font-weight:700;color:var(--muted2);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">ESC Criteria</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px">
                ${[['esc_youth_top50','Youth intl (Top-50)'],['esc_youth_outside','Youth intl (outside Top-50)'],
                   ['esc_youth_cont','Youth continental'],['esc_youth_dom','Domestic youth matches'],
                   ['esc_senior_top50','Senior intl (Top-50)'],['esc_senior_outside','Senior intl (outside Top-50)'],
                   ['esc_senior_cont','Senior continental'],['esc_senior_dom','Domestic senior (Band 1-5)']]
                  .map(([id,lbl]) => '<label style="font-size:10px;color:var(--muted);display:flex;align-items:center;gap:5px;cursor:pointer">'
                    + '<input type="checkbox" id="gbe-' + id + '" onchange="refreshGBE()" style="accent-color:var(--accent)"> ' + lbl + '</label>').join('')}
              </div>
            </div>

          </div>
        </details>
      </div>

    </div>

    <div class="notes-area">
      <div class="notes-header">
        <div class="notes-title">Scout Notes</div>
        <div style="font-size:11px;color:var(--muted2)">${(p.notes||[]).length} notes</div>
      </div>
      <div id="notes-list-${p.id}">
        ${(p.notes||[]).length === 0 ? `<div style="color:var(--muted2);font-size:12px;padding:8px 0">No notes yet.</div>` :
          (p.notes||[]).map(n => '<div class="note-item"><div class="note-text">' + n.text + '</div><div class="note-date">' + n.date + '</div></div>').join('')}
      </div>
      <div class="note-input-row">
        <input class="form-input" id="note-input-${p.id}" placeholder="Add scouting note..." onkeydown="if(event.key==='Enter')addNote('${p.id}')">
        <button class="btn btn-primary btn-sm" onclick="addNote('${p.id}')">Add</button>
      </div>
    </div>

    ${p.dataProfile ? `
    <div style="padding:10px 22px 14px;border-top:1px solid var(--border)">
      <button onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display==='none'?'block':'none';this.textContent=this.textContent.includes('View')?'▲ Hide Data Profile':'📋 View Data Profile'" style="background:var(--s3);border:1px solid var(--border);border-radius:7px;padding:7px 14px;font-size:12px;font-weight:600;color:var(--muted);cursor:pointer;width:100%;text-align:left">📋 View Data Profile</button>
      <div style="display:none;margin-top:10px">
        <img src="${p.dataProfile}" style="width:100%;border-radius:8px;border:1px solid var(--border)" />
        <div style="font-size:10px;color:var(--muted2);margin-top:6px;text-align:right"><a href="${p.dataProfile}" download="${(p.name||'player').replace(/ /g,'_')}_profile.png" style="color:var(--accent);text-decoration:none">⬇ Download</a></div>
      </div>
    </div>` : ''}

    <div style="padding:12px 22px 16px;display:flex;align-items:center;gap:10px;border-top:1px solid var(--border);flex-wrap:wrap">
      <span style="font-size:10px;color:var(--muted2)">Status:</span>
      <select class="filter-sel" style="font-size:11px" onchange="quickStatus('${p.id}',this.value)">
        ${['No Progress','Relationship','Agency Link','Contact Made'].map(s=>'<option value="' + s + '" ' + (s===(p.status||'')?'selected':'') + '>' + s + '</option>').join('')}
      </select>
      <select class="filter-sel" style="font-size:11px" onchange="quickWindow('${p.id}',this.value)">
        ${['Monitor','Summer','January','Signed'].map(w=>'<option value="' + w + '" ' + (w===(p.window||'')?'selected':'') + '>' + w + '</option>').join('')}
      </select>
      <span style="font-size:10px;color:var(--muted2);margin-left:auto">Added: ${p.createdAt ? new Date(p.createdAt).toLocaleDateString('en-GB') : '—'}</span>
    </div>
  `;
}

function buildCompositeScores(groups) {
  const pct = (group, label) => {
    const g = groups[group] || {};
    const d = g[label];
    return (d && d.pct != null) ? d.pct : null;
  };
  const out = {};

  // Detect if FB groups (has xA, Crosses) or CB groups
  const isFB = pct('ATT','xA') != null || pct('ATT','Crosses') != null;

  if (isFB) {
    // FB composite scores — Aerial, Ground, Carrying, Playmaking, Chance Creation, Retention
    const a1 = pct('DEF','Aerial Duels'), a2 = pct('DEF','Aerial Duel %');
    if (a1!=null && a2!=null) out['Aerial'] = Math.round(a1*0.30 + a2*0.70);

    const g1 = pct('DEF','Defensive Duels'), g2 = pct('DEF','Defensive Duel %');
    if (g1!=null && g2!=null) out['Ground'] = Math.round(g1*0.30 + g2*0.70);

    const c1 = pct('POS','Dribbles'), c2 = pct('POS','Dribbling %'), c3 = pct('POS','Progressive Runs') ?? pct('ATT','Progressive Runs');
    if (c1!=null && c2!=null && c3!=null) out['Carrying'] = Math.round(c1*0.40 + c2*0.20 + c3*0.40);

    const pm1 = pct('POS','Progressive Passes'), pm2 = pct('POS','Forward Passes'), pm3 = pct('POS','Passes to Final 3rd');
    if (pm1!=null && pm2!=null && pm3!=null) out['Playmaking'] = Math.round(pm1*0.50 + pm2*0.25 + pm3*0.25);

    const cc1 = pct('ATT','xA'), cc2 = pct('ATT','Crosses'), cc3 = pct('ATT','Touches in Box');
    if (cc1!=null && cc2!=null) out['Chance Creation'] = Math.round(
      (cc1!=null?cc1*0.60:0) + (cc2!=null?cc2*0.20:0) + (cc3!=null?cc3*0.20:0)
    );

    const r1 = pct('POS','Passing %'), r2 = pct('POS','Forward Pass %'), r3 = pct('POS','Prog Pass %');
    if (r1!=null && r2!=null && r3!=null) out['Retention'] = Math.round(r1*0.34 + r2*0.33 + r3*0.33);
  } else {
    // CB composite scores
    const a1 = pct('DEF','Aerial Duels'), a2 = pct('DEF','Aerial Duel Success %');
    if (a1!=null && a2!=null) out['Aerial'] = Math.round(a1*0.30 + a2*0.70);
    const g1 = pct('DEF','Defensive Duels'), g2 = pct('DEF','Defensive Duel Success %');
    if (g1!=null && g2!=null) out['Ground'] = Math.round(g1*0.30 + g2*0.70);
    const pos1 = pct('DEF','PAdj Interceptions'), pos2 = pct('DEF','Shots Blocked');
    if (pos1!=null && pos2!=null) out['Positioning'] = Math.round(pos1*0.70 + pos2*0.30);
    const c1 = pct('POS','Dribbles'), c2 = pct('POS','Dribbling %'), c3 = pct('POS','Progressive Runs');
    if (c1!=null && c2!=null && c3!=null) out['Carrying'] = Math.round(c1*0.40 + c2*0.20 + c3*0.40);
    const pm1 = pct('POS','Progressive Passes'), pm2 = pct('POS','Forward Passes'), pm3 = pct('POS','Passes to Final 3rd');
    if (pm1!=null && pm2!=null && pm3!=null) out['Playmaking'] = Math.round(pm1*0.50 + pm2*0.25 + pm3*0.25);
    const r1 = pct('POS','Passing Accuracy %'), r2 = pct('POS','Forward Passing %'), r3 = pct('POS','Progressive Passing %'), r4 = pct('POS','Long Passing %');
    if (r1!=null && r2!=null && r3!=null && r4!=null) out['Retention'] = Math.round(r1*0.25 + r2*0.25 + r3*0.25 + r4*0.25);
  }
  return out;
}

function renderDataProfile(data) {
  const { percentile_groups, roles, tags, raw } = data;

  // Inject height chip from server raw data
  if (raw?.height && raw.height !== 'nan' && raw.height !== '') {
    const hEl = document.getElementById('detail-height-chip');
    if (hEl) hEl.outerHTML = '<div class="detail-meta-chip">📏 ' + raw.height + ' cm</div>';
  }

  // Inject nationality from server — show flags, merge if same country
  if (raw?.birth_country || raw?.passport_country) {
    const bc = (raw.birth_country || '').trim();
    const pc = (raw.passport_country || '').trim();
    const natEl = document.getElementById('detail-nat-chip');
    if (natEl) {
      let html = '';
      if (bc && pc && bc.toLowerCase() === pc.toLowerCase()) {
        // Same — show once with flag
        html = '<div class="detail-meta-chip">' + countryFlagHtml(bc) + ' ' + bc + '</div>';
      } else {
        if (bc) html += '<div class="detail-meta-chip">' + countryFlagHtml(bc) + ' ' + bc + '</div>';
        if (pc && pc !== bc) html += '<div class="detail-meta-chip">🛂 ' + countryFlagHtml(pc) + ' ' + pc + '</div>';
      }
      natEl.outerHTML = html;
    }
  }
  const el = document.getElementById('profile-loading');
  if (!el) return;
  const groups = percentile_groups || {};
  const hasMetrics = Object.keys(groups).some(g => Object.keys(groups[g]||{}).length > 0);
  const hasRoles = roles && Object.keys(roles).length > 0;
  if (!hasRoles && !hasMetrics) {
    el.innerHTML = `<div style="color:var(--muted2);font-size:12px">Player not found in CSV — add data manually.</div>`;
    return;
  }
  let html = `<div style="display:flex;gap:22px;align-items:flex-start;flex-wrap:wrap">`;
  // Composite radar
  const composites = buildCompositeScores(groups);
  if (Object.keys(composites).length >= 3) {
    html += `<div style="display:flex;flex-direction:column;align-items:center;gap:4px">
      ${renderRadarSVG(composites)}
      <div style="font-size:9px;color:var(--muted2);letter-spacing:.04em">vs CB peers · same league · 500+ mins</div>
    </div>`;
  }
  // Role scores + tags
  html += `<div style="flex:1;min-width:180px">`;
  if (hasRoles) {
    html += `<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--muted2);margin-bottom:10px">Role Scores</div>`;
    Object.entries(roles).forEach(([role, score]) => {
      const tc = roleScoreTextColor(score);
      const w = score != null ? score : 0;
      html += `<div style="margin-bottom:8px">
        <div style="display:flex;justify-content:space-between;margin-bottom:3px">
          <span style="font-size:11px;color:var(--muted)">${role}</span>
          <span style="font-family:'DM Mono',monospace;font-size:11px;color:${tc};font-weight:600">${score!=null?score.toFixed(1):'—'}</span>
        </div>
        <div style="height:5px;background:var(--s4);border-radius:3px;overflow:hidden">
          <div style="height:100%;width:${w}%;background:${tc};border-radius:3px;transition:width 0.6s ease"></div>
        </div>
      </div>`;
    });
  }
  if (tags) {
    if (tags.styles?.length) {
      html += `<div style="margin-top:12px;margin-bottom:4px;font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--muted2)">Style</div>`;
      html += tags.styles.map(t=>`<span style="display:inline-block;background:rgba(99,179,237,0.15);color:#63b3ed;border:1px solid rgba(99,179,237,0.3);padding:3px 8px;border-radius:4px;font-size:11px;margin:2px">${t}</span>`).join('');
    }
    if (tags.strengths?.length) {
      html += `<div style="margin-top:8px;margin-bottom:4px;font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--muted2)">Strengths</div>`;
      html += tags.strengths.map(t=>`<span style="display:inline-block;background:rgba(52,211,153,0.15);color:#34d399;border:1px solid rgba(52,211,153,0.3);padding:3px 8px;border-radius:4px;font-size:11px;margin:2px">${t}</span>`).join('');
    }
    if (tags.weaknesses?.length) {
      html += `<div style="margin-top:8px;margin-bottom:4px;font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--muted2)">Weaknesses</div>`;
      html += tags.weaknesses.map(t=>`<span style="display:inline-block;background:rgba(252,165,165,0.15);color:#fca5a5;border:1px solid rgba(252,165,165,0.3);padding:3px 8px;border-radius:4px;font-size:11px;margin:2px">${t}</span>`).join('');
    }
  }
  html += `</div></div>`;

  // Percentile bars — paired order, position-aware
  const isFBBoard = currentBoard === 'RB' || currentBoard === 'LB';
  const ORDERED = isFBBoard ? [
    { group:'ATT', label:'⚡ Attacking', color:'#f97316', keys:[
      'Crosses', 'Crossing %',
      'Goals', 'Shots',
      'xA', 'xG',
      'Offensive Duels', 'Offensive Duel %',
      'Progressive Runs', 'Accelerations',
      'Touches in Box',
    ]},
    { group:'DEF', label:'🛡 Defensive', color:'#60a5fa', keys:[
      'Aerial Duels', 'Aerial Duel %',
      'Defensive Duels', 'Defensive Duel %',
      'PAdj Interceptions', 'Shots Blocked',
    ]},
    { group:'POS', label:'🎯 Possession', color:'#34d399', keys:[
      'Dribbles', 'Dribbling %',
      'Forward Passes', 'Forward Pass %',
      'Long Passes', 'Long Pass %',
      'Passes', 'Passing %',
      'Passes to Final 3rd', 'Passes to Final 3rd %',
      'Passes to Pen Area', 'Passes to Pen Area %',
      'Progressive Passes', 'Prog Pass %',
      'Smart Passes', 'Deep Completions',
    ]},
  ] : [
    { group:'ATT', label:'⚡ Attacking', color:'#f97316', keys:[
      'xG', 'Goals: Non-Penalty',
      'Offensive Duels', 'Offensive Duel Success %',
      'Progressive Runs', 'Accelerations'
    ]},
    { group:'DEF', label:'🛡 Defensive', color:'#60a5fa', keys:[
      'Aerial Duels', 'Aerial Duel Success %',
      'Defensive Duels', 'Defensive Duel Success %',
      'PAdj Interceptions', 'Shots Blocked'
    ]},
    { group:'POS', label:'🎯 Possession', color:'#34d399', keys:[
      'Dribbles', 'Dribbling %',
      'Forward Passes', 'Forward Passing %',
      'Long Passes', 'Long Passing %',
      'Passes', 'Passing Accuracy %',
      'Passes to Final 3rd', 'Passes to Final 3rd %',
      'Progressive Passes', 'Progressive Passing %',
    ]},
  ];

  if (hasMetrics) {
    html += `<div style="margin-top:18px;border-top:1px solid var(--border);padding-top:16px">`;
    ORDERED.forEach(({ group, label, color, keys }) => {
      const grp = groups[group] || {};
      const entries = keys.map(k => [k, grp[k]]).filter(([,d]) => d && d.pct != null);
      if (!entries.length) return;
      html += `<div style="margin-bottom:18px">
        <div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:${color};margin-bottom:8px">${label}</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 28px">`;
      entries.forEach(([lbl, d]) => {
        const col = pctColor(d.pct);
        html += `<div style="display:flex;align-items:center;gap:6px">
          <div style="width:126px;font-size:10px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex-shrink:0" title="${lbl}">${lbl}</div>
          <div style="flex:1;height:4px;background:var(--s4);border-radius:2px;overflow:hidden">
            <div style="height:100%;width:${d.pct}%;background:${col};border-radius:2px"></div>
          </div>
          <div style="font-family:'DM Mono',monospace;font-size:10px;color:${col};width:26px;text-align:right;flex-shrink:0">${Math.round(d.pct)}</div>
        </div>`;
      });
      html += `</div></div>`;
    });
    html += `</div>`;
  }
  el.innerHTML = html;
}


async function loadPlayerProfile(p) {
  if (!serverOnline) { serverOnline = await checkServer(); }
  if (!serverOnline) {
    const el = document.getElementById('profile-loading');
    if (el) el.innerHTML = '<span style="color:var(--muted2);font-size:11px">Server offline — run server.py</span>';
    return;
  }
  try {
    const url = `${SERVER}/api/player/profile?player=${encodeURIComponent(p.name||'')}&team=${encodeURIComponent(p.team||'')}&league=${encodeURIComponent(p.league||'')}&pos=${encodeURIComponent(currentBoard||'')}`;
    const r = await fetch(url);
    if (r.ok) renderDataProfile(await r.json());
  } catch(e) {
    const el = document.getElementById('profile-loading');
    if (el) el.innerHTML = '<span style="color:var(--muted2);font-size:11px">Server offline — run server.py</span>';
  }
  loadSimilar(p);
  loadGBE(p);
}

// Store current player for GBE refresh
let _gbePlayer = null;

async function loadSimilar(p) {
  const elG = document.getElementById('similar-global');
  const elUK = document.getElementById('similar-uk');
  if (!elG || !elUK) return;
  const base = `${SERVER}/api/similar?player=${encodeURIComponent(p.name||'')}&team=${encodeURIComponent(p.team||'')}&league=${encodeURIComponent(p.league||'')}&pos=${encodeURIComponent(currentBoard||'')}`;
  try {
    const [rG, rUK] = await Promise.all([fetch(base), fetch(base + '&uk=1')]);
    const [dG, dUK] = await Promise.all([rG.json(), rUK.json()]);
    elG.innerHTML  = renderSimilarList(dG.results  || []);
    elUK.innerHTML = renderSimilarList(dUK.results || []);
  } catch(e) {
    elG.innerHTML = elUK.innerHTML = '<span style="color:var(--muted2)">—</span>';
  }
}

function renderSimilarList(results) {
  if (!results.length) return '<span style="color:var(--muted2);font-size:11px">No matches found</span>';
  return results.map((r, i) => {
    const simColor = r.similarity >= 80 ? '#34d399' : r.similarity >= 65 ? '#fbbf24' : '#6b7280';
    return `<div style="display:flex;align-items:center;gap:8px;padding:5px 0;${i<results.length-1?'border-bottom:1px solid var(--border)':''}">
      <span style="font-size:10px;font-weight:700;color:var(--muted2);width:14px">${i+1}</span>
      <div style="flex:1;min-width:0">
        <div style="font-size:12px;font-weight:600;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${r.name}</div>
        <div style="font-size:10px;color:var(--muted2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${r.team} · ${r.league}</div>
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:11px;font-weight:700;color:${simColor};flex-shrink:0">${r.similarity.toFixed(0)}%</div>
    </div>`;
  }).join('');
}

async function loadGBE(p) {
  _gbePlayer = p;
  const el = document.getElementById('gbe-display');
  if (!el) return;
  const g = id => document.getElementById(id);
  const chk = id => g(id)?.checked ? '1' : '0';
  const val = id => g(id)?.value || '0';

  const params = new URLSearchParams({
    player: p.name||'', team: p.team||'', league: p.league||'', pos: currentBoard||'',
    use_intl:    chk('gbe-use-intl'),
    intl_rank:   val('gbe-intl-rank') || '200',
    intl_pct:    val('gbe-intl-pct')  || '0',
    use_cont:    chk('gbe-use-cont'),
    cont_band:   val('gbe-cont-band') || '1',
    cont_pct:    val('gbe-cont-pct')  || '0',
    use_finish:  chk('gbe-use-finish'),
    finish_cat:  val('gbe-finish-cat') || 'Mid-table',
    use_cprog:   chk('gbe-use-cprog'),
    cprog_band:  val('gbe-cprog-band') || '1',
    cprog_stage: val('gbe-cprog-stage') || 'Other',
    is_youth:    chk('gbe-is-youth'),
    youth_debut: chk('gbe-youth-debut'),
    use_lq:      chk('gbe-use-lq') || '1',
    esc_youth_top50:    chk('gbe-esc_youth_top50'),
    esc_youth_outside:  chk('gbe-esc_youth_outside'),
    esc_youth_cont:     chk('gbe-esc_youth_cont'),
    esc_youth_dom:      chk('gbe-esc_youth_dom'),
    esc_senior_top50:   chk('gbe-esc_senior_top50'),
    esc_senior_outside: chk('gbe-esc_senior_outside'),
    esc_senior_cont:    chk('gbe-esc_senior_cont'),
    esc_senior_dom:     chk('gbe-esc_senior_dom'),
  });
  try {
    const r = await fetch(`${SERVER}/api/gbe?${params}`);
    const d = await r.json();
    if (d.error) { el.innerHTML = `<span style="color:#fca5a5;font-size:11px">GBE error: ${d.error}</span>`; return; }
    const b = d.breakdown;
    const escHtml = d.esc_eligible && d.esc_reasons?.length
      ? `<div style="margin-top:5px;font-size:10px;color:#fbbf24">ESC: ${d.esc_reasons.join(' · ')}</div>` : '';
    el.innerHTML = `
      <div style="background:var(--s2);border:1px solid var(--border);border-radius:10px;padding:12px 16px">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
          <div>
            <div style="font-size:2rem;font-weight:800;line-height:1;color:var(--text)">${d.total}</div>
            <div style="font-size:9px;color:var(--muted2);margin-top:2px">Est. points</div>
          </div>
          <div style="padding:5px 12px;border-radius:999px;background:${d.status_color};color:#fff;font-weight:700;font-size:12px;white-space:nowrap">${d.status}</div>
          <div style="margin-left:auto;text-align:right">
            <div style="font-size:11px;color:var(--muted)">Band ${d.band}</div>
            <div style="font-size:10px;color:var(--muted2)">${d.domestic_pct}% mins</div>
          </div>
        </div>
        <div style="height:1px;background:var(--border);margin:6px 0"></div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px">
          ${[['Domestic',b.domestic],['Intl',b.international],['Continental',b.continental],
             ['League Pos',b.league_position],['Cont. Prog',b.cont_progression],['League Qual',b.league_quality]]
            .map(([l,v])=>`<div style="font-size:10px;color:var(--muted2)">${l}: <span style="color:var(--text);font-weight:600">${v}</span></div>`).join('')}
        </div>
        <div style="margin-top:5px;font-size:9px;color:var(--muted2)">0–9 Fail · 10–14 Exceptions Panel · 15+ Pass</div>
        ${escHtml}
      </div>`;
  } catch(e) {
    el.innerHTML = '<span style="color:var(--muted2)">Could not load GBE data</span>';
  }
}

function refreshGBE() {
  if (!_gbePlayer) return;
  // Show/hide conditional inputs
  const show = (id, visible) => { const el = document.getElementById(id); if(el) el.style.display = visible ? 'grid' : 'none'; };
  const showB = (id, visible) => { const el = document.getElementById(id); if(el) el.style.display = visible ? 'block' : 'none'; };
  show('gbe-intl-inputs',  document.getElementById('gbe-use-intl')?.checked);
  show('gbe-cont-inputs',  document.getElementById('gbe-use-cont')?.checked);
  showB('gbe-finish-inputs', document.getElementById('gbe-use-finish')?.checked);
  show('gbe-cprog-inputs', document.getElementById('gbe-use-cprog')?.checked);
  const isYouth = document.getElementById('gbe-is-youth')?.checked;
  const debutWrap = document.getElementById('gbe-debut-wrap');
  if (debutWrap) debutWrap.style.display = isYouth ? 'flex' : 'none';
  loadGBE(_gbePlayer);
}

async function tmUpdate(id) {
  const p = players().find(x => x.id === id);
  if (!p || !p.tm) return;
  showToast('🔄 Syncing from Transfermarkt...');
  try {
    const r = await fetch(`${SERVER}/api/player/tm_update?url=${encodeURIComponent(p.tm)}`);
    const data = await r.json();
    if (data.error) { showToast('❌ ' + data.error); return; }
    const idx = allPlayers[currentBoard].findIndex(x => x.id === id);
    if (idx !== -1) {
      if (data.team)        allPlayers[currentBoard][idx].team = data.team;
      if (data.age)         allPlayers[currentBoard][idx].age = data.age;
      if (data.value)       allPlayers[currentBoard][idx].marketValue = data.value;
      if (data.contract)    allPlayers[currentBoard][idx].contract = data.contract;
      if (data.recentmove)  allPlayers[currentBoard][idx].recentmove = data.recentmove;
      if (data.league)      allPlayers[currentBoard][idx].league = data.league;
      if (data.games)       allPlayers[currentBoard][idx].games = data.games;
      if (data.goals)       allPlayers[currentBoard][idx].goals = data.goals;
      if (data.assists)     allPlayers[currentBoard][idx].assists = data.assists;
      if (data.minutes)     allPlayers[currentBoard][idx].minutes = data.minutes;
      allPlayers[currentBoard][idx].updatedAt = new Date().toISOString();
      save();
      renderAll();
      showToast('✅ TM sync complete');
      openDetail(id);
    }
  } catch(e) {
    showToast('❌ Server offline — run server.py');
  }
}

function openDetail(id) {
  viewingId = id;
  const p = players().find(x => x.id === id);
  if (!p) return;
  document.getElementById('detail-title').textContent = (p.fullname || p.name || 'PLAYER').toUpperCase();
  document.getElementById('detail-body').innerHTML = renderDetailStatic(p);
  document.getElementById('detail-overlay').classList.add('open');
  loadPhoto(p.name, p.team, 'detail-photo-wrap', p.league);
  // Load team badge from FotMob
  if (p.team) {
    const teamId = getTeamFotmobId(p.team);
    if (teamId) {
      const badgeEl = document.getElementById('detail-team-badge');
      if (badgeEl) {
        badgeEl.src = `https://images.fotmob.com/image_resources/logo/teamlogo/${teamId}.png`;
        badgeEl.onload = () => {
          badgeEl.style.display = 'block';
          const emoji = document.getElementById('detail-team-emoji');
          if (emoji) emoji.style.display = 'none';
        };
        badgeEl.onerror = () => { badgeEl.style.display = 'none'; };
      }
    }
  }
  checkServer().then(() => loadPlayerProfile(p));
}

function closeDetail() { document.getElementById('detail-overlay').classList.remove('open'); viewingId = null; }

function editFromDetail() {
  const id = viewingId;
  closeDetail();
  if (id) setTimeout(() => openEdit(id), 150);
}

function addNote(id) {
  const el = document.getElementById('note-input-'+id);
  const text = el.value.trim();
  if (!text) return;
  const p = allPlayers[currentBoard].find(x => x.id === id);
  if (!p) return;
  const note = { text, date: new Date().toLocaleString('en-GB', {day:'2-digit',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit'}) };
  p.notes = [...(p.notes||[]), note];
  p.updatedAt = new Date().toISOString();
  save();
  el.value = '';
  const nl = document.getElementById('notes-list-'+id);
  if (nl) nl.innerHTML = p.notes.map(n=>`<div class="note-item"><div class="note-text">${n.text}</div><div class="note-date">${n.date}</div></div>`).join('');
  showToast('📝 Note added');
}

function quickStatus(id, val) {
  const p = allPlayers[currentBoard].find(x => x.id === id);
  if (!p) return;
  p.status = val;
  p.updatedAt = new Date().toISOString();
  save();
  renderAll();
  showToast('✅ Status → ' + val);
}

function quickWindow(id, val) {
  const p = allPlayers[currentBoard].find(x => x.id === id);
  if (!p) return;
  p.window = val;
  p.updatedAt = new Date().toISOString();
  save();
  renderAll();
  showToast('✅ Window → ' + val);
}

// ── EXPORT ──
function exportCSV() {
  const cols = ['name','fullname','team','league','pos','age','foot','nationality',
    'window','target','status','agency','recentmove','priority',
    'style','role1','score','potential','physical',
    'marketValue','contract','games','goals','assists','minutes','goalsAgainst',
    'tm','video','keynotes','createdAt'];
  const rows = [cols.join(',')];
  players().forEach(p => {
    rows.push(cols.map(c => `"${(p[c]||'').toString().replace(/"/g,'""')}"`).join(','));
  });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([rows.join('\n')], {type:'text/csv'}));
  a.download = currentBoard.replace(/ /g,'_') + '_' + new Date().toISOString().slice(0,10) + '.csv';
  a.click();
  showToast('📊 Exported: ' + currentBoard);
}

// ── IMPORT ──
function openImport() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.csv,.xlsx,.xls';
  input.onchange = e => {
    const file = e.target.files[0];
    if (!file) return;

    if (file.name.endsWith('.xlsx') || file.name.endsWith('.xls')) {
      importExcel(file);
    } else {
      importCSV(file);
    }
  };
  input.click();
}

function parseMondayRow(row) {
  // Extract TM link from Monday's "TM Link - https://..." format
  function extractTM(val) {
    if (!val) return '';
    const m = String(val).match(/https?:\/\/[^\s,]+transfermarkt[^\s,]*/i);
    return m ? m[0] : '';
  }

  // Clean value
  function clean(val) {
    if (val === null || val === undefined) return '';
    return String(val).trim();
  }

  // Extract numeric market value — keep as raw number
  function parseMV(val) {
    if (!val) return '';
    const s = String(val).replace(/[€£$\s]/g,'').toLowerCase();
    const n = parseFloat(s);
    if (isNaN(n)) return '';
    if (s.includes('m')) return Math.round(n * 1000000);
    if (s.includes('k')) return Math.round(n * 1000);
    return n;
  }

  // Map Monday column names → ScoutBoard field names
  const name = clean(row['Name'] || row['name']);
  if (!name) return null;

  return {
    id: uid(),
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    notes: [],
    name,
    team:        clean(row['Team'] || row['team']),
    league:      clean(row['League'] || row['league']),
    age:         clean(row['Age'] || row['age']),
    window:      clean(row['Window'] || row['window']),
    target:      clean(row['Target Move'] || row['target']),
    tm:          extractTM(row['Transfermarkt'] || row['tm']),
    style:       clean(row['Style'] || row['style']),
    role1:       clean(row['Roles'] || row['role1']),
    priority:    clean(row['Priority'] || row['priority']),
    status:      clean(row['Status'] || row['status']),
    agency:      clean(row['Agency'] || row['agency']),
    recentmove:  clean(row['Recent Move'] || row['recentmove']),
    marketValue: parseMV(row['Value'] || row['marketValue']),
    contract:    clean(row['Contract'] || row['contract']),
    foot:        clean(row['Foot'] || row['foot']),
    score:       clean(row['Score*'] || row['score']),
    potential:   clean(row['Potential*'] || row['potential']),
    physical:    clean(row['Physical Notes'] || row['physical']),
    video:       clean(row['Video'] || row['video']),
    keynotes:    clean(row['Extra Key Notes'] || row['keynotes']),
    games:       clean(row['Games'] || row['games']),
    goals:       clean(row['Goals'] || row['goals']),
    assists:     clean(row['Assists'] || row['assists']),
    minutes:     clean(row['Minutes'] || row['minutes']),
  };
}

async function importExcel(file) {
  showToast('📥 Reading Excel file...');
  // Use SheetJS to parse xlsx
  const script = document.createElement('script');
  script.src = 'https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js';
  script.onload = () => {
    const reader = new FileReader();
    reader.onload = ev => {
      try {
        const wb = XLSX.read(ev.target.result, {type:'array'});
        const ws = wb.Sheets[wb.SheetNames[0]];
        const raw = XLSX.utils.sheet_to_json(ws, {header:1, defval:''});

        // Find header row (look for 'Name' column)
        let headerIdx = -1;
        for (let i = 0; i < Math.min(raw.length, 10); i++) {
          if (raw[i].includes('Name')) { headerIdx = i; break; }
        }
        if (headerIdx === -1) { showToast('❌ Could not find header row'); return; }

        const headers = raw[headerIdx];
        const imported = [];
        for (let i = headerIdx + 1; i < raw.length; i++) {
          const rowArr = raw[i];
          if (!rowArr || !rowArr[0]) continue;
          const row = {};
          headers.forEach((h, j) => { row[h] = rowArr[j]; });
          const p = parseMondayRow(row);
          if (p) imported.push(p);
        }

        if (imported.length === 0) { showToast('❌ No players found'); return; }

        // Ask user: replace or merge?
        const replace = confirm(`Import ${imported.length} players.\n\nOK = Replace current ${currentBoard} list\nCancel = Merge with existing`);
        if (replace) {
          allPlayers[currentBoard] = imported;
        } else {
          // Merge — skip duplicates by name+team
          const existing = new Set(allPlayers[currentBoard].map(p => p.name+'|'+p.team));
          const newOnes = imported.filter(p => !existing.has(p.name+'|'+p.team));
          allPlayers[currentBoard] = [...allPlayers[currentBoard], ...newOnes];
        }
        save();
        renderAll();
        showToast(`✅ Imported ${imported.length} players to ${currentBoard}`);
      } catch(err) {
        showToast('❌ Error reading file: ' + err.message);
      }
    };
    reader.readAsArrayBuffer(file);
  };
  script.onerror = () => showToast('❌ Could not load Excel reader');
  if (!window.XLSX) {
    document.head.appendChild(script);
  } else {
    script.onload();
  }
}

function importCSV(file) {
  const reader = new FileReader();
  reader.onload = ev => {
    try {
      const text = ev.target.result;
      const lines = text.split('\n').filter(l => l.trim());

      // Find header row
      let headerIdx = 0;
      for (let i = 0; i < Math.min(lines.length, 5); i++) {
        if (lines[i].includes('Name')) { headerIdx = i; break; }
      }

      function parseCSVLine(line) {
        const result = [];
        let cur = '', inQ = false;
        for (let c of line) {
          if (c === '"') inQ = !inQ;
          else if (c === ',' && !inQ) { result.push(cur); cur = ''; }
          else cur += c;
        }
        result.push(cur);
        return result;
      }

      const headers = parseCSVLine(lines[headerIdx]).map(h => h.trim().replace(/^"|"$/g,''));
      const imported = [];
      for (let i = headerIdx + 1; i < lines.length; i++) {
        const vals = parseCSVLine(lines[i]);
        const row = {};
        headers.forEach((h, j) => { row[h] = (vals[j]||'').replace(/^"|"$/g,'').trim(); });
        const p = parseMondayRow(row);
        if (p) imported.push(p);
      }

      if (imported.length === 0) { showToast('❌ No players found'); return; }

      const replace = confirm(`Import ${imported.length} players.\n\nOK = Replace current ${currentBoard} list\nCancel = Merge with existing`);
      if (replace) {
        allPlayers[currentBoard] = imported;
      } else {
        const existing = new Set(allPlayers[currentBoard].map(p => p.name+'|'+p.team));
        const newOnes = imported.filter(p => !existing.has(p.name+'|'+p.team));
        allPlayers[currentBoard] = [...allPlayers[currentBoard], ...newOnes];
      }
      save();
      renderAll();
      showToast(`✅ Imported ${imported.length} players to ${currentBoard}`);
    } catch(err) {
      showToast('❌ Error: ' + err.message);
    }
  };
  reader.readAsText(file);
}

// ── TOAST ──
async function syncAll() {
  if (!serverOnline) {
    serverOnline = await checkServer();
    if (!serverOnline) { showToast('❌ Server offline'); return; }
  }

  // Get all players with TM links across current board
  const list = players().filter(p => p.tm && p.tm.trim());
  if (list.length === 0) { showToast('No players with TM links on this board'); return; }

  if (!confirm(`Sync ${list.length} players from Transfermarkt?\n\nThis will take ~${Math.ceil(list.length * 3 / 60)} minutes. Don't close the tab.`)) return;

  // Show progress overlay
  const overlay = document.createElement('div');
  overlay.id = 'sync-overlay';
  overlay.style.cssText = `position:fixed;inset:0;background:rgba(7,9,15,0.85);z-index:9999;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;font-family:'DM Sans',sans-serif`;
  overlay.innerHTML = `
    <div style="font-size:18px;font-weight:700;color:#fff;letter-spacing:1px">🔄 SYNCING FROM TRANSFERMARKT</div>
    <div id="sync-player-name" style="font-size:13px;color:var(--muted);min-height:20px"></div>
    <div style="width:360px;height:6px;background:#1a2235;border-radius:3px;overflow:hidden">
      <div id="sync-bar" style="height:100%;width:0%;background:var(--accent);border-radius:3px;transition:width 0.3s ease"></div>
    </div>
    <div id="sync-status" style="font-size:12px;color:var(--muted2)">0 / ${list.length}</div>
    <div id="sync-results" style="font-size:11px;color:#4ade80;margin-top:4px"></div>
    <button onclick="document.getElementById('sync-overlay').remove()" style="margin-top:8px;padding:6px 16px;background:transparent;border:1px solid var(--border);color:var(--muted);border-radius:6px;cursor:pointer;font-size:11px">Cancel (data saved so far)</button>
  `;
  document.body.appendChild(overlay);

  let updated = 0, failed = 0, cancelled = false;
  const btn = document.getElementById('sync-all-btn');
  if (btn) btn.disabled = true;

  for (let i = 0; i < list.length; i++) {
    // Check if cancelled
    if (!document.getElementById('sync-overlay')) { cancelled = true; break; }

    const p = list[i];
    const pct = Math.round((i / list.length) * 100);
    document.getElementById('sync-bar').style.width = pct + '%';
    document.getElementById('sync-status').textContent = `${i + 1} / ${list.length}`;
    document.getElementById('sync-player-name').textContent = `${p.name} — ${p.team}`;

    try {
      const r = await fetch(`${SERVER}/api/player/tm_update?url=${encodeURIComponent(p.tm)}`);
      const data = await r.json();

      if (!data.error) {
        const idx = allPlayers[currentBoard].findIndex(x => x.id === p.id);
        if (idx !== -1) {
          const player = allPlayers[currentBoard][idx];
          if (data.team)       player.team       = data.team;
          if (data.age)        player.age        = data.age;
          if (data.value)      player.marketValue = data.value;
          if (data.contract)   player.contract   = data.contract;
          if (data.recentmove) player.recentmove = data.recentmove;
          if (data.league)     player.league     = data.league;
          if (data.games)      player.games      = data.games;
          if (data.goals)      player.goals      = data.goals;
          if (data.assists)    player.assists    = data.assists;
          if (data.minutes)    player.minutes    = data.minutes;
          player.updatedAt = new Date().toISOString();
          updated++;
        }
      } else {
        failed++;
      }
    } catch(e) {
      failed++;
    }

    // Update results counter
    document.getElementById('sync-results').textContent =
      `✅ ${updated} updated  ${failed > 0 ? `❌ ${failed} failed` : ''}`;

    // Save every 10 players
    if (i % 10 === 0) await save();

    // Delay to avoid rate limiting
    await new Promise(r => setTimeout(r, 2500));
  }

  await save();
  renderAll();
  if (btn) btn.disabled = false;

  const syncOv = document.getElementById('sync-overlay');
  if (syncOv) {
    document.getElementById('sync-bar').style.width = '100%';
    document.getElementById('sync-player-name').textContent = cancelled ? 'Cancelled' : 'Complete!';
    document.getElementById('sync-status').textContent = `${list.length} / ${list.length}`;
    setTimeout(() => syncOv.remove(), 3000);
  }

  showToast(`✅ Sync complete: ${updated} updated, ${failed} failed`);
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}

// ── CLOSE ON OVERLAY CLICK ──
document.getElementById('add-overlay').addEventListener('click', e => { if (e.target === document.getElementById('add-overlay')) closeAdd(); });
document.getElementById('detail-overlay').addEventListener('click', e => { if (e.target === document.getElementById('detail-overlay')) closeDetail(); });

// ── INIT ──
async function init() {
  // Reset filters
  document.getElementById('windowFilter').value = '';
  document.getElementById('targetFilter').value = '';
  document.getElementById('statusFilter').value = '';
  document.getElementById('agencyFilter').value = '';
  document.getElementById('footFilter').value = '';
  document.getElementById('styleFilter').value = '';
  document.getElementById('valueFilter').value = '';
  document.getElementById('contractFilter').value = '';

  // Render tabs immediately with correct boards
  renderTabs();
  renderAll();

  // Then load data from server in background
  const fromServer = await loadFromServer();
  if (!fromServer) {
    const saved = localStorage.getItem(DATA_KEY);
    if (saved) {
      try {
        const data = JSON.parse(saved);
        CORRECT_BOARDS.forEach(b => { allPlayers[b] = data[b] || []; });
      } catch(e) {}
    }
  }
  renderTabs();
  renderAll();
}
init();
</script>
</body>
</html>











































































































































































































































