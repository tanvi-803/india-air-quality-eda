"""Builds an interactive standalone HTML5 Web Dashboard for Air Quality EDA & ML.
Embeds all images as base64 data URIs so it is 100% self-contained and never has missing images.
Uses free Esri basemap tiles with no API key requirement.

Visual design: "Frosted Monsoon" — pale sky gradient with drifting colour fields behind
frosted-glass cards; Source Serif 4 headings, Inter body text, IBM Plex Mono numbers.
"""

import os
import sys
import json
import base64
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import PROCESSED_DATA_DIR, TABLES_DIR, OUTPUTS_DIR, FIGURES_DIR, load_cities

def get_base64_image(filename):
    img_path = FIGURES_DIR / filename
    if img_path.exists():
        encoded = base64.b64encode(img_path.read_bytes()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    return ""


# ---------------------------------------------------------------------------
# HTML template. Placeholders of the form %%NAME%% are filled in by
# build_interactive_dashboard(); everything else is literal HTML/CSS/JS, so
# braces do not need escaping.
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Air Quality Analytics & Forecasting Dashboard — IT 462</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,500;0,8..60,600;0,8..60,700;1,8..60,500&display=swap">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>document.documentElement.classList.add('js');</script>
<style>
/* ===== Frosted Monsoon tokens ===== */
:root{
  --card:rgba(255,255,255,.80); --card-solid:#F7FAFD; --card-line:rgba(255,255,255,.75);
  --inset:rgba(240,245,251,.75); --inset-line:rgba(15,30,46,.08);
  --tx:#0F1E2E; --mu:#44566A; --faint:#7A8A9C;
  --ac:#2B52C4; --ac-soft:rgba(43,82,196,.10); --ac2:#C2571A;
  --grid:rgba(15,30,46,.07);
  --aq-good:#2E9E5B; --aq-mod:#E0A800; --aq-poor:#EE7F1E; --aq-sev:#D8452F;
  --c1:#D8452F; --c2:#1F9D8B; --c3:#6A55C9;
  --shadow:0 1px 1px rgba(15,30,46,.04),0 10px 30px rgba(43,82,196,.10);
  --shadow-hi:0 2px 4px rgba(15,30,46,.06),0 18px 44px rgba(43,82,196,.16);
  --serif:'Source Serif 4',Georgia,'Times New Roman',serif;
  --sans:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  --r:16px; --r-sm:10px; --nav-h:64px;
  --ease:cubic-bezier(.22,.8,.26,1);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:calc(var(--nav-h) + 24px)}
body{
  margin:0; color:var(--tx); font:400 14.5px/1.6 var(--sans); min-height:100vh;
  background:linear-gradient(160deg,#E6EEF8,#DDE9F2 55%,#E5EEF0) fixed;
  -webkit-font-smoothing:antialiased;
}
h1,h2,h3{font-family:var(--serif);font-weight:600;letter-spacing:-.015em;margin:0;text-wrap:balance;color:var(--tx)}
p{margin:0}
a{color:inherit}
button,select,input{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--ac);outline-offset:2px;border-radius:6px}

/* ===== Atmosphere: drifting colour fields ===== */
.sky{position:fixed;inset:0;z-index:-1;overflow:hidden;pointer-events:none}
.sky span{position:absolute;border-radius:50%;filter:blur(70px);opacity:.55;will-change:transform}
.sky .b1{width:46vw;height:40vw;left:-8vw;top:-10vw;background:#7FA4EC;animation:drift1 38s var(--ease) infinite alternate}
.sky .b2{width:40vw;height:36vw;right:-10vw;top:6vw;background:#7ED0CA;animation:drift2 44s var(--ease) infinite alternate}
.sky .b3{width:50vw;height:38vw;left:28vw;bottom:-18vw;background:#F2C193;opacity:.45;animation:drift3 50s var(--ease) infinite alternate}
@keyframes drift1{to{transform:translate(9vw,7vw) scale(1.12)}}
@keyframes drift2{to{transform:translate(-8vw,10vw) scale(.92)}}
@keyframes drift3{to{transform:translate(-10vw,-6vw) scale(1.08)}}

/* ===== Glass surfaces ===== */
.glass{
  background:var(--card); border:1px solid var(--card-line); border-radius:var(--r);
  box-shadow:var(--shadow);
  backdrop-filter:blur(14px) saturate(1.2); -webkit-backdrop-filter:blur(14px) saturate(1.2);
}
@media (prefers-reduced-transparency:reduce){.glass{background:var(--card-solid);backdrop-filter:none;-webkit-backdrop-filter:none}}
.pad{padding:22px}
.wrap{max-width:1240px;margin:0 auto;padding-left:24px;padding-right:24px}
@media (max-width:640px){.wrap{padding-left:14px;padding-right:14px}.pad{padding:16px}}

/* ===== Sticky navigation ===== */
.nav{position:fixed;top:0;left:0;right:0;z-index:1000;padding-top:12px;transition:transform .4s var(--ease)}
.nav .bar{display:flex;align-items:center;gap:18px;height:var(--nav-h);padding:0 10px 0 18px;border-radius:18px;position:relative;overflow:hidden}
.nav .mark{display:flex;align-items:center;gap:10px;text-decoration:none;flex:none}
.nav .mark .dot{width:28px;height:28px;border-radius:9px;background:conic-gradient(from 200deg,var(--aq-good),var(--aq-mod),var(--aq-poor),var(--aq-sev),var(--c3),var(--ac),var(--aq-good));box-shadow:inset 0 0 0 5px rgba(255,255,255,.75)}
.nav .mark b{font:600 14px var(--serif);line-height:1.1}
.nav .mark small{display:block;font:500 10.5px var(--sans);color:var(--mu);letter-spacing:.04em}
.links{position:relative;display:flex;gap:2px;margin-left:auto;overflow-x:auto;scrollbar-width:none}
.links::-webkit-scrollbar{display:none}
.links a{position:relative;z-index:1;display:flex;align-items:center;gap:7px;padding:8px 13px;border-radius:10px;text-decoration:none;font:500 13px var(--sans);color:var(--mu);white-space:nowrap;transition:color .25s}
.links a .n{font:500 10.5px var(--mono);color:var(--faint);transition:color .25s}
.links a:hover{color:var(--tx)}
.links a.active{color:var(--ac)}
.links a.active .n{color:var(--ac)}
.links .pill-bg{position:absolute;top:0;bottom:0;border-radius:10px;background:#fff;box-shadow:0 1px 3px rgba(15,30,46,.1);transition:left .45s var(--ease),width .45s var(--ease),opacity .3s;opacity:0}
.progress{position:absolute;left:0;bottom:0;height:2px;width:100%;background:linear-gradient(90deg,var(--ac),#1F9D8B,var(--ac2));transform-origin:left;transform:scaleX(0)}
@media (max-width:760px){.nav .mark small,.nav .mark b{display:none}}

/* ===== Hero ===== */
.hero{min-height:100vh;display:flex;flex-direction:column;justify-content:center;padding-top:calc(var(--nav-h) + 48px);padding-bottom:40px}
.hero-grid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:48px;align-items:center}
@media (max-width:1020px){.hero-grid{grid-template-columns:minmax(0,1fr);gap:32px}}
.kicker{display:inline-flex;align-items:center;gap:10px;font:600 11.5px var(--sans);letter-spacing:.1em;text-transform:uppercase;color:var(--ac)}
.kicker::before{content:"";width:28px;height:2px;background:var(--ac);border-radius:2px}
.hero h1{font-size:clamp(36px,5.4vw,64px);line-height:1.04;letter-spacing:-.025em;margin-top:18px}
.hero h1 em{font-style:italic;font-weight:500;color:var(--ac)}
.hero .subtitle{font:500 clamp(16px,1.6vw,19px)/1.45 var(--serif);color:var(--mu);margin-top:18px;max-width:46ch}
.hero .desc{font-size:15px;color:var(--mu);margin-top:16px;max-width:62ch}
.hero .desc strong{color:var(--tx);font-weight:600}
.credits{display:flex;flex-wrap:wrap;gap:8px 22px;margin-top:22px;font-size:13px;color:var(--mu)}
.credits b{display:block;font:600 10.5px var(--sans);letter-spacing:.09em;text-transform:uppercase;color:var(--faint);margin-bottom:2px}
.credits span{color:var(--tx);font-weight:500}
.cta{display:flex;flex-wrap:wrap;gap:10px;margin-top:28px}
.btn{display:inline-flex;align-items:center;gap:9px;padding:12px 20px;border-radius:12px;font:600 14px var(--sans);text-decoration:none;cursor:pointer;border:1px solid transparent;transition:transform .25s var(--ease),box-shadow .25s,background .25s}
.btn.primary{background:var(--ac);color:#fff;box-shadow:0 8px 22px rgba(43,82,196,.28)}
.btn.primary:hover{transform:translateY(-2px);box-shadow:0 12px 28px rgba(43,82,196,.34)}
.btn.ghost{background:rgba(255,255,255,.7);border-color:var(--inset-line);color:var(--tx)}
.btn.ghost:hover{background:#fff;transform:translateY(-2px)}
.btn svg{width:16px;height:16px}

.hero-kpis{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.kpi{position:relative;overflow:hidden;padding:20px;transition:transform .35s var(--ease),box-shadow .35s var(--ease)}
.kpi:hover{transform:translateY(-3px);box-shadow:var(--shadow-hi)}
.kpi::after{content:"";position:absolute;right:-34px;top:-34px;width:120px;height:120px;border-radius:50%;background:var(--k);opacity:.13;filter:blur(6px)}
.kpi .val{display:flex;align-items:baseline;flex-wrap:wrap;gap:4px 8px;margin-top:12px}
.kpi .val b{font:500 clamp(30px,3.2vw,40px)/1 var(--mono);font-variant-numeric:tabular-nums;color:var(--k)}
.kpi .val span{font-size:13px;color:var(--tx);font-weight:500}
.kpi .foot{font-size:12px;color:var(--mu);margin-top:8px}
.hero-sources{grid-column:1/-1;display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:12px 14px}
.hero-sources .eyebrow{margin-right:4px}

.questions{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:44px}
@media (max-width:900px){.questions{grid-template-columns:minmax(0,1fr)}}
.q{display:flex;flex-direction:column;gap:8px;padding:18px 20px;text-decoration:none;transition:transform .35s var(--ease),box-shadow .35s var(--ease)}
.q:hover{transform:translateY(-3px);box-shadow:var(--shadow-hi)}
.q .top{display:flex;justify-content:space-between;align-items:center}
.q .top span{font:600 11px var(--mono);color:var(--k);letter-spacing:.04em}
.q .top svg{width:16px;height:16px;color:var(--faint);transition:transform .3s var(--ease),color .3s}
.q:hover .top svg{transform:translateX(4px);color:var(--k)}
.q h3{font-size:17px}
.q p{font-size:13px;color:var(--mu)}
.scroll-cue{display:flex;flex-direction:column;align-items:center;gap:6px;margin-top:40px;font:500 11.5px var(--sans);letter-spacing:.08em;text-transform:uppercase;color:var(--faint);text-decoration:none}
.scroll-cue i{width:22px;height:34px;border:2px solid currentColor;border-radius:12px;position:relative}
.scroll-cue i::after{content:"";position:absolute;left:50%;top:7px;width:4px;height:7px;margin-left:-2px;border-radius:2px;background:currentColor;animation:wheel 1.8s ease-in-out infinite}
@keyframes wheel{0%{opacity:0;transform:translateY(-3px)}40%{opacity:1}100%{opacity:0;transform:translateY(9px)}}

/* ===== Sections ===== */
.section{padding:56px 0 24px;display:flex;flex-direction:column;gap:20px}
.sec-head{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:22px 26px;align-items:end;padding-bottom:6px}
.sec-num{font:500 clamp(54px,6vw,84px)/.85 var(--serif);color:transparent;-webkit-text-stroke:1.4px rgba(43,82,196,.45);letter-spacing:-.04em}
.sec-head h2{font-size:clamp(24px,2.8vw,34px);line-height:1.15;margin-top:8px}
.sec-head p{font-size:15px;color:var(--mu);margin-top:10px;max-width:80ch}
.sec-head p strong{color:var(--tx);font-weight:600}
.sec-head .stats{display:flex;gap:10px;flex-wrap:wrap}
@media (max-width:900px){.sec-head{grid-template-columns:auto minmax(0,1fr)}.sec-head .stats{grid-column:1/-1}}
@media (max-width:560px){.sec-head{grid-template-columns:minmax(0,1fr)}.sec-num{font-size:48px}}
.divider{height:1px;background:linear-gradient(90deg,transparent,rgba(15,30,46,.12),transparent);margin-top:36px}

.grid{display:grid;gap:18px}
.g3{grid-template-columns:repeat(3,minmax(0,1fr))}
.g2{grid-template-columns:repeat(2,minmax(0,1fr))}
.g-map{grid-template-columns:minmax(0,1.9fr) minmax(0,1fr)}
.g-sim{grid-template-columns:minmax(0,1.35fr) minmax(0,1fr)}
@media (max-width:1020px){.g-map,.g-sim,.g2,.g3{grid-template-columns:minmax(0,1fr)}}

/* scroll reveal (only hidden when JS is running) */
.js .reveal{opacity:0;transform:translateY(22px);transition:opacity .8s var(--ease),transform .8s var(--ease);transition-delay:var(--d,0s)}
.js .reveal.in{opacity:1;transform:none}

/* ===== Type helpers ===== */
.eyebrow{font:600 11px var(--sans);letter-spacing:.09em;text-transform:uppercase;color:var(--mu)}
.card-h{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.card-h h3{font-size:18px}
.sub{font-size:12.5px;color:var(--mu)}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.tag{display:inline-flex;align-items:center;gap:6px;font:500 11.5px var(--sans);padding:4px 10px;border-radius:99px;background:var(--inset);border:1px solid var(--inset-line);color:var(--mu)}
.src{display:inline-flex;align-items:center;gap:7px;font:500 12px var(--sans);padding:5px 11px;border-radius:99px;background:rgba(255,255,255,.7);border:1px solid var(--inset-line);color:var(--tx)}
.src i{width:7px;height:7px;border-radius:50%}
.stat{min-width:112px;padding:12px 14px;border-radius:var(--r-sm);background:rgba(255,255,255,.65);border:1px solid var(--inset-line)}
.stat b{display:block;font:500 24px/1.1 var(--mono);font-variant-numeric:tabular-nums;color:var(--ac)}
.stat span{font-size:11.5px;color:var(--mu)}

/* ===== Map ===== */
#map{height:480px;border-radius:var(--r-sm);z-index:10;border:1px solid var(--inset-line)}
.map-legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--mu);margin-top:12px;align-items:center}
.map-legend i{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px;vertical-align:-1px;box-shadow:0 0 0 2px #fff}
.leaflet-popup-content-wrapper{border-radius:12px;box-shadow:var(--shadow-hi);font-family:var(--sans)}
.leaflet-popup-content{margin:12px 14px;font-size:12.5px;line-height:1.5;color:var(--tx)}
.pop h4{font:600 14px var(--serif);margin:0 0 6px}
.pop .r{display:flex;justify-content:space-between;gap:18px}
.pop .r b{font-family:var(--mono);font-weight:500}
.pop .cl{color:var(--mu);margin-top:6px;font-size:11.5px}
.leaflet-container{font-family:var(--sans);background:#EEF3F8}
.marker-pulse{animation:pulse 2.6s ease-out infinite;transform-origin:center;transform-box:fill-box}
@keyframes pulse{0%{stroke-width:2;stroke-opacity:1}70%{stroke-width:14;stroke-opacity:0}100%{stroke-width:2;stroke-opacity:0}}

/* ===== City inspector ===== */
.inspector{display:flex;flex-direction:column;gap:14px}
.select{width:100%;appearance:none;-webkit-appearance:none;padding:10px 38px 10px 12px;border-radius:var(--r-sm);border:1px solid var(--inset-line);background:rgba(255,255,255,.85) url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'><path d='M1 1l5 5 5-5' fill='none' stroke='%2344566A' stroke-width='1.6'/></svg>") no-repeat right 14px center;font-size:14px;cursor:pointer}
.city-hero{display:flex;align-items:center;gap:14px;padding:14px;border-radius:var(--r-sm);background:var(--inset);border:1px solid var(--inset-line)}
.ring{position:relative;width:74px;height:74px;flex:none}
.ring svg{width:74px;height:74px;transform:rotate(-90deg)}
.ring circle{fill:none;stroke-width:7}
.ring .trk{stroke:rgba(15,30,46,.08)}
.ring .val{stroke-linecap:round;transition:stroke-dashoffset 1s var(--ease),stroke .5s}
.ring .lab{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;line-height:1}
.ring .lab b{font:600 17px var(--mono)}
.ring .lab small{font-size:9.5px;color:var(--mu);margin-top:3px}
.cluster-chip{display:inline-flex;align-items:center;gap:7px;font:600 12.5px var(--sans);margin-top:4px}
.cluster-chip i{width:8px;height:8px;border-radius:50%}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.metric{padding:11px 12px;border-radius:var(--r-sm);background:rgba(255,255,255,.6);border:1px solid var(--inset-line)}
.metric span{font-size:11.5px;color:var(--mu)}
.metric b{display:block;font:500 17px var(--mono);font-variant-numeric:tabular-nums;margin-top:2px}
.metric b small{font:400 11px var(--sans);color:var(--mu);margin-left:3px}
.bar{height:4px;border-radius:2px;background:rgba(15,30,46,.07);margin-top:7px;overflow:hidden}
.bar i{display:block;height:100%;border-radius:2px;background:var(--ac);transition:width .9s var(--ease)}
.note{font-size:12px;color:var(--mu);padding-top:12px;border-top:1px solid var(--inset-line)}
.note b{color:var(--tx);font-weight:600}
.swap{animation:swap .45s var(--ease)}
@keyframes swap{from{opacity:.2;transform:translateY(6px)}to{opacity:1;transform:none}}

/* ===== Segmented city picker ===== */
.seg{display:inline-flex;flex-wrap:wrap;gap:4px;padding:4px;border-radius:12px;background:var(--inset);border:1px solid var(--inset-line)}
.seg button{border:0;background:none;padding:6px 12px;border-radius:8px;font:500 12.5px var(--sans);color:var(--mu);cursor:pointer;transition:background .25s,color .25s,box-shadow .25s}
.seg button:hover{color:var(--tx)}
.seg button.on{background:#fff;color:var(--ac);box-shadow:0 1px 3px rgba(15,30,46,.1)}
.ts-layout{display:grid;grid-template-columns:200px minmax(0,1fr);gap:20px;align-items:stretch}
.ts-stats{display:flex;flex-direction:column;gap:10px}
.ts-stats .stat{flex:1}
.ts-stats .stat b{font-size:22px}
@media (max-width:820px){.ts-layout{grid-template-columns:minmax(0,1fr)}.ts-stats{flex-direction:row;flex-wrap:wrap}}
.chart-box{position:relative;height:340px}

/* ===== Cluster cards ===== */
.cluster{position:relative;overflow:hidden;padding:22px 22px 20px;transition:transform .35s var(--ease),box-shadow .35s var(--ease)}
.cluster:hover{transform:translateY(-3px);box-shadow:var(--shadow-hi)}
.cluster::before{content:"";position:absolute;left:0;right:0;top:0;height:4px;background:var(--k)}
.cluster .eyebrow{color:var(--k)}
.cluster h3{font-size:19px;margin-top:6px}
.cluster .cities{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}
.cluster .cities span{font:500 12px var(--sans);padding:3px 9px;border-radius:99px;background:color-mix(in srgb,var(--k) 11%,#fff);color:color-mix(in srgb,var(--k) 75%,#0F1E2E)}
.cluster p{font-size:13px;color:var(--mu);margin-top:12px}
.cluster p strong{color:var(--tx);font-weight:600}

/* ===== Figures ===== */
.figure{display:flex;flex-direction:column}
.figure .frame{position:relative;flex:1;display:flex;align-items:center;justify-content:center;border-radius:var(--r-sm);overflow:hidden;background:#fff;border:1px solid var(--inset-line);cursor:zoom-in}
.figure img{display:block;width:100%;height:auto;transition:transform .6s var(--ease)}
.figure .frame:hover img{transform:scale(1.02)}
.figure .zoom{position:absolute;right:10px;bottom:10px;font:500 11.5px var(--sans);padding:5px 10px;border-radius:99px;background:rgba(15,30,46,.72);color:#fff;opacity:0;transform:translateY(4px);transition:opacity .25s,transform .25s}
.figure .frame:hover .zoom,.figure .frame:focus-visible .zoom{opacity:1;transform:none}
.lightbox{position:fixed;inset:0;z-index:5000;display:none;align-items:center;justify-content:center;padding:4vh 4vw;background:rgba(15,30,46,.55);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);cursor:zoom-out}
.lightbox.open{display:flex;animation:fade .3s ease}
.lightbox figure{margin:0;max-width:min(1400px,92vw);background:#fff;border-radius:16px;padding:14px;box-shadow:0 30px 80px rgba(0,0,0,.35);animation:pop .4s var(--ease)}
.lightbox img{display:block;max-width:100%;max-height:80vh;margin:0 auto}
.lightbox figcaption{font:600 14px var(--serif);padding:10px 4px 2px;color:var(--tx)}
@keyframes fade{from{opacity:0}}
@keyframes pop{from{opacity:0;transform:scale(.96)}}

/* ===== Tables ===== */
.tbl-wrap{overflow-x:auto;border-radius:var(--r-sm);border:1px solid var(--inset-line);background:rgba(255,255,255,.55)}
table{width:100%;border-collapse:collapse;font-size:13px;min-width:720px}
thead th{text-align:left;font:600 11px var(--sans);letter-spacing:.07em;text-transform:uppercase;color:var(--mu);padding:12px 14px;background:rgba(240,245,251,.95);border-bottom:1px solid var(--inset-line);white-space:nowrap}
thead th .u{text-transform:none;letter-spacing:0}
tbody td{padding:12px 14px;border-bottom:1px solid var(--inset-line);vertical-align:middle}
tbody tr:last-child td{border-bottom:0}
tbody tr{transition:background .2s}
tbody tr:hover{background:rgba(43,82,196,.05)}
td.model{font-weight:600;color:var(--tx)}
td.type{color:var(--mu)}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
td.num .mb{display:block;height:3px;border-radius:2px;background:rgba(15,30,46,.07);margin-top:6px;width:84px;overflow:hidden}
td.num .mb i{display:block;height:100%;background:var(--c,var(--ac));border-radius:2px;width:0;transition:width 1.1s var(--ease)}
td.best{color:var(--ac);font-weight:600}
td.best .bst{font:600 9.5px var(--sans);letter-spacing:.06em;text-transform:uppercase;margin-left:7px;padding:1px 6px;border-radius:99px;background:var(--ac-soft);color:var(--ac);vertical-align:1px}
.pill{display:inline-block;font:600 11.5px var(--sans);padding:3px 10px;border-radius:99px}
.pill.yes{background:rgba(46,158,91,.13);color:#1E7B45}
.pill.no{background:rgba(15,30,46,.07);color:var(--mu)}

/* ===== Forecast simulator ===== */
.sim{position:relative;overflow:hidden}
.sim::before{content:"";position:absolute;inset:0;background:radial-gradient(500px 260px at 0% 0%,rgba(43,82,196,.10),transparent 70%),radial-gradient(420px 260px at 100% 100%,rgba(194,87,26,.08),transparent 70%);pointer-events:none}
.sim > *{position:relative}
.sliders{display:grid;grid-template-columns:1fr 1fr;gap:22px 26px}
@media (max-width:560px){.sliders{grid-template-columns:1fr}}
.sl label{display:flex;justify-content:space-between;align-items:baseline;font-size:12.5px;color:var(--mu);margin-bottom:9px}
.sl label output{font:600 15px var(--mono);color:var(--tx)}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:6px;border-radius:3px;background:linear-gradient(90deg,var(--ac) var(--p,50%),rgba(15,30,46,.1) var(--p,50%));outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;border-radius:50%;background:#fff;border:2px solid var(--ac);box-shadow:0 2px 6px rgba(43,82,196,.3);transition:transform .15s}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.15)}
input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:#fff;border:2px solid var(--ac);box-shadow:0 2px 6px rgba(43,82,196,.3)}
.sl .range-ends{display:flex;justify-content:space-between;font:400 10.5px var(--mono);color:var(--faint);margin-top:6px}
.gauge-card{display:flex;flex-direction:column;align-items:center;text-align:center;padding:22px;border-radius:var(--r);background:rgba(255,255,255,.75);border:1px solid var(--inset-line)}
.gauge{position:relative;width:250px;max-width:100%}
.gauge svg{display:block;width:100%;height:auto}
.gauge .trk{fill:none;stroke:rgba(15,30,46,.08);stroke-width:16;stroke-linecap:round}
.gauge .arc{fill:none;stroke-width:16;stroke-linecap:round;transition:stroke-dashoffset .7s var(--ease),stroke .4s}
.gauge .read{position:absolute;left:0;right:0;bottom:4px;text-align:center}
.gauge .read b{display:block;font:600 38px/1 var(--mono);font-variant-numeric:tabular-nums;transition:color .4s}
.risk{margin-top:12px;font:600 12px var(--sans);padding:5px 12px;border-radius:99px;transition:background .4s,color .4s}
.risk.high{background:rgba(216,69,47,.13);color:#A8321F}
.risk.mid{background:rgba(224,168,0,.17);color:#8A5A00}
.risk.low{background:rgba(46,158,91,.13);color:#1E7B45}
.persist{font-size:12.5px;color:var(--mu);margin-top:12px}
.persist b{color:var(--tx)}

/* ===== Footer & back-to-top ===== */
footer{margin:48px 0 28px}
footer .glass{padding:26px;display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:20px;align-items:center}
footer h3{font-size:18px}
footer p{font-size:12.5px;color:var(--mu);margin-top:6px}
footer .srcs{display:flex;flex-wrap:wrap;gap:8px;justify-content:flex-end}
@media (max-width:820px){footer .glass{grid-template-columns:minmax(0,1fr)}footer .srcs{justify-content:flex-start}}
.to-top{position:fixed;right:22px;bottom:22px;z-index:900;width:46px;height:46px;border-radius:14px;border:1px solid var(--card-line);background:var(--card);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);box-shadow:var(--shadow-hi);display:flex;align-items:center;justify-content:center;cursor:pointer;color:var(--ac);opacity:0;transform:translateY(12px) scale(.9);pointer-events:none;transition:opacity .35s var(--ease),transform .35s var(--ease)}
.to-top.show{opacity:1;transform:none;pointer-events:auto}
.to-top:hover{transform:translateY(-3px)}
.to-top svg{width:18px;height:18px}

@media (prefers-reduced-motion:reduce){
  html{scroll-behavior:auto}
  *,*::before,*::after{animation:none!important;transition:none!important}
  .js .reveal{opacity:1;transform:none}
}
</style>
</head>
<body>
<div class="sky" aria-hidden="true"><span class="b1"></span><span class="b2"></span><span class="b3"></span></div>

<!-- ================= Sticky navigation ================= -->
<header class="nav">
  <div class="wrap">
    <nav class="bar glass" aria-label="Sections">
      <a href="#top" class="mark"><span class="dot"></span><span><b>Air Quality Analytics</b><small>IT 462 · DA-IICT</small></span></a>
      <div class="links" id="navLinks">
        <span class="pill-bg" id="navPill"></span>
        <a href="#top" data-sec="top"><span class="n">00</span>Overview</a>
        <a href="#tab-overview" data-sec="tab-overview"><span class="n">01</span>City Map &amp; Stations</a>
        <a href="#tab-problem1" data-sec="tab-problem1"><span class="n">02</span>Pollution Profiles</a>
        <a href="#tab-problem2" data-sec="tab-problem2"><span class="n">03</span>Severe Forecasting</a>
        <a href="#tab-problem3" data-sec="tab-problem3"><span class="n">04</span>Econometric Drivers</a>
      </div>
      <span class="progress" id="progress"></span>
    </nav>
  </div>
</header>

<!-- ================= Hero: project title & description ================= -->
<section class="hero wrap" id="top">
  <div class="hero-grid">
    <div class="reveal">
      <span class="kicker">IT 462 · Exploratory Data Analysis</span>
      <h1>Air Quality Analytics &amp; <em style="white-space:nowrap">Severe-Episode</em> Forecasting</h1>
      <p class="subtitle">Exploratory Data Analysis of Air Pollution in Indian Cities: Pollutant Signatures, Severe-Episode Forecasting, and the Role of Weather, Festivals and Crop Burning</p>
      <p class="desc">An end-to-end, reproducible study of ambient air quality across <strong>12 major Indian cities</strong>, spanning <strong>155 CAAQMS monitoring stations</strong> and over <strong>5.23 million 15-minute observations</strong>. CPCB pollution records are joined with Open-Meteo weather, NASA FIRMS crop-fire detections and a festival calendar to profile each city, forecast next-day severe episodes, and measure what drives PM2.5.</p>
      <div class="credits">
        <div><b>Course</b><span>IT 462 EDA Project</span></div>
        <div><b>Academic Guide</b><span>Dr. Gopinath Panda</span></div>
        <div><b>Institute</b><span>DA-IICT</span></div>
      </div>
      <div class="cta">
        <a href="#tab-overview" class="btn primary">Explore the dashboard
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M8 3v10M3.5 8.5 8 13l4.5-4.5"/></svg></a>
        <a href="#tab-problem2" class="btn ghost">Try the forecaster</a>
      </div>
    </div>

    <div class="hero-kpis">
      <div class="glass kpi reveal" style="--k:var(--ac);--d:.08s">
        <span class="eyebrow">Monitored Metros</span>
        <div class="val"><b data-count="12">12</b><span>Major Cities</span></div>
        <p class="foot">North, South, East, West</p>
      </div>
      <div class="glass kpi reveal" style="--k:#1F9D8B;--d:.16s">
        <span class="eyebrow">CAAQMS Stations</span>
        <div class="val"><b data-count="155">155</b><span>Active Monitors</span></div>
        <p class="foot">5.23M 15-min Readings</p>
      </div>
      <div class="glass kpi reveal" style="--k:var(--aq-sev);--d:.24s">
        <span class="eyebrow">Severe Episode Days</span>
        <div class="val"><b data-count="137">137</b><span>City-Days</span></div>
        <p class="foot">AQI &ge; 300 / PM2.5 &ge; 150</p>
      </div>
      <div class="glass kpi reveal" style="--k:var(--ac2);--d:.32s">
        <span class="eyebrow">Peak Crop Fires</span>
        <div class="val"><b data-count="5180">5,180</b><span>/ Day</span></div>
        <p class="foot">Punjab &amp; Haryana Airshed</p>
      </div>
      <div class="glass hero-sources reveal" style="--d:.4s">
        <span class="eyebrow">Data</span>
        <span class="src"><i style="background:var(--aq-good)"></i>CPCB NAQI</span>
        <span class="src"><i style="background:var(--ac)"></i>Open-Meteo Weather</span>
        <span class="src"><i style="background:var(--c3)"></i>NASA FIRMS Fires</span>
      </div>
    </div>
  </div>

  <div class="questions">
    <a href="#tab-problem1" class="glass q reveal" style="--k:var(--c1)">
      <div class="top"><span>PROBLEM 1</span><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 8h10M8.5 3.5 13 8l-4.5 4.5"/></svg></div>
      <h3>Which cities share a pollution profile?</h3>
      <p>PCA and clustering group the 12 cities into distinct atmospheric profiles.</p>
    </a>
    <a href="#tab-problem2" class="glass q reveal" style="--k:var(--ac);--d:.08s">
      <div class="top"><span>PROBLEM 2</span><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 8h10M8.5 3.5 13 8l-4.5 4.5"/></svg></div>
      <h3>Can we forecast tomorrow's severe day?</h3>
      <p>Next-day severe-episode models, tested against the persistence baseline.</p>
    </a>
    <a href="#tab-problem3" class="glass q reveal" style="--k:var(--ac2);--d:.16s">
      <div class="top"><span>PROBLEM 3</span><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 8h10M8.5 3.5 13 8l-4.5 4.5"/></svg></div>
      <h3>What drives PM2.5 up or down?</h3>
      <p>Panel regression of weather, Diwali and crop-burning effects.</p>
    </a>
  </div>

  <a href="#tab-overview" class="scroll-cue" aria-label="Scroll to the dashboard"><i></i>Scroll</a>
</section>

<main class="wrap">

  <!-- ========================================================= -->
  <!-- 01: OVERVIEW & MAP                                         -->
  <!-- ========================================================= -->
  <section id="tab-overview" class="section" data-spy>
    <div class="sec-head reveal">
      <span class="sec-num">01</span>
      <div>
        <span class="eyebrow">Overview</span>
        <h2>City Map &amp; Stations</h2>
        <p>Where the 12 cities sit, how polluted each one is on average, and how daily PM2.5 moved through 2025.</p>
      </div>
    </div>

    <div class="grid g-map">
      <div class="glass pad reveal">
        <div class="card-h">
          <div><h3>Spatial Network &amp; Ambient PM2.5 Severity</h3><p class="sub">Marker size and colour scale with city mean PM2.5 · click a city to inspect it</p></div>
        </div>
        <div id="map"></div>
        <div class="map-legend">
          <span><i style="background:var(--aq-good)"></i>Mean PM2.5 &le; 50</span>
          <span><i style="background:var(--aq-poor)"></i>50 – 80</span>
          <span><i style="background:var(--aq-sev)"></i>&gt; 80 µg/m³</span>
        </div>
      </div>
      <div class="glass pad inspector reveal" style="--d:.1s">
        <div>
          <h3 style="font-size:18px">City Profile Inspector</h3>
          <p class="sub" style="margin-top:2px">Select a city, or click one on the map</p>
        </div>
        <label for="citySelect" class="eyebrow" style="margin-bottom:-6px">Select City</label>
        <select id="citySelect" onchange="updateCityCard()" class="select"></select>
        <div id="cityDetails"></div>
        <div class="note"><b>CPCB Validation:</b> &ge;16 valid hours and &ge;3 criteria pollutants.</div>
      </div>
    </div>

    <div class="glass pad reveal">
      <div class="card-h">
        <div><h3>Longitudinal Daily Air Quality Trajectory (2025)</h3><p class="sub">Daily mean PM2.5 against the 60 µg/m³ national ambient standard</p></div>
        <div class="seg" id="tsSeg" role="group" aria-label="City"></div>
      </div>
      <div class="ts-layout">
        <div class="ts-stats" id="tsStats"></div>
        <div class="chart-box"><canvas id="tsChart"></canvas></div>
      </div>
    </div>
    <div class="divider"></div>
  </section>

  <!-- ========================================================= -->
  <!-- 02: PROBLEM 1 (CLUSTERING & PROFILES)                      -->
  <!-- ========================================================= -->
  <section id="tab-problem1" class="section" data-spy>
    <div class="sec-head reveal">
      <span class="sec-num">02</span>
      <div>
        <span class="eyebrow">Problem 1 · Unsupervised learning</span>
        <h2>Problem 1: Unsupervised City Pollution Profiles</h2>
        <p>Using multi-pollutant stoichiometric ratios, winter-to-monsoon seasonality amplitude, and severe frequency, Indian cities segment into 3 distinct atmospheric clusters explaining <strong>79.15%</strong> of variance in the first two principal components.</p>
      </div>
      <div class="stats">
        <div class="stat"><b>3</b><span>clusters</span></div>
        <div class="stat"><b>79.15%</b><span>variance, PC1 + PC2</span></div>
      </div>
    </div>

    <div class="grid g3">
      <div class="glass cluster reveal" style="--k:var(--c1)">
        <span class="eyebrow">Cluster 1</span>
        <h3>Gangetic High Smog Core</h3>
        <div class="cities"><span>Delhi</span><span>Patna</span><span>Lucknow</span></div>
        <p><strong>Signature:</strong> Extreme winter thermal inversion, elevated PM2.5/PM10 ratio, high seasonal amplitude.</p>
      </div>
      <div class="glass cluster reveal" style="--k:var(--c2);--d:.08s">
        <span class="eyebrow">Cluster 2</span>
        <h3>Urban-Vehicular Coastal</h3>
        <div class="cities"><span>Mumbai</span><span>Kolkata</span><span>Chennai</span><span>Surat</span></div>
        <p><strong>Signature:</strong> Marine boundary layer dispersion, maritime breeze, moderate annual PM2.5 with NO2 spikes.</p>
      </div>
      <div class="glass cluster reveal" style="--k:var(--c3);--d:.16s">
        <span class="eyebrow">Cluster 3</span>
        <h3>Deccan &amp; Semi-Arid Plateau</h3>
        <div class="cities"><span>Bengaluru</span><span>Hyderabad</span><span>Pune</span><span>Ahmedabad</span><span>Jaipur</span></div>
        <p><strong>Signature:</strong> Higher ventilation index, crustal coarse dust (PM10) dominance, low severe episodes.</p>
      </div>
    </div>

    <div class="grid g2">
      <div class="glass pad figure reveal">
        <div class="card-h"><h3>Principal Component Analysis (PCA) Biplot Projection</h3></div>
        <div class="frame" tabindex="0" role="button" aria-label="Enlarge PCA biplot"><img src="%%IMG_FIG11%%" alt="PCA Biplot"><span class="zoom">Click to enlarge</span></div>
      </div>
      <div class="glass pad figure reveal" style="--d:.1s">
        <div class="card-h"><h3>Hierarchical Dendrogram (Ward Linkage)</h3></div>
        <div class="frame" tabindex="0" role="button" aria-label="Enlarge dendrogram"><img src="%%IMG_FIG12%%" alt="Dendrogram"><span class="zoom">Click to enlarge</span></div>
      </div>
    </div>
    <div class="divider"></div>
  </section>

  <!-- ========================================================= -->
  <!-- 03: PROBLEM 2 (SEVERE FORECASTING SIMULATOR)               -->
  <!-- ========================================================= -->
  <section id="tab-problem2" class="section" data-spy>
    <div class="sec-head reveal">
      <span class="sec-num">03</span>
      <div>
        <span class="eyebrow">Problem 2 · Classification</span>
        <h2>Problem 2: Severe Episode Prediction Forecaster (Next-Day Day t+1)</h2>
        <p>Evaluated out-of-sample against the <strong>Persistence Baseline</strong>. Predicts whether tomorrow will exceed <strong>AQI &ge; 300 / PM2.5 &ge; 150 µg/m³</strong> strictly from information up to day t.</p>
      </div>
      <div class="stats" id="p2Stats"></div>
    </div>

    <div class="glass pad sim reveal">
      <div class="card-h">
        <div><h3>Interactive Next-Day Forecasting Simulator</h3><p class="sub">Adjust current day's (t) atmospheric inputs to compute the next-day (t+1) severe episode probability.</p></div>
      </div>
      <div class="grid g-sim" style="align-items:center">
        <div class="sliders">
          <div class="sl">
            <label for="sim-pm25">Today's PM2.5 (µg/m³) <output id="val-pm25">120</output></label>
            <input type="range" id="sim-pm25" min="10" max="350" value="120" oninput="runSimulation()">
            <div class="range-ends"><span>10</span><span>350</span></div>
          </div>
          <div class="sl">
            <label for="sim-ws">Wind Speed (km/h) <output id="val-ws">6.0</output></label>
            <input type="range" id="sim-ws" min="1" max="25" value="6" step="0.5" oninput="runSimulation()">
            <div class="range-ends"><span>1</span><span>25</span></div>
          </div>
          <div class="sl">
            <label for="sim-blh">Boundary Layer Height (m) <output id="val-blh">350</output></label>
            <input type="range" id="sim-blh" min="50" max="1500" value="350" step="25" oninput="runSimulation()">
            <div class="range-ends"><span>50</span><span>1500</span></div>
          </div>
          <div class="sl">
            <label for="sim-fire">Upwind Fire Count <output id="val-fire">800</output></label>
            <input type="range" id="sim-fire" min="0" max="4500" value="800" step="50" oninput="runSimulation()">
            <div class="range-ends"><span>0</span><span>4500</span></div>
          </div>
        </div>
        <div class="gauge-card">
          <span class="eyebrow">Predicted Severe Probability (t+1)</span>
          <div class="gauge" style="margin-top:10px">
            <svg viewBox="0 0 240 136" aria-hidden="true">
              <path class="trk" d="M20 124 A100 100 0 0 1 220 124"/>
              <path class="arc" id="gaugeArc" d="M20 124 A100 100 0 0 1 220 124" pathLength="100" stroke-dasharray="100" stroke-dashoffset="100"/>
            </svg>
            <div class="read"><b id="sim-prob-badge">68.4%</b></div>
          </div>
          <span id="sim-risk-label" class="risk high">High Risk Episode</span>
          <p class="persist">Persistence Baseline: <b id="sim-persist-val">Non-Severe</b></p>
        </div>
      </div>
    </div>

    <div class="glass pad reveal">
      <div class="card-h"><h3>Out-of-Sample Test Set Classification Benchmark</h3><span class="tag">Severe class metrics</span></div>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>Model</th><th>Type</th><th>Precision</th><th>Recall</th><th>Severe F1</th><th>ROC-AUC</th><th>PR-AUC</th></tr></thead>
          <tbody id="clsTableBody"></tbody>
        </table>
      </div>
    </div>

    <div class="grid g2">
      <div class="glass pad figure reveal">
        <div class="card-h"><h3>Precision-Recall &amp; ROC Curves</h3></div>
        <div class="frame" tabindex="0" role="button" aria-label="Enlarge PR and ROC curves"><img src="%%IMG_FIG14%%" alt="PR ROC Curves"><span class="zoom">Click to enlarge</span></div>
      </div>
      <div class="glass pad figure reveal" style="--d:.1s">
        <div class="card-h"><h3>Confusion Matrix Comparison</h3></div>
        <div class="frame" tabindex="0" role="button" aria-label="Enlarge confusion matrices"><img src="%%IMG_FIG15%%" alt="Confusion Matrix"><span class="zoom">Click to enlarge</span></div>
      </div>
    </div>
    <div class="divider"></div>
  </section>

  <!-- ========================================================= -->
  <!-- 04: PROBLEM 3 (ECONOMETRIC DRIVERS & INFERENCE)            -->
  <!-- ========================================================= -->
  <section id="tab-problem3" class="section" data-spy>
    <div class="sec-head reveal">
      <span class="sec-num">04</span>
      <div>
        <span class="eyebrow">Problem 3 · Regression &amp; inference</span>
        <h2>Problem 3: Drivers of PM2.5 (Regression &amp; Econometric Inference)</h2>
        <p>Quantifies the marginal effects of temperature, horizontal wind, boundary-layer stagnation, Diwali festive shock, and transboundary biomass plumes with <strong>City Fixed Effects</strong> and <strong>Cluster-Robust Standard Errors</strong> (R² = <strong>0.858</strong>, Durbin-Watson = <strong>1.914</strong>).</p>
      </div>
      <div class="stats">
        <div class="stat"><b>0.858</b><span>R², fixed effects</span></div>
        <div class="stat"><b>1.914</b><span>Durbin-Watson</span></div>
      </div>
    </div>

    <div class="grid g2">
      <div class="glass pad figure reveal">
        <div class="card-h"><h3>Econometric Marginal Effects (95% Clustered CI)</h3></div>
        <div class="frame" tabindex="0" role="button" aria-label="Enlarge forest plot"><img src="%%IMG_FIG17%%" alt="Forest Plot"><span class="zoom">Click to enlarge</span></div>
      </div>
      <div class="glass pad figure reveal" style="--d:.1s">
        <div class="card-h"><h3>Regression Residual Diagnostics</h3></div>
        <div class="frame" tabindex="0" role="button" aria-label="Enlarge residual diagnostics"><img src="%%IMG_FIG18%%" alt="Residual Diagnostics"><span class="zoom">Click to enlarge</span></div>
      </div>
    </div>

    <div class="glass pad reveal">
      <div class="card-h"><h3>Predictive Machine Learning Benchmark (Test Set)</h3><span class="tag">Lower RMSE / MAE is better</span></div>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>Model</th><th>Test RMSE <span class="u">(µg/m³)</span></th><th>Test MAE <span class="u">(µg/m³)</span></th><th>R² Score</th><th>Beats Baseline</th></tr></thead>
          <tbody id="regTableBody"></tbody>
        </table>
      </div>
    </div>
  </section>
</main>

<footer class="wrap">
  <div class="glass reveal">
    <div>
      <h3>Air Pollution in 12 Indian Cities, 2025</h3>
      <p>An exploratory data analysis and machine learning study for IT 462 Exploratory Data Analysis, guided by Dr. Gopinath Panda at Dhirubhai Ambani Institute of Information and Communication Technology (DA-IICT), Gandhinagar.</p>
    </div>
    <div class="srcs">
      <span class="src"><i style="background:var(--aq-good)"></i>CPCB NAQI</span>
      <span class="src"><i style="background:var(--ac)"></i>Open-Meteo Weather</span>
      <span class="src"><i style="background:var(--c3)"></i>NASA FIRMS Fires</span>
    </div>
  </div>
</footer>

<button class="to-top" id="toTop" aria-label="Back to top">
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 13V3M3.5 7.5 8 3l4.5 4.5"/></svg>
</button>

<div class="lightbox" id="lightbox" role="dialog" aria-modal="true" aria-label="Enlarged figure">
  <figure><img id="lbImg" alt=""><figcaption id="lbCap"></figcaption></figure>
</div>

<script>
const cities = %%CITIES_JSON%%;
const tsData = %%TS_JSON%%;
const clsData = %%CLS_JSON%%;
const regData = %%REG_JSON%%;
const SEVERE_THR = %%SEVERE_THR%%;

const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const css = getComputedStyle(document.documentElement);
const tok = n => css.getPropertyValue(n).trim();

/* PM2.5 colour bands (same thresholds as the original map) */
function pmColor(v){ return v > 80 ? tok('--aq-sev') : (v > 50 ? tok('--aq-poor') : tok('--aq-good')); }
function clusterColor(name){
  const n = (name || '').toLowerCase();
  if (n.includes('gangetic') || n.includes('extreme') || n.includes('core') || n.includes('delhi')) return tok('--c1');
  if (n.includes('coastal') || n.includes('clean') || n.includes('southern') || n.includes('maritime')) return tok('--c2');
  return tok('--c3');
}

/* ---------- Count-up for KPI numbers ---------- */
function countUp(el){
  const target = +el.dataset.count, fmt = n => n.toLocaleString('en-US');
  if (REDUCED) { el.textContent = fmt(target); return; }
  const t0 = performance.now(), dur = 1400;
  const step = t => {
    const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
    el.textContent = fmt(Math.round(target * e));
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ---------- Navigation: scroll-spy, progress, back-to-top ---------- */
function initNav(){
  const links = [...document.querySelectorAll('#navLinks a')];
  const pill = document.getElementById('navPill');
  const progress = document.getElementById('progress');
  const toTop = document.getElementById('toTop');
  const ids = links.map(a => a.dataset.sec);

  const setActive = id => {
    links.forEach(a => a.classList.toggle('active', a.dataset.sec === id));
    const a = links.find(x => x.dataset.sec === id);
    if (a) { pill.style.left = a.offsetLeft + 'px'; pill.style.width = a.offsetWidth + 'px'; pill.style.opacity = 1;
      a.scrollIntoView({block:'nearest', inline:'nearest'}); }
  };
  const onScroll = () => {
    const y = window.scrollY, h = document.documentElement.scrollHeight - innerHeight;
    progress.style.transform = `scaleX(${h > 0 ? y / h : 0})`;
    toTop.classList.toggle('show', y > innerHeight * .6);
    let current = 'top';
    const line = innerHeight * .35;
    ids.forEach(id => { const el = document.getElementById(id); if (el && el.getBoundingClientRect().top - line <= 0) current = id; });
    if (current !== onScroll.last) { onScroll.last = current; setActive(current); }
  };
  window.addEventListener('scroll', onScroll, {passive:true});
  window.addEventListener('resize', () => { onScroll.last = null; onScroll(); });
  toTop.addEventListener('click', () => window.scrollTo({top:0, behavior: REDUCED ? 'auto' : 'smooth'}));
  onScroll();
  if (document.fonts) document.fonts.ready.then(() => { onScroll.last = null; onScroll(); });
}

/* ---------- Scroll reveal ---------- */
function initReveal(){
  const els = document.querySelectorAll('.reveal');
  if (REDUCED || !('IntersectionObserver' in window)) { els.forEach(e => e.classList.add('in')); return; }
  const io = new IntersectionObserver(entries => {
    entries.forEach(en => {
      if (!en.isIntersecting) return;
      en.target.classList.add('in');
      en.target.querySelectorAll('[data-count]').forEach(countUp);
      if (en.target.querySelector('.mb i')) animateBars(en.target);
      io.unobserve(en.target);
    });
  }, {threshold:.12, rootMargin:'0px 0px -6% 0px'});
  els.forEach(e => io.observe(e));
}

/* ---------- Map ---------- */
function initMap(){
  const map = L.map('map', {scrollWheelZoom:false}).setView([21.5, 78.5], 4.8);
  window.map = map;
  const esriLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 16, attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ'
  });
  const esriStreet = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 16, attribution: 'Tiles &copy; Esri'
  });
  esriLayer.addTo(map);
  L.control.layers({
    "Clean Light Canvas (Esri)": esriLayer,
    "Detailed Street Map (Esri)": esriStreet
  }, null, { position: 'topright' }).addTo(map);

  cities.forEach(c => {
    const color = pmColor(c.pm25);
    const r = Math.max(7, Math.min(20, c.pm25 / 5.5));
    if (c.pm25 > 80) {
      L.circleMarker([c.lat, c.lon], {radius:r, fill:false, color:color, weight:2, className:'marker-pulse', interactive:false}).addTo(map);
    }
    const circle = L.circleMarker([c.lat, c.lon], {
      radius: r, fillColor: color, color: '#ffffff', weight: 2, opacity: 1, fillOpacity: 0.88
    }).addTo(map);
    circle.bindTooltip(c.city, {direction:'top', offset:[0,-r]});
    circle.bindPopup(`
      <div class="pop">
        <h4>${c.city} (${c.state})</h4>
        <div class="r"><span>Mean PM2.5</span><b>${c.pm25} µg/m³</b></div>
        <div class="r"><span>Mean PM10</span><b>${c.pm10} µg/m³</b></div>
        <div class="r"><span>Severe Days</span><b style="color:${tok('--aq-sev')}">${c.severe_share}%</b></div>
        <p class="cl">Cluster: ${c.cluster}</p>
      </div>`);
    circle.on('click', () => {
      const sel = document.getElementById('citySelect');
      sel.value = c.city; updateCityCard();
    });
  });
  // the map lives in a card that fades in; make sure tiles fill it once visible
  setTimeout(() => map.invalidateSize(), 400);
}

/* ---------- City inspector ---------- */
function initCityInspector(){
  const sel = document.getElementById('citySelect');
  cities.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c.city;
    opt.textContent = `${c.city} (${c.state})`;
    sel.appendChild(opt);
  });
  if (cities.find(c => c.city === 'Delhi')) sel.value = 'Delhi';
  updateCityCard();
}
function updateCityCard(){
  const cName = document.getElementById('citySelect').value;
  const c = cities.find(x => x.city === cName) || cities[0];
  const maxOf = k => Math.max(...cities.map(x => x[k]));
  const rank = [...cities].sort((a,b) => b.pm25 - a.pm25).findIndex(x => x.city === c.city) + 1;
  const col = pmColor(c.pm25), cc = clusterColor(c.cluster);
  const C = 2 * Math.PI * 30, frac = c.pm25 / maxOf('pm25');
  const m = (label, v, unit, key, color) => `
    <div class="metric"><span>${label}</span><b>${v}<small>${unit}</small></b>
      <div class="bar"><i style="width:0;background:${color || 'var(--ac)'}" data-w="${(v / maxOf(key) * 100).toFixed(1)}"></i></div></div>`;
  const div = document.getElementById('cityDetails');
  div.innerHTML = `
    <div class="swap" style="display:flex;flex-direction:column;gap:12px">
      <div class="city-hero">
        <div class="ring">
          <svg viewBox="0 0 74 74"><circle class="trk" cx="37" cy="37" r="30"/>
            <circle class="val" cx="37" cy="37" r="30" stroke="${col}" stroke-dasharray="${C}" stroke-dashoffset="${C}" data-off="${C * (1 - frac)}"/></svg>
          <div class="lab"><b>${c.pm25}</b><small>PM2.5</small></div>
        </div>
        <div>
          <span class="eyebrow">Classification Cluster</span>
          <div class="cluster-chip"><i style="background:${cc}"></i>${c.cluster}</div>
          <p class="sub" style="margin-top:4px">#${rank} of ${cities.length} by mean PM2.5</p>
        </div>
      </div>
      <div class="metrics">
        ${m('Mean PM2.5', c.pm25, 'µg/m³', 'pm25', col)}
        ${m('Mean PM10', c.pm10, 'µg/m³', 'pm10')}
        ${m('Mean NO2', c.no2, 'µg/m³', 'no2')}
        ${m('Severe Day Share', c.severe_share, '%', 'severe_share', tok('--aq-sev'))}
      </div>
      <p class="sub">Monitors: <b class="mono" style="color:var(--tx)">${c.stations}</b> regulatory CAAQMS stations</p>
    </div>`;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    div.querySelectorAll('.bar i').forEach(i => i.style.width = i.dataset.w + '%');
    const ring = div.querySelector('.ring .val'); if (ring) ring.style.strokeDashoffset = ring.dataset.off;
  }));
}

/* ---------- Time series ---------- */
let tsChartInstance = null;
let tsCity = 'Delhi';
function initTsPicker(){
  const seg = document.getElementById('tsSeg');
  ['Delhi','Mumbai','Bengaluru','Kolkata','Ahmedabad','Pune'].filter(c => tsData[c]).forEach(c => {
    const b = document.createElement('button');
    b.textContent = c; b.dataset.city = c;
    b.onclick = () => { tsCity = c; updateTsChart(); };
    seg.appendChild(b);
  });
}
function updateTsChart(){
  const city = tsCity;
  const data = tsData[city];
  if (!data) return;
  document.querySelectorAll('#tsSeg button').forEach(b => {
    b.classList.toggle('on', b.dataset.city === city);
    b.setAttribute('aria-pressed', b.dataset.city === city);
  });

  const vals = data.pm25.filter(v => v !== null && !isNaN(v));
  const mean = vals.reduce((a,b) => a + b, 0) / (vals.length || 1);
  const peak = Math.max(...vals);
  const above = vals.filter(v => v > 60).length;
  document.getElementById('tsStats').innerHTML = `
    <div class="stat swap"><b>${mean.toFixed(1)}</b><span>mean µg/m³</span></div>
    <div class="stat swap"><b style="color:${pmColor(peak)}">${peak.toFixed(1)}</b><span>peak µg/m³</span></div>
    <div class="stat swap"><b>${above}</b><span>days above 60 µg/m³</span></div>
    <div class="stat swap"><b>${vals.length}</b><span>days observed</span></div>`;

  const ctx = document.getElementById('tsChart').getContext('2d');
  if (tsChartInstance) tsChartInstance.destroy();
  const ac2 = tok('--ac2');
  tsChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.dates,
      datasets: [
        {
          label: `${city} PM2.5 (µg/m³)`,
          data: data.pm25,
          borderColor: ac2,
          backgroundColor: (context) => {
            const {chart} = context; const {ctx, chartArea} = chart;
            if (!chartArea) return 'rgba(194,87,26,.12)';
            const g = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
            g.addColorStop(0, 'rgba(194,87,26,.28)'); g.addColorStop(1, 'rgba(194,87,26,0)');
            return g;
          },
          fill: 'origin',
          borderWidth: 1.8,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHoverBackgroundColor: ac2,
          tension: 0.25
        },
        {
          label: 'National Ambient Standard (60 µg/m³)',
          data: data.dates.map(() => 60),
          borderColor: tok('--aq-sev'),
          borderDash: [6, 5],
          borderWidth: 1.3,
          pointRadius: 0,
          pointHoverRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: REDUCED ? false : { duration: 900, easing: 'easeOutQuart' },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top', align: 'end', labels: { usePointStyle: true, pointStyle: 'line', boxWidth: 26, color: tok('--mu'), font: { family: 'Inter', size: 12 } } },
        tooltip: {
          backgroundColor: 'rgba(15,30,46,.92)', padding: 10, cornerRadius: 8,
          titleFont: { family: 'Inter', weight: '600' }, bodyFont: { family: 'IBM Plex Mono' }
        }
      },
      scales: {
        x: { grid: { display: false }, border: { display: false }, ticks: { maxTicksLimit: 12, color: tok('--mu'), font: { family: 'Inter', size: 11 } } },
        y: {
          title: { display: true, text: 'PM2.5 (µg/m³)', color: tok('--mu'), font: { family: 'Inter', size: 11.5 } },
          grid: { color: tok('--grid') }, border: { display: false },
          ticks: { color: tok('--mu'), font: { family: 'IBM Plex Mono', size: 11 } }
        }
      }
    }
  });
}

/* ---------- Forecast simulator ---------- */
function paintRange(el){
  const p = (el.value - el.min) / (el.max - el.min) * 100;
  el.style.setProperty('--p', p + '%');
}
function runSimulation(){
  const pm25 = parseFloat(document.getElementById('sim-pm25').value);
  const ws = parseFloat(document.getElementById('sim-ws').value);
  const blh = parseFloat(document.getElementById('sim-blh').value);
  const fire = parseFloat(document.getElementById('sim-fire').value);
  ['sim-pm25','sim-ws','sim-blh','sim-fire'].forEach(id => paintRange(document.getElementById(id)));

  document.getElementById('val-pm25').textContent = pm25;
  document.getElementById('val-ws').textContent = ws.toFixed(1);
  document.getElementById('val-blh').textContent = blh;
  document.getElementById('val-fire').textContent = fire;

  // Fitted logistic regression approximation:
  const z = -3.2 + (0.025 * pm25) - (0.14 * ws) - (0.0022 * blh) + (0.00055 * fire);
  const prob = 1 / (1 + Math.exp(-z));
  const pct = (prob * 100).toFixed(1);

  const badge = document.getElementById('sim-prob-badge');
  badge.textContent = pct + '%';
  const riskLbl = document.getElementById('sim-risk-label');
  let color;
  if (prob >= 0.5) {
    riskLbl.textContent = 'High Severe Risk'; riskLbl.className = 'risk high'; color = tok('--aq-sev');
  } else if (prob >= 0.25) {
    riskLbl.textContent = 'Moderate Elevated Risk'; riskLbl.className = 'risk mid'; color = tok('--aq-mod');
  } else {
    riskLbl.textContent = 'Low Risk Baseline'; riskLbl.className = 'risk low'; color = tok('--aq-good');
  }
  badge.style.color = color;
  const arc = document.getElementById('gaugeArc');
  arc.style.stroke = color;
  arc.style.strokeDashoffset = (100 - prob * 100).toFixed(2);
  document.getElementById('sim-persist-val').textContent = pm25 >= SEVERE_THR ? 'Severe' : 'Non-Severe';
}

/* ---------- Tables ---------- */
function bestIdx(rows, key, lower){
  let bi = -1, bv = lower ? Infinity : -Infinity;
  rows.forEach((r, i) => { const v = parseFloat(r[key]); if (isNaN(v)) return; if (lower ? v < bv : v > bv) { bv = v; bi = i; } });
  return bi;
}
function numCell(v, frac, isBest, color){
  const w = isNaN(frac) ? 0 : Math.max(0, Math.min(1, frac)) * 100;
  return `<td class="num${isBest ? ' best' : ''}">${v}${isBest ? '<span class="bst">best</span>' : ''}<span class="mb"><i style="--c:${color || 'var(--ac)'}" data-w="${w.toFixed(1)}"></i></span></td>`;
}
function initTables(){
  const clsBody = document.getElementById('clsTableBody');
  const keys = ['Severe_Precision','Severe_Recall','Severe_F1','ROC_AUC','PR_AUC'];
  const best = Object.fromEntries(keys.map(k => [k, bestIdx(clsData, k, false)]));
  clsData.forEach((r, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="model">${r.Model}</td>
      <td class="type">${r.Type}</td>
      ${keys.map(k => numCell(r[k], parseFloat(r[k]), best[k] === i)).join('')}`;
    clsBody.appendChild(tr);
  });

  const regBody = document.getElementById('regTableBody');
  const rmseKey = "RMSE (µg/m³)", maeKey = "MAE (µg/m³)";
  const maxR = Math.max(...regData.map(r => parseFloat(r[rmseKey]) || 0));
  const maxM = Math.max(...regData.map(r => parseFloat(r[maeKey]) || 0));
  const bR = bestIdx(regData, rmseKey, true), bM = bestIdx(regData, maeKey, true), bQ = bestIdx(regData, 'R_Squared', false);
  regData.forEach((r, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="model">${r.Model}</td>
      ${numCell(r[rmseKey], parseFloat(r[rmseKey]) / maxR, bR === i, tok('--ac2'))}
      ${numCell(r[maeKey], parseFloat(r[maeKey]) / maxM, bM === i, tok('--ac2'))}
      ${numCell(r.R_Squared, parseFloat(r.R_Squared), bQ === i, tok('--aq-good'))}
      <td><span class="pill ${r.Beats_Baseline === 'Yes' ? 'yes' : 'no'}">${r.Beats_Baseline}</span></td>`;
    regBody.appendChild(tr);
  });

  // headline stats for Problem 2, read straight from the benchmark table
  const base = clsData.find(r => /persistence/i.test(r.Model));
  const top = clsData[best.ROC_AUC];
  const s = document.getElementById('p2Stats');
  if (top) s.innerHTML += `<div class="stat"><b>${top.ROC_AUC}</b><span>best ROC-AUC</span></div>`;
  if (base) s.innerHTML += `<div class="stat"><b style="color:var(--mu)">${base.ROC_AUC}</b><span>persistence ROC-AUC</span></div>`;
}
function animateBars(scope){
  const bars = scope.querySelectorAll('.mb i');
  requestAnimationFrame(() => requestAnimationFrame(() => bars.forEach(i => i.style.width = i.dataset.w + '%')));
}

/* ---------- Figure lightbox ---------- */
function initLightbox(){
  const lb = document.getElementById('lightbox'), img = document.getElementById('lbImg'), cap = document.getElementById('lbCap');
  const open = f => {
    const src = f.querySelector('img');
    img.src = src.src; img.alt = src.alt;
    cap.textContent = f.closest('.figure').querySelector('h3').textContent;
    lb.classList.add('open'); document.body.style.overflow = 'hidden';
  };
  const close = () => { lb.classList.remove('open'); document.body.style.overflow = ''; };
  document.querySelectorAll('.figure .frame').forEach(f => {
    f.addEventListener('click', () => open(f));
    f.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(f); } });
  });
  lb.addEventListener('click', close);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
}

window.onload = function(){
  initMap();
  initCityInspector();
  initTsPicker();
  updateTsChart();
  initTables();
  runSimulation();
  initLightbox();
  initNav();
  initReveal();
  if (REDUCED) document.querySelectorAll('.mb i').forEach(i => i.style.width = i.dataset.w + '%');
};
</script>
</body>
</html>
"""


def build_interactive_dashboard():
    print("=" * 60)
    print("BUILDING SELF-CONTAINED INTERACTIVE DASHBOARD")
    print("=" * 60)

    dash_dir = OUTPUTS_DIR / "dashboard"
    os.makedirs(dash_dir, exist_ok=True)

    cities_dict = load_cities()
    df_master = pd.read_csv(PROCESSED_DATA_DIR / "master.csv")
    df_cls = pd.read_csv(TABLES_DIR / "problem2_classification_comparison.csv")
    with open(TABLES_DIR / "severe_threshold.json") as fh:
        severe_thr = json.load(fh)["pm25_threshold"]
    df_reg = pd.read_csv(TABLES_DIR / "problem3_regression_comparison.csv")
    df_coef = pd.read_csv(TABLES_DIR / "problem3_econometric_coefficients.csv", index_col=0)
    df_clusters = pd.read_csv(TABLES_DIR / "problem1_city_cluster_assignments.csv")

    # 1. Base64 Images for 100% self-contained offline rendering
    img_fig11 = get_base64_image("fig11_pca_biplot_clusters.png")
    img_fig12 = get_base64_image("fig12_hierarchical_dendrogram.png")
    img_fig14 = get_base64_image("fig14_pr_roc_curves.png")
    img_fig15 = get_base64_image("fig15_confusion_matrices.png")
    img_fig17 = get_base64_image("fig17_coefficients_forest_plot.png")
    img_fig18 = get_base64_image("fig18_regression_diagnostics.png")

    # 2. City Metrics Summary for Map & Cards
    city_summaries = []
    for city, row in df_clusters.iterrows():
        c_name = row['city']
        c_meta = cities_dict.get(c_name, {})
        city_summaries.append({
            "city": c_name,
            "state": c_meta.get("state", "India"),
            "lat": c_meta.get("latitude", 20.0),
            "lon": c_meta.get("longitude", 78.0),
            "pm25": round(row['pm25_mean'], 1),
            "pm10": round(row['pm10_mean'], 1),
            "no2": round(row['no2_mean'], 1),
            "so2": round(row['so2_mean'], 1),
            "co": round(row['co_mean'], 1),
            "severe_share": round(row['severe_day_share'] * 100, 1),
            "cluster": row['Cluster_Profile_Name'],
            "stations": int(row.get('stations_count', 7))
        })

    # 3. Time-series data
    ts_data = {}
    for c in ['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata', 'Ahmedabad', 'Pune']:
        sub = df_master[df_master['city'] == c].sort_values('date')
        ts_data[c] = {
            "dates": sub['date'].tolist(),
            "pm25": sub['PM2.5'].round(1).tolist(),
            "aqi": sub['AQI'].fillna(-1).round(1).tolist(),
            "temp": sub['temp_mean'].round(1).tolist(),
            "ws": sub['wind_speed_mean'].round(1).tolist()
        }

    # 4. Embedded JSON blobs
    cities_json = json.dumps(city_summaries)
    ts_json = json.dumps(ts_data)
    cls_json = json.dumps(df_cls.to_dict(orient='records'))
    reg_json = json.dumps(df_reg.to_dict(orient='records'))

    replacements = {
        "%%CITIES_JSON%%": cities_json,
        "%%TS_JSON%%": ts_json,
        "%%CLS_JSON%%": cls_json,
        "%%REG_JSON%%": reg_json,
        "%%SEVERE_THR%%": str(severe_thr),
        "%%IMG_FIG11%%": img_fig11,
        "%%IMG_FIG12%%": img_fig12,
        "%%IMG_FIG14%%": img_fig14,
        "%%IMG_FIG15%%": img_fig15,
        "%%IMG_FIG17%%": img_fig17,
        "%%IMG_FIG18%%": img_fig18,
    }
    html_content = HTML_TEMPLATE
    for key, value in replacements.items():
        html_content = html_content.replace(key, value)

    out_html_path = dash_dir / "dashboard.html"
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Also save a copy directly in project root for 1-click convenience
    root_html_path = PROJECT_ROOT / "dashboard.html"
    with open(root_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[Dashboard] Standalone Self-Contained Dashboard saved to:\n  1. {out_html_path}\n  2. {root_html_path}")
    return out_html_path

if __name__ == "__main__":
    build_interactive_dashboard()
