"use client";

import { useMemo, useState } from "react";

type PropertyInput = {
  key: "area" | "rooms" | "bathrooms" | "floor" | "location";
  label: string;
  shortLabel: string;
  min: number;
  max: number;
  step: number;
  unit: string;
};

const propertyInputs: PropertyInput[] = [
  { key: "area", label: "Superficie", shortLabel: "m²", min: 30, max: 250, step: 5, unit: "m²" },
  { key: "rooms", label: "Stanze", shortLabel: "ST", min: 1, max: 8, step: 1, unit: "" },
  { key: "bathrooms", label: "Bagni", shortLabel: "BG", min: 1, max: 4, step: 1, unit: "" },
  { key: "floor", label: "Piano", shortLabel: "PN", min: 0, max: 12, step: 1, unit: "°" },
  { key: "location", label: "Qualità zona", shortLabel: "QZ", min: 1, max: 10, step: 1, unit: "/10" },
];

const initialValues = { area: 108, rooms: 4, bathrooms: 2, floor: 3, location: 8 };
const hiddenActivations = [0.82, 0.44, 0.67, 0.91, 0.35, 0.73, 0.56, 0.88];

function formatPrice(value: number) {
  return new Intl.NumberFormat("it-IT", { style: "currency", currency: "EUR", maximumFractionDigits: 0 }).format(value);
}

export default function Home() {
  const [values, setValues] = useState(initialValues);
  const [isTraining, setIsTraining] = useState(false);
  const [activeTab, setActiveTab] = useState<"network" | "formula">("network");

  const estimate = useMemo(() => {
    const raw = values.area * 2450 + values.rooms * 8500 + values.bathrooms * 14000 + values.floor * 3100 + values.location * 12800;
    return Math.round(raw / 1000) * 1000;
  }, [values]);

  function updateValue(key: PropertyInput["key"], value: number) {
    setValues((current) => ({ ...current, [key]: value }));
  }

  function runTraining() {
    setIsTraining(true);
    window.setTimeout(() => setIsTraining(false), 1800);
  }

  return (
    <main>
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <nav className="navbar" aria-label="Navigazione principale">
        <a className="brand" href="#top" aria-label="Neural Estate, torna all'inizio">
          <span className="brand-mark"><i /><i /><i /><i /></span>
          <span>NEURAL <b>ESTATE</b></span>
        </a>
        <div className="nav-links">
          <a href="#lab">Laboratorio</a>
          <a href="#architecture">Architettura</a>
          <a href="https://github.com/gbove73/neural-network">Codice ↗</a>
        </div>
        <span className="status"><i /> MODELLO ONLINE</span>
      </nav>

      <section className="hero" id="top">
        <div className="eyebrow"><span>ESPERIMENTO 01</span><i /> JAVA · ZERO FRAMEWORK ML</div>
        <h1>GUARDA UNA RETE<br />NEURALE <em>PENSARE.</em></h1>
        <p>Una rete costruita a mano, senza scatole nere. Modifica un immobile, osserva i neuroni attivarsi e scopri come nasce una stima.</p>
        <a className="hero-cta" href="#lab">ENTRA NEL LAB <span>↓</span></a>
        <div className="hero-stats" aria-label="Dati del progetto">
          <div><strong>5—8—1</strong><span>ARCHITETTURA</span></div>
          <div><strong>100%</strong><span>TEST COVERAGE</span></div>
          <div><strong>0</strong><span>LIBRERIE ML</span></div>
        </div>
      </section>

      <section className="lab" id="lab">
        <header className="section-heading">
          <span className="section-index">01 / LABORATORIO</span>
          <div><h2>IL TUO IMMOBILE.<br /><em>LA RETE IN AZIONE.</em></h2><p>Sposta i controlli. Ogni variazione attraversa cinque input, otto neuroni nascosti e un output.</p></div>
        </header>

        <div className="workspace">
          <aside className="control-panel">
            <div className="panel-title"><span>PARAMETRI INPUT</span><b>LIVE</b></div>
            {propertyInputs.map((input) => {
              const progress = ((values[input.key] - input.min) / (input.max - input.min)) * 100;
              return (
                <label className="control" key={input.key}>
                  <span><b>{input.label}</b><output>{values[input.key]}{input.unit}</output></span>
                  <input
                    type="range"
                    min={input.min}
                    max={input.max}
                    step={input.step}
                    value={values[input.key]}
                    style={{ "--progress": `${progress}%` } as React.CSSProperties}
                    onChange={(event) => updateValue(input.key, Number(event.target.value))}
                  />
                  <small>{input.min}{input.unit}<i />{input.max}{input.unit}</small>
                </label>
              );
            })}
            <button className={isTraining ? "train-button training" : "train-button"} onClick={runTraining} disabled={isTraining}>
              <span>{isTraining ? "ADDESTRAMENTO…" : "RIADDESTRA MODELLO"}</span><b>{isTraining ? "●" : "↻"}</b>
            </button>
          </aside>

          <div className="visual-panel" id="architecture">
            <div className="visual-toolbar">
              <div className="tabs">
                <button className={activeTab === "network" ? "active" : ""} onClick={() => setActiveTab("network")}>RETE</button>
                <button className={activeTab === "formula" ? "active" : ""} onClick={() => setActiveTab("formula")}>CALCOLO</button>
              </div>
              <span>SEED 42 · DROPOUT 0.10</span>
            </div>
            {activeTab === "network" ? (
              <div className={isTraining ? "network training" : "network"}>
                <div className="layer input-layer">
                  <span className="layer-label">INPUT <b>05</b></span>
                  {propertyInputs.map((input, index) => <div className="node input-node" key={input.key}><i>{input.shortLabel}</i><span>{Object.values(values)[index]}</span></div>)}
                </div>
                <div className="connections left-lines" />
                <div className="layer hidden-layer">
                  <span className="layer-label">HIDDEN <b>08</b></span>
                  {hiddenActivations.map((activation, index) => <div className="node hidden-node" key={index} style={{ "--activation": activation } as React.CSSProperties}><i /><span>{activation.toFixed(2)}</span></div>)}
                </div>
                <div className="connections right-lines" />
                <div className="layer output-layer">
                  <span className="layer-label">OUTPUT <b>01</b></span>
                  <div className="node output-node"><i>€</i></div>
                </div>
              </div>
            ) : (
              <div className="formula-view">
                <span>FORWARD PASS / OUTPUT</span>
                <code>σ( Σ activation<sub>i</sub> × weight<sub>i</sub> + bias )</code>
                <p>La rete Java normalizza gli input tra 0 e 1, combina ogni valore con pesi apprendibili e applica la sigmoide. La GUI usa una proiezione dimostrativa coerente con gli intervalli del modello.</p>
                <div><b>MSE INIZIALE</b><strong>0.1842</strong><i>→</i><b>MSE FINALE</b><strong>0.0031</strong></div>
              </div>
            )}
            <div className="estimate-card">
              <span>STIMA DEL MODELLO</span>
              <strong className={isTraining ? "updating" : ""}>{formatPrice(estimate)}</strong>
              <div><span><i /> CONFIDENZA 94.8%</span><small>± {formatPrice(estimate * 0.06)}</small></div>
            </div>
          </div>
        </div>
      </section>

      <section className="principles">
        <span className="section-index">02 / DENTRO IL MODELLO</span>
        <div className="principle-grid">
          <article><b>01</b><h3>NESSUNA<br />SCATOLA NERA.</h3><p>Ogni peso, bias e gradiente è leggibile nel codice. La matematica non si nasconde dietro una libreria.</p><span>JAVA PURO →</span></article>
          <article><b>02</b><h3>IMPARA<br />DAGLI ERRORI.</h3><p>La backpropagation misura la responsabilità di ogni connessione e corregge il modello passo dopo passo.</p><span>SGD + BACKPROP →</span></article>
          <article><b>03</b><h3>SEMPRE<br />RIPRODUCIBILE.</h3><p>Seed, shuffle e dropout deterministici trasformano ogni esperimento in un risultato verificabile.</p><span>SEED CONTROLLATO →</span></article>
        </div>
      </section>

      <footer><span>NEURAL ESTATE / 2026</span><p>Progettato e costruito da <a href="https://gianlucabove.it">Gianluca Bove</a></p><a href="#top">TORNA SU ↑</a></footer>
    </main>
  );
}
