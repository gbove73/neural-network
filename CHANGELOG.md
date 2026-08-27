# Changelog

Tutte le modifiche rilevanti del progetto sono documentate in questo file.

Il formato segue [Keep a Changelog](https://keepachangelog.com/it-IT/1.1.0/) e il progetto adotta [Semantic Versioning](https://semver.org/lang/it/).

## [Unreleased]

### Added

- Interfaccia Next.js responsive per esplorare visivamente la rete neurale e simulare una stima immobiliare.
- Esportazione statica configurata per la pubblicazione in `gianlucabove.it/neural-network`.
- Pipeline GitHub Actions che verifica il modello Java, controlla e compila la GUI, quindi distribuisce il sito su `main` tramite SSH.

## [1.0.0] - 2026-07-17

### Added

- Bias apprendibili per lo strato nascosto e lo strato di output.
- Configurazione immutabile della rete con seed riproducibile.
- Inizializzazione Xavier dei pesi.
- Inverted dropout coerente tra forward pass e backpropagation.
- Configurazione del training con shuffle deterministico e frequenza delle metriche.
- Risultato del training con MSE iniziale, finale e storico per epoca.
- Valore di dominio `PropertyFeatures` per rappresentare un immobile senza parametri posizionali ambigui.
- Normalizzatori addestrabili sul solo training set e politiche esplicite per i valori fuori range.
- Gradient checking numerico e test distinti per matematica, applicazione, dominio, adapter e normalizzazione.
- Controllo JaCoCo obbligatorio al 100% per istruzioni e rami.
- Configurazione del logging che evita output dettagliato per ogni operazione numerica.
- Documentazione divulgativa di formule, architettura, limiti e strategia di test.

### Changed

- Corretta la backpropagation includendo il delta completo dello strato di output.
- Il dropout viene ora applicato durante il forward pass di training e riutilizza la stessa maschera nel calcolo del gradiente.
- `feedForward` restituisce un risultato indipendente dallo stato interno della rete.
- Gli esempi di training possono essere mescolati a ogni epoca.
- Il metodo `train` restituisce metriche osservabili invece di comunicare il risultato soltanto tramite log.
- La valutazione rifiuta dataset vuoti, prezzi non positivi e caratteristiche immobiliari non valide.
- Le dipendenze JUnit sono state rese coerenti e riproducibili.
- Il progetto è versionato come `1.0.0`.

### Fixed

- Eliminata l’incompatibilità tra versioni JUnit che impediva l’esecuzione dei test parametrizzati.
- Corretta la propagazione incompleta del gradiente dallo strato di output allo strato nascosto.
- Eliminata la modifica accidentale dello stato interno tramite l’array restituito da `feedForward`.
- Eliminati i test probabilistici e le inizializzazioni casuali non riproducibili.

[Unreleased]: https://github.com/gbove73/neural-network/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/gbove73/neural-network/releases/tag/v1.0.0
