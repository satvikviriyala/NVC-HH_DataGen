# NVC-HH Data Gen: Ontologies & Prompts

This repository contains the ontologies, schemas, and prompts used for generating the **NVC-HH (Nonviolent Communication - Helpful & Harmless)** dataset. The core objective is to extract structured **OFNR** (Observations, Feelings, Needs, Requests) from dialogue transcripts while strictly mitigating hallmark hallucinations (e.g., confusing strategies with needs, or evaluations with feelings).

## Ontologies Structure

The `ontologies/` folder contains the canonical truth-files that steer the model's behavior. Below is an explanation of each file:

### Core OFNR Ontologies

- **`needs_ontology.json`**
  The canonical list of **universal human needs** (based on Max-Neef and Rosenberg). This file is "locked" — the model must only output needs from this list. It includes `taxonomy` (grouping needs by category like "Connection" or "Autonomy") and `anti_examples` to help distinguish needs from strategies.

- **`feelings_ontology.json`**
  A verified inventory of **true affective states** (e.g., "sad", "joyful", "anxious"). It organizes feelings by valence (needs met/unmet) and arousal/intensity. Crucially, it excludes "pseudo-feelings" to prevent the model from validating blame narratives.

- **`request_quality_ontology.json`**
  Defines the criteria for high-quality **Requests**. It includes scoring rules for *actionability*, *specificity*, and *positive phrasing*, and defines "anti-patterns" like demands or vague complaints. Used to score and rewrite implicit requests.

### Hallucination Mitigation & Filtering

- **`pseudo_feelings_lexicon.json`**
  A dictionary of **pseudo-feelings** (evaluative words like "ignored", "betrayed", "manipulated") that mimic feelings but imply external blame. The file provides translation rules to map these tokens into their underlying *true feelings* and *needs*.

- **`judgment_markers_ontology.json`**
  A taxonomy of **judgmental language** (e.g., insults, absolute generalizations like "always/never", moralistic labels). Used to sanitize *Observations*—ensuring they remain neutral and "camera-like" rather than evaluative.

- **`plato_strategy_filter.json`**
  Implements the **PLATO Test** (Person, Location, Action, Time, Object). Any candidate "need" containing these elements is flagged as a **strategy** (not a universal need) and must be moved to the `explicit_needs` or `requests` fields.

- **`somatic_markers_ontology.json`**
  A list of **bodily sensations** (e.g., "tight chest", "shaky hands"). This distinguishes physical symptoms from emotional states, helping the model ground its feeling inferences in physiological evidence without over-interpreting.

### Schema

- **`schema_ofnr.json`**
  The master **JSON schema** for the output dataset. It defines the strict structure the model must output, including fields for `ofnr` extraction, `safety` checks, `quality` confidence scores, and metadata.
