# Kansenradar

AI-onderzoeksplatform dat businessopportuniteiten voor **België en Nederland** ontdekt,
onderzoekt, kritisch beoordeelt en rangschikt. Alle output is Nederlandstalig.

## Status

| Fase | Onderdeel | Status |
|---|---|---|
| 0 | Fundament: datamodel, bewijsregels, LLM-laag, kostenbewaking, database | ✅ klaar |
| 1 | Historische database + dedup (embeddings) | gepland |
| 2 | Ideeëngeneratie + originaliteitsmotor | gepland |
| 3 | Snelle screening | gepland |
| 4 | Marktonderzoek + bronverificatie | gepland |
| 5 | Concurrentieanalyse | gepland |
| 6 | Financiële analyse (deterministisch, 3 scenario's) | gepland |
| 7 | Kritische investeerder (red team) + validatietest | gepland |
| 8 | Scoring, betrouwbaarheid, rangschikking | gepland |
| 9 | Rapportage (14-puntenrapport) | gepland |
| 10 | Dashboard | gepland |
| 11 | Kalibratie op gouden set | gepland |

## Kernprincipes in de code

- **Bewijs is een datatype** (`domain/evidence.py`): een FEIT vereist bron-URL, letterlijk
  citaat, verificatie en ophaaldatum. Anders wordt het automatisch SCHATTING of HYPOTHESE.
  Verouderde feiten tellen als SCHATTING.
- **Eén LLM-toegangspunt** (`llm/client.py`): model per niveau (analyse = Opus 5.5,
  extractie = Sonnet 5.5, screening = Haiku 5.5), gestructureerde Pydantic-output,
  prompt caching, refusal-fallbacks, foutafhandeling.
- **Harde kostenlimiet per run** (`llm/budget.py`): een aanroep die de limiet zou kunnen
  overschrijden, wordt vooraf geweigerd.
- **Het LLM rekent niet**: financiële berekeningen gebeuren in Python (fase 6).

## Gebruik

```bash
pip install -e ".[dev]"
python -m pytest
```

Configuratie via omgevingsvariabelen met prefix `KR_` (zie `src/kansenradar/config.py`),
en `ANTHROPIC_API_KEY` voor de Claude API.
