# Kansenradar

AI-onderzoeksplatform dat businessopportuniteiten voor **België en Nederland** ontdekt,
onderzoekt, kritisch beoordeelt en rangschikt. Alle output is Nederlandstalig.

## Status

| Fase | Onderdeel | Status |
|---|---|---|
| 0 | Fundament: datamodel, bewijsregels, LLM-laag, kostenbewaking, database | ✅ klaar |
| 1 | Historische database + dedup (embeddings) | ✅ klaar |
| 2 | Ideeëngeneratie + originaliteitsmotor | ✅ klaar |
| 3 | Snelle screening | ✅ klaar |
| 4 | Marktonderzoek + bronverificatie | ✅ klaar |
| 5 | Concurrentieanalyse | ✅ klaar |
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

## Uitgangspunten voor de omgeving

- **€0 budget**: geen betaalde diensten of API's vereist. Betaalde providers zijn hooguit
  optionele extra's.
- **Draait lokaal op een Chromebook (16 GB RAM)** in de Linux-omgeving (Crostini), alleen CPU.
- **Werkt zonder online providers**: elke stap die een LLM kan gebruiken, heeft een
  terugvaloptie zonder LLM.

## Installatie

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,semantic]"
python -m pytest
```

`semantic` installeert fastembed (ONNX, geen PyTorch). Het meertalige model
(`paraphrase-multilingual-MiniLM-L12-v2`, ~220 MB, ~0,5 GB RAM) wordt bij het eerste
gebruik eenmalig gedownload. Zonder dat pakket of zonder internet valt het systeem
automatisch terug op de lexicale embedder.

## Duplicaatcontrole (fase 1)

Elk nieuw idee wordt vergeleken met de historische database:

1. Identiek aan een eerder idee → **DUPLICAAT**
2. Zeer hoge gelijkenis → **DUPLICAAT**
3. Lage gelijkenis → **UNIEK**
4. Grijze zone → een rechter beslist: **DUPLICAAT** of **VERWANT**
   - zonder LLM: doelgroep, probleem én oplossing moeten elk overeenkomen
   - met LLM (optioneel): het model oordeelt; valt het weg, dan neemt de heuristiek over

Bij twijfel wordt een idee als **VERWANT** gemarkeerd en niet weggegooid. Duplicaten worden
ook opgeslagen, zodat de historiek volledig is, en eerdere uitkomsten ("eerder afgewezen")
worden getoond.

Drempels kalibreren op je eigen machine (aanbevolen na installatie van `semantic`):

```bash
python -m kansenradar.history.calibrate
```

Na het wisselen van embedder: `kansenradar.history.reindex()` herberekent de embeddings.

## Ideeën genereren (fase 2)

```bash
python -m kansenradar genereer --aantal 10
python -m kansenradar genereer --aantal 5 --sector logistiek --strategie INTERNATIONAL_ARBITRAGE
python -m kansenradar ideeen            # nieuwe ideeën (--alle: ook afgewezen/duplicaten)
python -m kansenradar toon <idee-id>
python -m kansenradar status            # welke provider en embedder actief zijn
```

**Originaliteitsmotor**: 462 invalshoeken uit zes strategieën (probleemgedreven, ongebruikelijke
niche, nieuwe technologie, internationaal model, marktcombinatie, fundamentele verbetering),
plus regelgevende triggers. Invalshoeken worden gespreid over de strategieën en nooit twee keer
gebruikt. De catalogus (`generation/catalog.py`) bevat uitsluitend **hypotheses**; niets daarvan
telt als bewijs.

**Originaliteitsscore** (0-10) is een apart label en telt **niet** mee in de commerciële score.

**Generiek-filter** (zonder LLM): te brede doelgroep, modewoorden en clichéformules
("Uber voor..."), vaag probleem, geen focus op BE/NL. Afgewezen ideeën blijven in de historiek.

**Twee modi**:
- *LLM-modus*: rijke ideeën via een gratis provider. Valt de provider weg, dan schakelt de
  generator per invalshoek over op sjablonen.
- *Sjabloonmodus* (geen LLM): probleem x passende technologie uit de catalogus. Eenvoudiger
  ideeën, beperkt tot 24 (één per catalogusprobleem); daarna meldt het systeem dat de catalogus
  uitgeput is.

## Snelle screening (fase 3)

```bash
python -m kansenradar screen
```

Elk nieuw idee wordt gescoord (1-5) op: **pijn en urgentie, betalingsbereidheid, bereikbaarheid
van klanten, marktomvang BE/NL, concurrentiedruk en uitvoerbaarheid** (incl. regelgeving).

- De screener levert scores; **de code beslist** (`screening/decision.py`):
  - score 1 op pijn, betalingsbereidheid, concurrentie of uitvoerbaarheid → **AFGEWEZEN**
  - gewogen score < 2,6 → **AFGEWEZEN**; ≥ 3,4 zonder dealbreaker → **DOOR**; anders **TWIJFEL**
- Betrouwbaarheid is bij screening nooit "hoog": er zijn nog geen geverifieerde bronnen.
  Alle bevindingen worden als HYPOTHESE opgeslagen, met onderzoeksvragen voor fase 4.
- **Zonder LLM**: regels op de tekst (pijnsignalen, B2B/B2C, verdienmodel, regelgeving,
  gezondheidsgegevens, hardware, marktplaats). Wat de regels niet kunnen weten, staat als
  "onbekend" (neutrale score 3).
- **Met LLM**: per 5 ideeën één aanroep; zekere regelbevindingen (geen verdienmodel) kan het
  model niet overrulen. Mislukt de LLM, dan nemen de regels het over.
- **Volgorde voor marktonderzoek**: commerciële score beslist; originaliteit is enkel een
  tiebreaker (weegt nooit zwaarder dan één punt verschil op één criterium).

## Marktonderzoek (fase 4)

```bash
python -m kansenradar onderzoek --top 3
python -m kansenradar onderzoek --idee <id> --url https://... --url https://...
```

Tien onderzoeksonderwerpen per idee (vraag, klanten, probleem, huidige oplossing, concurrenten,
marktgrootte, prijzen, bereikbaarheid, regelgeving, trends). Bronnen, allemaal gratis:

- **Eurostat** (officieel, geen sleutel): aantal ondernemingen per sector (NACE) in BE en NL.
  De doelgroep-naar-sector-koppeling wordt apart als SCHATTING vastgelegd, met beperkingen.
- **Webzoeken** (optioneel): `KR_SEARCH_PROVIDER=searxng` (lokaal) of `tavily`
  (gratis versie, 1.000 zoekopdrachten/maand, geen creditcard; `TAVILY_API_KEY`).
  Brave Search vraagt sinds februari 2026 een creditcard en is daarom niet opgenomen.
- **Eigen bronnen**: `--url` werkt ook volledig zonder zoekfunctie.

**Bronverificatie**: pagina's worden opgehaald met respect voor robots.txt. Elk citaat moet
letterlijk (of nagenoeg letterlijk) in de bron staan; cijfers moeten exact overeenkomen.
Niet-verifieerbare citaten worden verworpen en geteld. Onderwerpen zonder geverifieerd FEIT
verschijnen als open vragen.

## Concurrentieanalyse (fase 5)

```bash
python -m kansenradar concurrentie
```

Kandidaten uit gerichte zoekopdrachten en uit bedrijfsbronnen van het marktonderzoek; media,
fora, reviewsites en overheden zijn bronnen, geen concurrenten. Prijzen komen alleen van de
eigen pagina's van concurrenten (ook /prijzen, /pricing, /tarieven), omgerekend naar per maand,
en worden als FEIT met citaat opgeslagen. Verzadiging: 0-1 directe concurrenten = laag,
2-5 = gemiddeld, 6+ = hoog. **Zonder webzoekfunctie is de verzadiging "onbekend"**, niet
"laag". Met LLM: indeling direct/indirect, sterktes, zwaktes en gaten (hypotheses); offline
mag het LLM bekende concurrenten noemen, maar enkel als HYPOTHESE.

## LLM-providers (allemaal optioneel)

| Provider | Kosten | Sleutel | Opmerking |
|---|---|---|---|
| `none` (standaard) | gratis | — | werkt volledig zonder LLM, met beperkte generatie |
| `groq` | gratis versie | `GROQ_API_KEY` | snel; limieten per minuut/dag |
| `gemini` | gratis versie | `GEMINI_API_KEY` | invoer kan door Google gebruikt worden om producten te verbeteren |
| `openrouter` | gratis `:free`-modellen | `OPENROUTER_API_KEY` | daglimieten |
| `ollama` | gratis, lokaal | — | offline; traag op een Chromebook (~5 GB RAM) |
| `anthropic` | betaald | `ANTHROPIC_API_KEY` | optioneel, niet nodig |

Kopieer `.env.example` naar `.env` en vul de provider in. Modelnamen veranderen vaak; pas ze aan
met `KR_LLM_MODEL_*` als een provider "model not found" meldt.

Configuratie via omgevingsvariabelen met prefix `KR_` (zie `src/kansenradar/config.py`).
