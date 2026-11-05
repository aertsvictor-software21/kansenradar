# Kalibratie

`archetypes.json` bevat 8 **synthetische** gevallen. Ze toetsen of de beoordelingslogica de
juiste kant op beslist (sterk idee met bewijs → kansrijk; verzadigde markt → afwijzen; mooi
verhaal zonder bewijs → meer onderzoek nodig). Ze bewijzen **niet** dat de scores in de echte
wereld kloppen.

Voeg echte gevallen toe in een eigen bestand (zelfde formaat) en draai:

```bash
python -m kansenradar kalibreer --bestand kalibratie/mijn_gevallen.json
```

Velden per geval: `naam`, `omschrijving`, `probleem`, `doelgroep`, `verdienmodel`, optioneel
`sector`, `strategie`, `onderscheidend`, `waarom_nu`, `feiten` (onderwerp → bronkwaliteit, bv.
`"DEMAND": "OFFICIAL"`), `concurrentie` (`verzadiging`, `direct`, `prijzen` per maand),
`financieel` (gevalideerde cijfers, bv. uit je validatietest: `cac`, `churn`, `price`,
`market`; met `basis`: `FEIT` of `SCHATTING`), `verwacht` (lijst aanvaardbare conclusies) en
`synthetisch` (`false` voor echte gevallen).

Goede bronnen voor echte gevallen: je eigen afgeronde validatietests (uitkomst bekend), en
bekende Belgische/Nederlandse startups waarvan het succes of falen publiek gedocumenteerd is.
