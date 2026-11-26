# JJS

JJS is een native iOS-app in ontwikkeling. De app gebruikt geen Apple Intelligence,
Foundation Models, cloud-API of netwerkdienst voor antwoorden. Modelbestanden,
gesprekken, bijlagen en gegenereerde projecten blijven op het toestel totdat de
gebruiker zelf een exportactie kiest.

## Huidige status

GitHub Actions bouwt een **ongesigneerde** `JJS.ipa` in het artefact
`JJS-unsigned-IPA`. Voor installatie op een iPhone moet de gebruiker die IPA met
een geldige iOS-methode signeren. De huidige build is geen bevestiging dat ieder
GGUF-model werkt. De eigen inferentiecode is nog niet gevalideerd met een echt
gespreksmodel en implementeert slechts een deel van de GGML-opslagvormen en
transformerarchitecturen. Er is geen modelnaam-whitelist bij import, maar een
geïmporteerd model kan tijdens uitvoering nog op een ontbrekende tensorbewerking
stuiten.

De UI bevat lokale modelimport, gesprekken in SQLite, JSON-import en -export,
tekstextractie uit PDF en afbeeldingen, spraakherkenning op het toestel en
projectexport als ZIP. De bouwroute heeft geen externe app-dependencies.

## Bouwen

De workflow `.github/workflows/build-ipa.yml` gebruikt XcodeGen en Xcode op
`macos-latest`. Het resultaat staat onder de betreffende Actions-run als ZIP.
