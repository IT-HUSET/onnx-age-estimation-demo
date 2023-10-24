# Labbinstruktioner
Den här filen innehåller övergripande instruktionerna för labben. Om du fastnar eller vill ha ett lösningsförslag direkt kan du kolla i filen `README.md` som innehåller lösningsförslag och fullständig kod för labben. Varje steg länkar också till ett lösningsförslag som du kan kika på om du fastnar.

## Målbild
I den här labben är tanken att vi ska bygga ett system för att estimera ålder från en porträttbild.
Vi kommer att göra detta direkt i webbläsaren med hjälp av ett ramverk som heter `onnxruntime-web`.
Det låter oss utföra inferens med en deep learning-modell som har förtränats för att estimera ålder. 

<!-- ![goal](media/goal.png) -->
<img src="media/goal.png" width=700 style="display: block; margin: 0 auto;"/>
<br/>
Den största delen av labben kommer att gå ut på att manipulera indatabilden i ett förprocesseringssteg.
Eftersom modellen vi använder har tränats på en viss typ av bilder (storlek, pixelintensitet, bildformat, motivets position, etc)
är det extremt viktigt att se till att den data vi skickar in matchar detta så nära som det går. 
Annars kan vi inte räkna med att få ett korrekt resultat från modellen.
Det framgår lite tydligare i bilden nedan att förprocesseringen är den största delen av labben.

<br/>
<br/>

<!-- ![preprocessing](media/preprocessing-pipeline.png) -->
<img src="media/preprocessing-pipeline.png" width=700 style="display: block; margin: 0 auto; marginVertical: 10"/>
<!-- <img src="media/preprocessing-pipeline.png" width=700/> -->

<br/>
När vi är klara med förprocesseringen deserialiserar vi modellen, anropar den med vår indata och presenterar resultatet.


## Förberedelser ([lösningsförslag](README.md#Förberedelser))
1. Installera `npm` och `npx`.
2. Hämta en testbild i form av ett porträtt du vill åldersbestämma.
3. Hämta den färdigtränade modellen från [https://github.com/onnx/models/blob/main/vision/body_analysis/age_gender/models/age_googlenet.onnx](https://github.com/onnx/models/blob/main/vision/body_analysis/age_gender/models/age_googlenet.onnx)

## Ett enkelt UI  ([lösningsförslag](README.md#ett-enkelt-ui))
I det här steget ska du:
- Installera biblioteket `onnxruntime-web` till projektet.
- Skapa ett enkelt UI som visar bilden som du har laddat ned och en knapp med texten "Estimate Age".
- Skapa en så kallad react ref-hook som refererar till img elementet så att vi senare kan plocka ut pixeldatan från bilden.

## Förprocessering av indatabilden ([lösningsförslag](README.md#förprocessering-av-indata-bilden))
Nu måste vi plocka ut pixeldatan från img-taggen och bearbeta den så att den matchar formatet som modellen tränades på.
I slutet av det här steget vill vi ha en 224x224 planar RGB bild i float32 format som vi kan skicka till vår modell.
Vi kommer alltså att behöva konvertera från interleaved RGBA (formatet som vi får av browsern) och planar RGB.
Skillnaden mellan interleaved formatet och planar formatet visualiseras i nedanstående bild för en 2x2 pixlar bild.
![interleaved-vs-planar](media/interleaved-vs-planar.png)

I det här steget måste vi också subtrahera en normaliseringskonstant från alla pixlar i bilden. 
Det gör vi för att matcha hur träningsdatan har förprocesserats. Information om detta hittas i [dokumentationen](https://github.com/onnx/models/tree/main/vision/body_analysis/age_gender) för modellen.

## Använda modellen (inferens) ([lösningsförslag](README.md#använda-modellen-inferens))
Nu har vi äntligen kommit så långt att vi kan anropa modellen med vår bild. Detta steg kallas för inferens och går ut på att skicka vår förprocesserade data till modellen och få ut ett resultat.

I det här steget kommer vi att:
1. Deserialisera en färdigtränad modell från vår age_googlenet.onnx fil som innehåller den färdigtränade modellens vikter.
2. Skapa en Tensor från vår indata med rätt dimensioner.
3. Applicera modellen på den skapade tensorn.

Allt detta görs med `onnxruntime-web` biblioteket via `InferenceSession` och `Tensor` objekten.
