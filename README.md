# Förberedelse:
1. Installera npm
2. Kör `npm install -g npx` för att installera npx
3. Hitta och ladda ner en testbild på en person du vill åldersbestämma.
4. Hämta den färdigtränade modellen från [https://github.com/onnx/models/blob/main/vision/body_analysis/age_gender/models/age_googlenet.onnx](https://github.com/onnx/models/blob/main/vision/body_analysis/age_gender/models/age_googlenet.onnx)

# Ett enkelt UI
Vi börjar med att skapa ett nytt react-typescript project och går in i det.

```bash
npx create-react-app onnx-demo --template typescript
cd onnx-demo
```

Vi testar att projektet sattes upp ordentligt genom att starta development servern som följer med 
react projektet.

```bash
npm start
```

Om allt fungerar som det ska, startar developmentservern och en ny flik att öppnas i din webbläsare 
med ett exempelprojekt som skapades av `create-react-app`.
Avsluta developmentservern med `Ctrl-c`.

För att köra en maskininlärningsmodell i webbläsaren kommer vi att behöva ett javascriptbibliotek som heter `onnxruntime-web`.
Biblioteket används för att deserialisera och köra `.onnx` modeller som är ett vanligt filformat för att spara tränade modeller.

Vi använder npm för att installera biblioteket lokalt till det här projektet:
```bash
npm install onnxruntime-web
```

Vi måste nu kopiera vår exempelbild och `.onnx` modellen som vi hämtade i förberedelse-steget till `public/example_image.jpg` respektive `public/age_googlenet.onnx`.
Detta gör att development-servern automatiskt servar dessa filer statiskt.

Vi börjar med ett enkelt UI med en rubrik, bilden som vi vill köra modellen på samt en knapp för att köra modellen.
Vi öppnar filen `src/App.tsx` och byter ut innehållet med:
```react-typescript
import { InferenceSession, Tensor } from "onnxruntime-web";
import { useRef, useState } from "react";

function App() {
  return (
    <div>
      <h1>Age Estimator</h1>
      <img id="input_image" src="example_image.jpg" alt="example" />
      <br/>
      <button id="estimate_age" type="button"> Estimate Age </button>
    </div>
  );
}

export default App;
```
De två första raderna importerar funktioner som vi kommer att behöva framöver.
För att komma åt bilden programmatiskt använder vi en så kallad ref-hook. Det är för att vi ska kunna hämta bildens innehåll till modellen.
Vi måste också sätta `crossOrigin` attributet på bildtaggen för att få browsern att låta oss använda bildens innehåll.

Som första rad i `App()`-funktionen lägger vi till
```typescript
const input_image = useRef<HTMLImageElement>(null);
```

`<img>` taggen blir nu istället
```
<img id="input_image" src="example_image.jpg" alt="example" crossOrigin="anonymous" ref={input_image}/>
```
# Förprocessering av indata-bilden
Modellen som vi kommer att köra förväntar sig ett visst format på indatan. Mer specifikt måste vi:
1. Skala om bilden till 224x224 pixlar.
2. Plocka ut bilddatan som en array av pixlar (av typen Uint8ClampedArray).
3. Ta bort alpha kanalen som representerar pixlarnas transparens.
4. Konvertera bilden till en array av flyttal (Float32Array).
5. Konvertera bilden från  "interleaved"-rgb format till "planar"-rgb format. I interleaved formatet är pixeldatan strukturerad så att var tredje element tillhör samma kanal. En 2x2 bild har alltså den underliggande datastrukturen
`RGBRGBRGBRGB`. Vi måste konvertera formatet så att varje kanal ligger för sig. För 2x2 exemplet blir det alltså `RRRRGGGGBBBB`. 
6. Subtrahera en normaliseringskonstant från varje pixel för att matcha formatet som modellen tränades med. I vårt fall måste vi subtrahera varje pixelintensitet med `120`.
Slutresultatet av denna förprocessering kommer att vara en array av typen Float32Array med storleken `3 * 224 * 224 = 150528` med pixlarna representerade i planarformat. 

Skillnaden mellan interleaved RGBA och planar RGB illustreras tydligast genom ett exempel.
Här tänker vi oss att vi har en 2x2 pixlar bild som ska konverteras från RGBA interleaved till RGB planar:
![interleaved-vs-planar](interleaved-vs-planar.png)
Slutresultatet är alltså att alla kanaler lagras för sig i röd-grön-blå ordning. 

Vi gör allt detta i en `preprocess` funktion som körs när bilden laddas:
```typescript
  const [preprocessed, set_preprocessed] = useState<Uint8ClampedArray>();
  const preprocess = () => {
    const canvas = document.createElement("canvas");

    const img_w = input_image.current!.width;
    const img_h = input_image.current!.height;
    canvas.width = 224;
    canvas.height = 224;

    // 1. Skala om bilden till 224x224
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(input_image.current!, 0, 0, img_w, img_h, 0, 0, 224, 224);

    // 2. Plocka ut pixeldatan från vårt canvas-element
    const array = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

    // 3. Ta bort alphakanalen
    const without_alpha = remove_alpha(array);

    // 4. och 5. Konvertera till flyttalsarray och subtrahera 120
    const f32array = Float32Array.from(without_alpha, x => x - 120);

    // 6. Konvertera till planar-format
    const channel_separated = separate_channels(f32array);

    set_preprocessed(channel_separated);
  }
  ...
  <img ... onLoad={preprocess}/>
```
Detaljerna är inte jätteviktiga här, men vi skapar en canvas och ritar bilden - skalad - på canvasen för att kunna läsa ut pixeldatan med `getImageData`. 

Sedan använder vi funktionen `remove_alpha` för att ta bort alpha-kanalen.

Därefter måste vi konvertera datan från en array av typen `Uint8ClampedArray` som är en array med 8-bitarselement till en `Float32Array` med flyttal som modellen accepterar. I samma veva subtraherar vi `120.0` från varje pixel för att matcha träningsdatan.

Vi konverterar från interleaved till planar med `separate_channels()` funktionen som vi implementerar nedan. 

Sist men inte minst sparar vi resultatet i react statet via `set_preprocessed`.

Vi måste nu implementera funktionerna `remove_alpha` och `separate_channels`. 
Detta gör vi som globala funktioner (utanför `App()`):
```typescript
/*
 * Remove the alpha channel from an interleaved RGBA imagedata array.
 */
const remove_alpha = (array: Uint8ClampedArray) => {
  const result = new Uint8ClampedArray(array.length / 4 * 3);
  for (let i = 0; i < array.length; i += 4) {
    result[i / 4 * 3] = array[i];         // R
    result[i / 4 * 3 + 1] = array[i + 1]; // G
    result[i / 4 * 3 + 2] = array[i + 2]; // B
  }
  return result;
}

/* 
 * Convert from interleaved RGB to planar RGB.
 */
const separate_channels = (array: Float32Array) => {
  const plane_size = array.length / 3;
  const result = new Float32Array(array.length);
  for (let i = 0; i < plane_size; i++) {
    result[i] = array[i * 3];
    result[i + plane_size] = array[i * 3 + 1];
    result[i + plane_size * 2] = array[i * 3 + 2];
  }
  return result;
}
```
Båda funktionerna itererar helt enkelt igenom alla pixlar i arrayen och sparar i en ny array med den önskade strukturen.

# Använda modellen
I det här steget kommer vi att:
1. Deserialisera en färdigtränad modell från vår `age_googlenet.onnx` fil.
2. Skapa en så kallad Tensor (en onnx-typ som beskriver en n-dimensionell array) från vår indata.
3. Applicera modellen på tensorn.

Vi vill göra åldersestimeringen när användaren trycker på "Estimate Age"-knappen.
Därför skapar vi en ny funktion och lägger till den som handler till knappens `onClick` event.
```typescript 
const estimate_age = async () => {}
...
    <button id="estimate_age" type="button" onClick={estimate_age}> Estimate Age </button>
```
Vi markerar funktionen med `async` för att kunna använda `await` för att vänta på deserialiseringen och körningen av modellen med onnxruntime som är asynkrona anrop.

Det första vi vill göra i `estimate_age` är att deserialisera modellen. Detta gör vi med ett anrop till `InferenceSession.create(..)`:
```typescript 
const model = await InferenceSession.create('age_googlenet.onnx', { executionProviders: ['webgl']});
```

Nästa steg är att skapa en så kallad `Tensor` från vår indata. En tensor är en multidimensionell array (till exmpel en bild) som onnx-modellen accepterar som indata. Här måste vi också specificera formen på tensorn eftersom vår indata är en enkel Float array 
utan storleksinformation.
```typescript
const tensor = new Tensor(preprocessed!, [1, 3, 224, 224]);
```
Den första ettan i det andra argumentet kan verka lite konstig, men den representerar faktumet att vi endast vill utföra estimeringen på en bild och inte en lista med bilder. Trean representerar tre kanalerna R, G och B och de två sista elementen i listan är y respektive x dimensionerna i bilden.

Nu kan vi applicera modellen på tensorn och få ut ett resultat:
```typescript 
const results = await model.run({input: tensor});
const output = results['loss3/loss3_Y'].data;
console.log(output);
```
`loss3/loss3_Y` är en referens till utdatalagret i modellfilen och motsvarar en output för modellen. Om du använder andra modeller måste du se till att du 
specificerar rätt output här.

Om vi nu testar att köra projektet med `npm start` och klickar på "Estimate Age" så ska vi om allt gått rätt få ut en lista med sannorlikheter 
i browserterminalen. I mitt fall får jag ut:
```
Float32Array(8) [ 0.007054295856505632, 0.014829231426119804, 0.08670540153980255, 0.03400164470076561, 0.7336888313293457, 0.0609135702252388, 0.053118083626031876, 0.00968888122588396 ]
```
Här representerar varje element sannorlikheten för ett visst åldersintervall. Vi kan se att det femte intervallet (index 4) har högst sannorlikhet 
enligt modellen. Detta motsvarar åldersintervallet 25-32 år enligt modellens dokumentation.

# Presentation av resultatet
Ny återstår bara att presentera resultatet för användaren. Detta kan göras på många olika sätt men vi gör det enkelt för oss själva och 
skriver ut intervallet med högst sannorlikhet.
För att skriva ut ett snyggt intervall behöver vi en mappning mellan vår output-array och ålderintervall. 
Vi kan åstadkomma detta med en enkel lista av strängar:
```typescript
const AGE_INTERVALS = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100'];
```
Värdena är tagna från modellens dokumentationssida.

Nu måste vi bara hitta indexet för det intervall med den högsta sannorlikheten och indexera vår `AGE_INTERVALS` med detta index.
```typescript
const highest_probability_index = argmax(output as Float32Array);
const age_interval = AGE_INTERVALS[highest_probability_index];
```
Här måste vi hjälpa typescript genom att explicit kasta outputen till en Float32Array. Detta är bara för att hålla 
typescript-typcheckingen glad och inget som påverkar logiken.

Funktionen `argmax` måste vi implementera själva. Den ska returnera indexet av det element som har störst värde:
```typescript
const argmax = (array: Float32Array) => {
  let max = array[0];
  let max_index = 0;
  for (let i = 1; i < array.length; i++) {
    if (array[i] > max) {
      max = array[i];
      max_index = i;
    }
  }
  return max_index;
}
```

För att presentera resultatet sparar vi `age_interval` som ett react state och presenterar resultatet i en enkel `<label>` komponent:
```react-typescript
const [estimated_age, set_estimated_age] = useState<string>();
...
...
// I estimate_age() funktionen
set_estimated_age(age_interval);
...
...
// Efter <button> komponenten
<label>Estimated Age: {estimated_age}</label>
```
Om vi startar testservern igen (`npm start`) och klickar på "Estimate Age" knappen igen får vi ett åldersintervall presenterat i UI:t.


# Fortsättningsideer
- Det är fullt möjligt att köra modellen på resultatet från en kameraström, eller låta användaren välja bild från sitt filsystem.
- Nu laddas och deserialiseras modellen varje gång användaren trycker på "Estimate Age". Detta är inte nödvändigt, utan det bör göras direkt när sidan laddar.
- På [https://github.com/onnx/models/tree/main](https://github.com/onnx/models/tree/main) finns en uppsjö av andra intressanta modeller i .onnx format som kan testas.
- Med bibliotek som PyTorch eller Tensorflow kan du träna egna modeller och exportera till .onnx. Dessa modeller går utmärkt att använda på samma sätt som vi gjort i det här exemplet.
- Man skulle kunna visualisera resultatet snyggare. Till exempel kan man presentera sannorlikheterna för olika intervall i en "bar chart".

# Hela källkoden för `App.tsx`
Här finns den fullständiga lösningen med all kod för `App.tsx`:
```typescript
import { InferenceSession, Tensor } from "onnxruntime-web";
import { useRef, useState } from "react";

/*
 * Remove the alpha channel from an interleaved RGBA imagedata array.
 */
const remove_alpha = (array: Uint8ClampedArray) => {
  const result = new Uint8ClampedArray(array.length / 4 * 3);
  for (let i = 0; i < array.length; i += 4) {
    result[i / 4 * 3] = array[i];         // R
    result[i / 4 * 3 + 1] = array[i + 1]; // G
    result[i / 4 * 3 + 2] = array[i + 2]; // B
  }
  return result;
}

/* 
 * Convert from interleaved RGB to planar RGB.
 */
const separate_channels = (array: Float32Array) => {
  const plane_size = array.length / 3;
  const result = new Float32Array(array.length);
  for (let i = 0; i < plane_size; i++) {
    result[i] = array[i * 3];
    result[i + plane_size] = array[i * 3 + 1];
    result[i + plane_size * 2] = array[i * 3 + 2];
  }
  return result;
}

/*
 * Find the index of the largest element in an array.
 */
const argmax = (array: Float32Array) => {
  let max = array[0];
  let max_index = 0;
  for (let i = 1; i < array.length; i++) {
    if (array[i] > max) {
      max = array[i];
      max_index = i;
    }
  }
  return max_index;
}

const AGE_INTERVALS = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100'];

function App() {
  const input_image = useRef<HTMLImageElement>(null);
  const [preprocessed, set_preprocessed] = useState<Float32Array>();
  const [estimated_age, set_estimated_age] = useState<string>();

  const preprocess = () => {
    const canvas = document.createElement("canvas");
    const img_w = input_image.current!.width;
    const img_h = input_image.current!.height;
    canvas.width = 224;
    canvas.height = 224;

    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(input_image.current!, 0, 0, img_w, img_h, 0, 0, 224, 224);
    const array = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const without_alpha = remove_alpha(array);
    const f32array = Float32Array.from(without_alpha, x => x - 120);
    const channel_separated = separate_channels(f32array);

    set_preprocessed(channel_separated);
  }


  const estimate_age = async () => {
    const model = await InferenceSession.create('age_googlenet.onnx', { executionProviders: ['webgl'], graphOptimizationLevel: 'all' });
    const tensor = new Tensor(preprocessed!, [1, 3, 224, 224]);
    const results = await model.run({ input: tensor });
    const output = results['loss3/loss3_Y'].data;

    const highest_prob_index = argmax(output as Float32Array);
    const age_interval = AGE_INTERVALS[highest_prob_index];
    set_estimated_age(age_interval);
  }

  return (
    <div>
      <h1>Age Estimator</h1>
      <img
        id="input_image"
        src="example_image.jpg"
        alt="example"
        crossOrigin="anonymous"
        ref={input_image}
        onLoad={preprocess}
      />
      <br />
      <button id="estimate_age" type="button" onClick={estimate_age}> Estimate Age </button>
      <label>Estimated age: {estimated_age}</label>
    </div>
  );
}

export default App;
```
