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
const interleaved_to_planear = (array: Float32Array) => {
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

// Från modellens dokumentation (https://github.com/onnx/models/tree/main/vision/body_analysis/age_gender)
const TRAINING_INPUT_DATA_MEAN = 120.0; 

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
    const f32array = Float32Array.from(without_alpha, x => x - TRAINING_INPUT_DATA_MEAN);
    const channel_separated = interleaved_to_planear(f32array);
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
