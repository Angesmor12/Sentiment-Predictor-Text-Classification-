let normalizationData = null;
let allow = 1
let SentimentValue = document.querySelector(".sentiment_container_value")
let loadingImage = document.querySelector(".loading-image-container")

let session = ""
let loadPath = ""

function time(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function predict(inputFeatures, path, key) {

  if (loadPath != path)
  {  
    window.alert("The model may take a few minutes to load")
    session = await ort.InferenceSession.create(path);
  }
  else {
    await time(250)
  }
  
  const max_length = 60;

  let processedFeatures;
  if (inputFeatures.length > max_length) {
      processedFeatures = inputFeatures.slice(0, max_length);
  } else {
      processedFeatures = inputFeatures.concat(new Array(max_length - inputFeatures.length).fill(0));
  }

  const input = new BigInt64Array(processedFeatures.map(item => BigInt(item)));

  const tensor = new ort.Tensor('int64', input, [1, max_length]);

  const feeds = {};
  feeds[key] = tensor;

  const result = await session.run(feeds);

  const logits = result.output.data; 
  const probabilities = softmax(logits);

  loadPath = path

  return probabilities;
}

function softmax(logits) {
  const maxLogit = Math.max(...logits); 
  const expLogits = logits.map(logit => Math.exp(logit - maxLogit));
  const sumExp = expLogits.reduce((a, b) => a + b, 0);
  return expLogits.map(value => value / sumExp);
}

async function loadNormalizationInfo(file) {
  if (normalizationData) {
    return normalizationData;
  }

  const response = await fetch('./models/' + file);
  normalizationData = await response.json(); 
  return normalizationData;
}

async function normalizeInputs(text) {

  const dictionary = await loadNormalizationInfo(
    document.querySelector(".algorithm-input").selectedOptions[0].getAttribute("data-json")
  );

  const textArray = text.split(" ");

  const textArrayNormalize = [];

  textArray.forEach((item) => {

    if (dictionary[item]) {
      textArrayNormalize.push(dictionary[item]);
    } else {
      // <UNK> token
      textArrayNormalize.push(1);
    }
  });

  return textArrayNormalize;
}

function formatValue(value) {

  if (value.includes('.')) {
    return false;
  }

  const valueStr = String(value).replace(/,/g, ''); 

  if (valueStr.length > 3) {
    return `${valueStr.slice(0, -3)}.${valueStr.slice(-3)}`;
  } else {
    return `0.${valueStr.padStart(3, '0')}`;
  }
}

async function deNormalizeValue(normalizedValue, target) {

    const normalizationJsonInfo = await loadNormalizationInfo();
  
    let value_min_value = normalizationJsonInfo.min_values[target]
    let value_max_value = normalizationJsonInfo.max_values[target]
    
    value = (normalizedValue * (value_max_value - value_min_value)) + value_min_value;

    return Math.round(value)
  }

document.querySelector('.calculate').addEventListener('click', async (e) => {
  e.preventDefault()

  if (allow == 1){
    
  allow = 0  
  SentimentValue.classList.add("hidden")
  loadingImage.classList.remove("hidden")

  const text = document.querySelector('#message').value;

  if (text.length > 60){
    loadingImage.classList.add("hidden")
    allow = 1 
    return window.alert("The text cannot be longer than 60 characters.")
  }
  else if(document.querySelector(".algorithm-input").value == "none"){
    loadingImage.classList.add("hidden")
    allow = 1 
    return window.alert("Please select a model first before making a prediction")
  }
  else if(text.length < 1){
    loadingImage.classList.add("hidden")
    allow = 1 
    return window.alert("The text field cannot be empty. Please provide a valid input.")
  }

  const normalizeValues = await normalizeInputs(text.toLowerCase()); 

  const algorithm = document.querySelector(".algorithm-input").value

  let normalizePrediction = await predict(normalizeValues, algorithm, "input")

  loadingImage.classList.add("hidden")
  SentimentValue.classList.remove("hidden")

  const classValue = normalizePrediction.indexOf(Math.max(...normalizePrediction))

  let finalPrediction = ""

  if (classValue == 0){
    finalPrediction = "Neutral"
  }
  else if (classValue == 1){
    finalPrediction = "Positive"
  }
  else {
    finalPrediction = "Negative"
  }

  document.querySelector("#sentiment_value_value").innerHTML = finalPrediction
  allow = 1
  
}
});



