import * as tf from "@tensorflow/tfjs";
import * as Papa from "papaparse";

const BASE_URL =
  "https://gist.githubusercontent.com/ManrajGrover/6589d3fd3eb9a0719d2a83128741dfc1/raw/d0a86602a87bfe147c240e87e6a9641786cafc19/";

const parseCsv = async (data) => {
  return new Promise((resolve) => {
    data = data.map((row) => {
      return Object.keys(row)
        .sort()
        .map((key) => parseFloat(row[key]));
    });
    resolve(data);
  });
};

export const loadCsv = async (filename) => {
  return new Promise((resolve) => {
    const url = `${BASE_URL}${filename}.csv`;

    console.log(` * Downloading data from ${url}`);
    Papa.parse(url, {
      download: true,
      header: true,
      complete: (results) => {
        resolve(parseCsv(results["data"]));
      }
    });
  });
};

export const shuffle = async (data, label) => {
  let counter = data.length;
  let temp = 0;
  let index = 0;

  while (counter > 0) {
    index = (Math.random() * counter) | 0;
    counter--;

    temp = data[counter];
    data[counter] = data[index];
    data[index] = temp;

    temp = label[counter];
    label[counter] = label[index];
    label[index] = temp;
  }
};

const mean = (vector) => {
  let sum = 0;
  for (const x of vector) {
    sum += x;
  }

  return sum / vector.length;
};

const stdDev = (vector) => {
  let squareSum = 0;
  const vectorMean = mean(vector);
  for (const x of vector) {
    squareSum += (x - vectorMean) * (x - vectorMean);
  }
  return Math.sqrt(squareSum / (vector.length - 1));
};

const normalizeVector = (vector, vectorMean, vectorStdDev) => {
  return vector.map((el) => (el - vectorMean) / vectorStdDev);
};

export const normalizeData = (
  dataset,
  isTrainData = true,
  vectorMeans = [],
  vectorStdDevs = []
) => {
  const numFeatures = dataset[0].length;
  let vectorMean;
  let vectorStdDev;

  for (let i = 0; i < numFeatures; i++) {
    const vector = dataset.map((row) => row[i]);

    if (isTrainData) {
      vectorMean = mean(vector);
      vectorStdDev = stdDev(vector);

      vectorMeans.push(vectorMean);
      vectorStdDevs.push(vectorStdDev);
    } else {
      vectorMean = vectorMeans[i];
      vectorStdDev = vectorStdDevs[i];
    }

    const vectorNormalized = normalizeVector(vector, vectorMean, vectorStdDev);

    vectorNormalized.forEach((value, index) => {
      dataset[index][i] = value;
    });
  }

  return { dataset, vectorMeans, vectorStdDevs };
};

export const binarize = (y, threshold) => {
  if (threshold == null) {
    threshold = 0.5;
  }
  tf.util.assert(
    threshold >= 0 && threshold <= 1,
    `Expected threshold to be >=0 and <=1, but got ${threshold}`
  );

  return tf.tidy(() => {
    const condition = y.greater(tf.scalar(threshold));
    return tf.where(condition, tf.onesLike(y), tf.zerosLike(y));
  });
};
