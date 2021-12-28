import * as tf from "@tensorflow/tfjs";
import * as utils from "./utils";

const TRAIN_DATA = "train-data";
const TRAIN_TARGET = "train-target";
const TEST_DATA = "test-data";
const TEST_TARGET = "test-target";

export class WebsitePhishingDataset {
  constructor() {
    this.dataset = null;
    this.trainSize = 0;
    this.testSize = 0;
    this.trainBatchIndex = 0;
    this.testBatchIndex = 0;

    this.NUM_FEATURES = 30;
    this.NUM_CLASSES = 2;
  }

  get numFeatures() {
    return this.NUM_FEATURES;
  }

  async loadData() {
    this.dataset = await Promise.all([
      utils.loadCsv(TRAIN_DATA),
      utils.loadCsv(TRAIN_TARGET),
      utils.loadCsv(TEST_DATA),
      utils.loadCsv(TEST_TARGET)
    ]);

    let {
      dataset: trainDataset,
      vectorMeans,
      vectorStdDevs
    } = utils.normalizeData(this.dataset[0]);

    this.dataset[0] = trainDataset;

    let { dataset: testDataset } = utils.normalizeData(
      this.dataset[2],
      false,
      vectorMeans,
      vectorStdDevs
    );

    this.dataset[2] = testDataset;

    this.trainSize = this.dataset[0].length;
    this.testSize = this.dataset[2].length;

    utils.shuffle(this.dataset[0], this.dataset[1]);
    utils.shuffle(this.dataset[2], this.dataset[3]);
  }

  getTrainData() {
    const dataShape = [this.trainSize, this.NUM_FEATURES];

    const trainData = Float32Array.from([].concat.apply([], this.dataset[0]));
    const trainTarget = Float32Array.from([].concat.apply([], this.dataset[1]));

    return {
      data: tf.tensor2d(trainData, dataShape),
      target: tf.tensor1d(trainTarget)
    };
  }

  getTestData() {
    const dataShape = [this.testSize, this.NUM_FEATURES];

    const testData = Float32Array.from([].concat.apply([], this.dataset[2]));
    const testTarget = Float32Array.from([].concat.apply([], this.dataset[3]));

    return {
      data: tf.tensor2d(testData, dataShape),
      target: tf.tensor1d(testTarget)
    };
  }
}
