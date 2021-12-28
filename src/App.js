import "./styles/index.css";
import { useState, useRef, useEffect } from "react";

//imports
import * as tf from "@tensorflow/tfjs";
import { WebsitePhishingDataset } from "./js/data";
import * as utils from "./js/utils";
import * as tfvis from "@tensorflow/tfjs-vis";

import Header from "./components/Header";
import Status from "./components/Status";
import Roc from "./components/Roc";

const rocValues = [];
const rocSeries = [];

export default function App() {
  //hooks
  const [updateStatus, setUpdateStatus] = useState("Loading Data...");
  const plotLossContainer = useRef(null);
  const plotAccuraciesContainer = useRef(null);
  const ROCContainer = useRef(null);
  const [start, setStart] = useState(true);

  useEffect(() => {
    const data = new WebsitePhishingDataset();
    start &&
      data.loadData().then(async () => {
        setUpdateStatus("Getting training and testing data...");
        const trainData = data.getTrainData();
        const testData = data.getTestData();
        setUpdateStatus("Building Model.....");
        //create the model
        const model = tf.sequential();
        model.add(
          tf.layers.dense({
            inputShape: [data.numFeatures],
            units: 100,
            activation: "sigmoid"
          })
        );

        model.add(
          tf.layers.dense({
            units: 100,
            activation: "sigmoid"
          })
        );

        model.add(
          tf.layers.dense({
            units: 1,
            activation: "sigmoid"
          })
        );

        model.compile({
          optimizer: "adam",
          loss: "binaryCrossentropy",
          metrics: ["accuracy"]
        });

        const trainLogs = [];
        let auc;

        setUpdateStatus("Training Starting...");
        await model.fit(trainData.data, trainData.target, {
          batchSize,
          epochs,
          validationSplit: 0.2,
          callbacks: {
            onEpochBegin: async (epoch) => {
              if (
                (epoch + 1) % 100 === 0 ||
                epoch === 0 ||
                epoch === 2 ||
                epoch === 4
              ) {
                const probs = model.predict(testData.data);
                auc = drawROC(testData.target, probs, epoch);
              }
            },
            onEpochEnd: async (epoch, logs) => {
              setUpdateStatus(`Epoch ${epoch + 1} of ${epochs} completed`);
              trainLogs.push(logs);
              PlotLosses(trainLogs);
              plotAccuracies(trainLogs);
            }
          }
        });
        setUpdateStatus("Running on test data...");
        tf.tidy(() => {
          const result = model.evaluate(testData.data, testData.target, {
            batchSize: batchSize
          });

          const lastTrainLog = trainLogs[trainLogs.length - 1];
          const testLoss = result[0].dataSync()[0];
          const testAcc = result[1].dataSync()[0];

          const probs = model.predict(testData.data);
          const predictions = utils.binarize(probs).as1D();

          const precision = tf.metrics
            .precision(testData.target, predictions)
            .dataSync()[0];

          const recall = tf.metrics
            .recall(testData.target, predictions)
            .dataSync()[0];

          const fpr = falsePositiveRate(
            testData.target,
            predictions
          ).dataSync()[0];
          setUpdateStatus(
            `Final train-set loss: ${lastTrainLog.loss.toFixed(
              4
            )} accuracy: ${lastTrainLog.acc.toFixed(4)}\n` +
              `Final validation-set loss: ${lastTrainLog.val_loss.toFixed(
                4
              )} accuracy: ${lastTrainLog.val_acc.toFixed(4)}\n` +
              `Test-set loss: ${testLoss.toFixed(
                4
              )} accuracy: ${testAcc.toFixed(4)}\n` +
              `Precision: ${precision.toFixed(4)}\n` +
              `Recall: ${recall.toFixed(4)}\n` +
              `False positive rate (FPR): ${fpr.toFixed(4)}\n` +
              `Area under the curve (AUC): ${auc.toFixed(4)}`
          );
        });
      });
  }, []);
  //hooks end
  const falsePositives = (yTrue, yPred) => {
    return tf.tidy(() => {
      const one = tf.scalar(1);
      const zero = tf.scalar(0);

      return tf
        .logicalAnd(yTrue.equal(zero), yPred.equal(one))
        .sum()
        .cast("float32");
    });
  };

  const trueNegatives = (yTrue, yPred) => {
    return tf.tidy(() => {
      const zero = tf.scalar(0);
      return tf
        .logicalAnd(yTrue.equal(zero), yPred.equal(zero))
        .sum()
        .cast("float32");
    });
  };

  const falsePositiveRate = (yTrue, yPred) => {
    return tf.tidy(() => {
      const fp = falsePositives(yTrue, yPred);
      const tn = trueNegatives(yTrue, yPred);
      return fp.div(fp.add(tn));
    });
  };

  const drawROC = (targets, probs, epoch) => {
    return tf.tidy(() => {
      const thresholds = [
        0.0,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.92,
        0.94,
        0.96,
        0.98,
        1.0
      ];
      const tprs = []; // True positive rates.
      const fprs = []; // False positive rates.
      let area = 0;
      for (let i = 0; i < thresholds.length; ++i) {
        const threshold = thresholds[i];

        const threshPredictions = utils.binarize(probs, threshold).as1D();
        const fpr = falsePositiveRate(targets, threshPredictions).dataSync()[0];
        const tpr = tf.metrics.recall(targets, threshPredictions).dataSync()[0];
        fprs.push(fpr);
        tprs.push(tpr);

        // Accumulate to area for AUC calculation.
        if (i > 0) {
          area += ((tprs[i] + tprs[i - 1]) * (fprs[i - 1] - fprs[i])) / 2;
        }
      }
      plotROC(fprs, tprs, epoch);
      return area;
    });
  };

  const epochs = 400;
  const batchSize = 350;

  const PlotLosses = async (trainLogs) => {
    return (
      plotLossContainer.current &&
      tfvis.show.history(
        plotLossContainer.current,
        trainLogs,
        ["loss", "val_loss"],
        {
          width: 450,
          height: 320,
          xLabel: "Epoch",
          yLabel: "Loss"
        }
      )
    );
  };

  const plotAccuracies = async (trainLogs) => {
    console.log(plotAccuracies.current);
    plotAccuraciesContainer.current &&
      tfvis.show.history(
        plotAccuraciesContainer.current,
        trainLogs,
        ["acc", "val_acc"],
        {
          width: 450,
          height: 320,
          xLabel: "Epoch",
          yLabel: "Accuracy"
        }
      );
  };

  const plotROC = async (fprs, tprs, epoch) => {
    epoch++; // Convert zero-based to one-based.

    // Store the series name in the list of series
    const seriesName =
      "epoch " +
      (epoch < 10 ? `00${epoch}` : epoch < 100 ? `0${epoch}` : `${epoch}`);
    rocSeries.push(seriesName);

    const newSeries = [];
    for (let i = 0; i < fprs.length; i++) {
      newSeries.push({
        x: fprs[i],
        y: tprs[i]
      });
    }
    rocValues.push(newSeries);

    return tfvis.render.linechart(
      document.getElementById("rocCurve"),
      { values: rocValues, series: rocSeries },
      {
        width: 450,
        height: 320
      }
    );
  };

  return (
    <div className="tfjs-example-container centered-container">
      <Header />
      <Status updateStatus={updateStatus} />
      <Roc
        plotLossContainer={plotLossContainer}
        plotAccuraciesContainer={plotAccuraciesContainer}
        ROCContainer={ROCContainer}
      />
    </div>
  );
}
