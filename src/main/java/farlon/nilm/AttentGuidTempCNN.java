package farlon.nilm;

import java.io.File;
import org.bytedeco.opencv.opencv_dnn.FlattenLayer;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation.Metric;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class AttentGuidTempCNN {

  // Load your dataset (normalized and prepared as required)
  public static DataSet loadData() {
    // Implement data loading logic similar to helper.load_data() in TensorFlow example
    // Return a DataSet object
      return null;
  }

  public static ComputationGraphConfiguration createTCNModel(int inputSequenceLength) {
    ComputationGraphConfiguration.GraphBuilder graph =
        new ComputationGraphConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.RELU)
            .updater(new Adam())
            .graphBuilder()
            .addInputs("input")
            .setInputTypes(InputType.recurrent(1));

    // Create Temporal CNN blocks with attention mechanism
    for (int i = 0; i < 3; i++) {
      String prevLayer = (i == 0) ? "input" : "attention" + (i - 1);

      graph
          // Temporal CNN block
          .addLayer(
              "conv1_" + i,
              new Convolution1DLayer.Builder(4).dilation(2 ^ i).activation(Activation.RELU).build(),
              prevLayer)
          .addLayer(
              "conv2_" + i,
              new Convolution1DLayer.Builder(4).dilation(2 ^ i).activation(Activation.RELU).build(),
              "conv1_" + i)

          // Attention mechanism
          .addLayer(
              "attention_" + i,
              new DenseLayer.Builder().activation(Activation.TANH).nOut(1).build(),
              "conv2_" + i)
          .addLayer("flatten_" + i, new FlattenLayer(), "attention_" + i)
          .addLayer(
              "softmax_" + i,
              new OutputLayer.Builder()
                  .activation(Activation.SOFTMAX)
                  .lossFunction(LossFunctions.LossFunction.MSE)
                  .nOut(inputSequenceLength)
                  .build(),
              "flatten_" + i);
    }

    graph
        // Fully connected and output layers
        .addLayer(
            "dense",
            new DenseLayer.Builder().activation(Activation.RELU).nOut(128).build(),
            "attention2")
        .addLayer(
            "output",
            new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nOut(1)
                .build(),
            "dense")
        .setOutputs("output");

    return graph.build();
  }

  public static void main(String[] args) throws Exception {
    // Define model input shape
    int inputSequenceLength = 15; // Window size
    int batchSize = 8;

    // Load training and testing data
    DataSet trainData = loadData(); // Use your helper.load_data() equivalent here
    DataSet testData = loadData();

    // Normalize data
    NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
    normalizer.fit(trainData);
    normalizer.transform(trainData);
    normalizer.transform(testData);

    // Prepare the model
    ComputationGraphConfiguration config = createTCNModel(inputSequenceLength);
    ComputationGraph model = new ComputationGraph(config);
    model.init();

    // Early stopping configuration
    EarlyStoppingModelSaver<ComputationGraph> saver = new InMemoryModelSaver<>();
    EarlyStoppingListener<ComputationGraph> listener =
        new EarlyStoppingListener<ComputationGraph>() {
          @Override
          public void onStart(EarlyStoppingResult<ComputationGraph> result) {
            System.out.println("Early stopping training started");
          }

          @Override
          public void onEpoch(EarlyStoppingResult<ComputationGraph> result, int epochNum) {
            System.out.println("Epoch " + epochNum + " completed");
          }

          @Override
          public void onCompletion(EarlyStoppingResult<ComputationGraph> result) {
            System.out.println("Early stopping training complete");
            System.out.println("Best epoch: " + result.getBestModelEpoch());
            System.out.println("Best score: " + result.getBestModelScore());
          }
        };

    model.setListeners(new ScoreIterationListener(10)); // Print score every 10 iterations

    // Train the model
    for (int epoch = 0; epoch < 10; epoch++) {
      model.fit(trainData);
      System.out.println("Epoch " + epoch + " complete");

      // Evaluate on test data after each epoch
      RegressionEvaluation eval =
          model.evaluateRegression(new Seq2SeqDataSetIterator(testData, batchSize));
      System.out.println(eval.stats());
    }

    // Save the model
    File modelFile = new File("latest_att_temp_cnn.zip");
    model.save(modelFile);

    // Evaluate on test set
    RegressionEvaluation eval = new RegressionEvaluation();
    INDArray output = model.output(testData.getFeatures());
    eval.eval(testData.getLabels(), output);
    System.out.println("Test MAE: " + eval.meanAbsoluteError(Metric.MAE.ordinal()));
  }
}
