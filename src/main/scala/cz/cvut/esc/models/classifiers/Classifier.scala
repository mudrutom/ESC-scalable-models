package cz.cvut.esc.models.classifiers

import cz.cvut.esc.models.{InputDataParser, Params}
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 * An abstraction of the (binary) classifier.
 */
trait Classifier[P <: Params] extends InputDataParser[P] {

	/** @return classifier name */
	def name: String

	/** @return Spark configuration */
	def sparkConf: SparkConf = new SparkConf().setAppName(name)

	/**
	 * Train a binary classifier on the training data.
	 * @param trainData train data
	 * @param params input parameters
	 * @return learned classification model
	 */
	def trainClassifier(trainData: RDD[LabeledPoint], params: P): ClassificationModel

	/** Runs learning & evaluation process. */
	def run(params: P) {
		val sc = new SparkContext(sparkConf)

		// parse and split the input data
		val (train, test) = parseAndSplitData(sc, params)

		// train the classifier
		val model = trainClassifier(train.cache(), params)

		// evaluation on the test set
		val prediction = model.predict(test.map(_.features)).zip(test.map(_.label))
		evaluateBinaryClassifier(prediction)
	}

	/**
	 * Evaluates the results of the classification.
	 * @param prediction prediction and label pairs
	 */
	def evaluateBinaryClassifier(prediction: RDD[(Double, Double)]) {
		// get evaluation metrics
		val metrics = new BinaryClassificationMetrics(prediction)
		val auPR = metrics.areaUnderPR()
		val auROC = metrics.areaUnderROC()

		println(s"Area under the Precision-Recall curve = $auPR")
		println(s"Area under the Receiver-Operating-Characteristic curve = $auROC")
	}
}
