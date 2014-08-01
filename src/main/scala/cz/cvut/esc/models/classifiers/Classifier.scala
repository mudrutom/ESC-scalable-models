package cz.cvut.esc.models.classifiers

import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/** Input data format enumeration. */
object InputFormat extends Enumeration {
	type InputFormat = Value
	val SVM, LP = Value
}

import cz.cvut.esc.models.classifiers.InputFormat._

/** Basic CLI input parameters. */
abstract class Params {
	def input: String
	def inputFormat: InputFormat
	def trainSplit: Double
}

/**
 * An abstraction of the (binary) classifier.
 *
 *
 */
trait Classifier[P <: Params] {

	/** @return classifier name */
	def name: String

	/** @return seed for the random number generator */
	def seed: Long = 11L

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
	 * Parse and split the input data into train and test sets.
	 * @param sc Spark context
	 * @param params input parameters
	 * @return train and test sets
	 */
	protected def parseAndSplitData(sc: SparkContext, params: P) = {
		// parse the input data
		val data = params.inputFormat match {
			case SVM => MLUtils.loadLibSVMFile(sc, params.input)
			case LP => MLUtils.loadLabeledData(sc, params.input)
		}

		// split data into training and test sets
		val splits = data.randomSplit(Array(params.trainSplit, 1.0 - params.trainSplit), seed = seed)
		(splits(0), splits(1))
	}

	/**
	 * Evaluates the results of the classification.
	 * @param prediction prediction and label pairs
	 */
	protected def evaluateBinaryClassifier(prediction: RDD[(Double, Double)]) {
		// get evaluation metrics
		val metrics = new BinaryClassificationMetrics(prediction)
		val auROC = metrics.areaUnderROC()

		println(s"Area under ROC = $auROC")
	}
}

/**
 * CLI (Command Line Interface) application.
 */
trait CliApp[P] {

	/**
	 * Returns option parameter parser with default values
	 * @param args input arguments
	 * @return default params and the parameter parser
	 */
	def paramsParser(args: Array[String]): (OptionParser[P], P)

	def run(params: P)

	/** The main method. */
	def main(args: Array[String]) {
		val (parser, default) = paramsParser(args)
		parser.parse(args, default).map { params =>
			run(params)
		} getOrElse {
			System.exit(1)
		}
	}
}
