package cz.cvut.esc.models.classifiers

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.{SparkConf, SparkContext}

object SparkSVM extends Classifier with Serializable {

	case class Params(input: String)

	def main(args: Array[String]) {
		if (args.length < 1) sys.exit(1) else run(new Params(args(0)))
	}

	def run(params: Params) {
		val conf = new SparkConf().setAppName("SparkSVM")
		val sc = new SparkContext(conf)

		// parse the input data
		val data = parseLabeledData(sc, params.input)

		// split data into training (60%) and test (40%)
		val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
		val train = splits(0).cache()
		val test = splits(1)

		// run training algorithm to build the model
		val model = SVMWithSGD.train(train, 10)
		// clear the default threshold.
		model.clearThreshold()

		// evaluate the classifier
		evaluateBinaryClassifier(model, test)
	}

}
