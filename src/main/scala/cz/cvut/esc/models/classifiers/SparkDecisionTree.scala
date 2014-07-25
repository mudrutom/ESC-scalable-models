package cz.cvut.esc.models.classifiers

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.{SparkContext, SparkConf}

object SparkDecisionTree extends Classifier {

	case class Params(input: String)

	def main(args: Array[String]) {
		if (args.length < 1) sys.exit(1) else run(new Params(args(0)))
	}

	def run(params: Params) {
		val conf = new SparkConf().setAppName("SparkDecisionTree")
		val sc = new SparkContext(conf)

		// parse the input data
		val data = parseLabeledData(sc, params.input)

		// split data into training (60%) and test (40%)
		val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
		val train = splits(0).cache()
		val test = splits(1)

		// run training algorithm to build the model
		val model = DecisionTree.train(train, Algo.Classification, Gini, 3)

		// evaluation on the test set
		val prediction = model.predict(test.map(_.features))
		val predictionAndLabel = prediction.zip(test.map(_.label))

		// get evaluation metrics
		val metrics = new BinaryClassificationMetrics(predictionAndLabel)
		val auROC = metrics.areaUnderROC()

		println(s"Area under ROC = $auROC")
	}

}
