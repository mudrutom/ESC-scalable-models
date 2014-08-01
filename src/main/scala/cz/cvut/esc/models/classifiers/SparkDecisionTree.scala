package cz.cvut.esc.models.classifiers

import cz.cvut.esc.models.classifiers.InputFormat._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.{Impurity, Entropy => ImpurityEntropy, Gini => ImpurityGini, Variance => ImpurityVariance}
import org.apache.spark.rdd.RDD
import scopt.OptionParser

/** Impurity type enumeration. */
object ImpurityType extends Enumeration {
	type ImpurityType = Value
	val Gini, Entropy, Variance = Value

	/** @return Impurity class of given type */
	def impurity(impurity: ImpurityType): Impurity = impurity match {
		case Gini => ImpurityGini
		case Entropy => ImpurityEntropy
		case Variance => ImpurityVariance
	}
}

import cz.cvut.esc.models.classifiers.ImpurityType._

/** Parameters for the DecisionTree app. */
case class ParamsDT(
										 input: String = null,
										 inputFormat: InputFormat = SVM,
										 trainSplit: Double = 0.6,
										 maxDepth: Int = 5,
										 impurity: ImpurityType = Gini
										 ) extends Params

/**
 * Decision tree classifier.
 */
object SparkDecisionTree extends Classifier[ParamsDT] with CliApp[ParamsDT] with Serializable {

	override def name = "SparkDecisionTree"

	override def paramsParser(args: Array[String]): (OptionParser[ParamsDT], ParamsDT) = {
		val parser = new OptionParser[ParamsDT](name) {
			head("Decision tree classifier")
			arg[String]("<input>")
				.required()
				.text("path to the input dataset")
				.action((x, p) => p.copy(input = x))
			opt[String]('f', "format")
				.text("input file format: " + InputFormat.values.mkString(","))
				.action((x, p) => p.copy(inputFormat = InputFormat.withName(x)))
			opt[Double]("trainSplit")
				.text("fraction of the dataset to use for training")
				.action((x, p) => p.copy(trainSplit = x))
			opt[Int]('d', "maxDepth")
				.text("maximum depth of the tree (default is 5)")
				.action((x, p) => p.copy(maxDepth = x))
			opt[String]("impurity")
				.text("impurity criterion used for information gain calculation: " + ImpurityType.values.mkString(","))
				.action((x, p) => p.copy(impurity = ImpurityType.withName(x)))
		}
		(parser, new ParamsDT())
	}

	override def trainClassifier(trainData: RDD[LabeledPoint], params: ParamsDT): ClassificationModel = null

	override def run(params: ParamsDT) = {
		val sc = new SparkContext(sparkConf)

		// parse and split the input data
		val (train, test) = parseAndSplitData(sc, params)

		// run training algorithm to build the model
		val model = DecisionTree.train(train.cache(), Algo.Classification, ImpurityType.impurity(params.impurity), params.maxDepth)

		// evaluation on the test set
		val prediction = model.predict(test.map(_.features)).zip(test.map(_.label))
		evaluateBinaryClassifier(prediction)
	}
}
