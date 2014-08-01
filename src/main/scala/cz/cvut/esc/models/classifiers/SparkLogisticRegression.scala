package cz.cvut.esc.models.classifiers

import cz.cvut.esc.models.InputFormat._
import cz.cvut.esc.models.{CliApp, InputFormat, Params}
import org.apache.spark.mllib.classification.{ClassificationModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import scopt.OptionParser

/** Parameters for the LogisticRegression app. */
case class ParamsLR(
										 input: String = null,
										 inputFormat: InputFormat = SVM,
										 trainSplit: Double = 0.6,
										 iterations: Int = 100,
										 stepSize: Double = 1.0
										 ) extends Params

/**
 * A classifier trained using Logistic Regression with Stochastic Gradient Descent.
 */
object SparkLogisticRegression extends Classifier[ParamsLR] with CliApp[ParamsLR] with Serializable {

	override def name = "SparkLogisticRegression"

	override def paramsParser(args: Array[String]): (OptionParser[ParamsLR], ParamsLR) = {
		val parser = new OptionParser[ParamsLR](name) {
			head("A classifier trained using Logistic Regression with Stochastic Gradient Descent")
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
			opt[Int]('n', "iterations")
				.text("number of iterations of gradient descent to run (default is 100)")
				.action((x, p) => p.copy(iterations = x))
			opt[Double]("stepSize")
				.text("step size to be used for each iteration of gradient descent (default is 1.0)")
				.action((x, p) => p.copy(stepSize = x))
		}
		(parser, new ParamsLR())
	}

	override def trainClassifier(trainData: RDD[LabeledPoint], params: ParamsLR): ClassificationModel = {
		// run training algorithm to build the model
		LogisticRegressionWithSGD.train(trainData, params.iterations, params.stepSize)
	}
}
