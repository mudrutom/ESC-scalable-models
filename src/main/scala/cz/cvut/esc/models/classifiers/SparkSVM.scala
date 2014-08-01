package cz.cvut.esc.models.classifiers

import cz.cvut.esc.models.InputFormat._
import cz.cvut.esc.models.{CliApp, InputFormat, Params}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import scopt.OptionParser

/** Parameters for the SVM app. */
case class ParamsSVM(
											input: String = null,
											inputFormat: InputFormat = SVM,
											trainSplit: Double = 0.6,
											iterations: Int = 100,
											stepSize: Double = 1.0,
											regParam: Double = 1.0
											) extends Params

/**
 * Support Vector Machine (SVM) classifier using Stochastic Gradient Descent
 */
object SparkSVM extends Classifier[ParamsSVM] with CliApp[ParamsSVM] with Serializable {

	override def name = "SparkSVM"

	override def paramsParser(args: Array[String]): (OptionParser[ParamsSVM], ParamsSVM) = {
		val parser = new OptionParser[ParamsSVM](name) {
			head("Support Vector Machine (SVM) classifier using Stochastic Gradient Descent")
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
			opt[Double]("regParam")
				.text("regularization parameter (default is 1.0)")
				.action((x, p) => p.copy(regParam = x))
		}
		(parser, new ParamsSVM())
	}

	override def trainClassifier(trainData: RDD[LabeledPoint], params: ParamsSVM): SVMModel = {
			// run training algorithm to build the model
			val model = SVMWithSGD.train(trainData, params.iterations, params.stepSize, params.regParam)
			// clear the default threshold.
			model.clearThreshold()
			model
	}
}
