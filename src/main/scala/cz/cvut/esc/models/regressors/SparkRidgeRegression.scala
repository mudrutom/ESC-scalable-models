package cz.cvut.esc.models.regressors

import cz.cvut.esc.models.InputFormat._
import cz.cvut.esc.models.{CliApp, InputFormat, Params}
import org.apache.spark.mllib.regression.{LabeledPoint, RidgeRegressionModel, RidgeRegressionWithSGD}
import org.apache.spark.rdd.RDD
import scopt.OptionParser

/** Parameters for the Ridge Regression app. */
case class ParamsRidge(
												input: String = null,
												inputFormat: InputFormat = SVM,
												trainSplit: Double = 0.6,
												iterations: Int = 100,
												stepSize: Double = 1.0,
												regParam: Double = 1.0
												) extends Params

/**
 * Ridge Regression using Stochastic Gradient Descent (with L2 regularization).
 */
object SparkRidgeRegression extends Regressor[ParamsRidge] with CliApp[ParamsRidge] with Serializable {

	override def name = "SparkRidgeRegression"

	override def paramsParser(args: Array[String]) = {
		val parser = new OptionParser[ParamsRidge](name) {
			head("Ridge regression using Stochastic Gradient Descent (with L2 regularization)")
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
		(parser, new ParamsRidge())
	}

	override def trainRegressor(trainData: RDD[LabeledPoint], params: ParamsRidge): RidgeRegressionModel = {
		// run training algorithm to build the model
		RidgeRegressionWithSGD.train(trainData, params.iterations, params.stepSize, params.regParam)
	}
}
