package cz.cvut.esc.models.classifiers

import cz.cvut.esc.models.InputFormat._
import cz.cvut.esc.models.{CliApp, InputFormat, Params}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import scopt.OptionParser

/** Parameters for the SVM app. */
case class ParamsBayes(
												input: String = null,
												inputFormat: InputFormat = SVM,
												trainSplit: Double = 0.6,
												lambda: Double = 1.0
												) extends Params

/**
 * Naive Bayes Classifier.
 */
object SparkNaiveBayes extends Classifier[ParamsBayes] with CliApp[ParamsBayes] with Serializable {

	override def name = "SparkNaiveBayes"

	override def paramsParser(args: Array[String]) = {
		val parser = new OptionParser[ParamsBayes](name) {
			head("Naive Bayes Classifier")
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
			opt[Double]('l', "lambda")
				.text("the smoothing parameter lambda (default is 1.0)")
				.action((x, p) => p.copy(lambda = x))
		}
		(parser, new ParamsBayes())
	}

	override def trainClassifier(trainData: RDD[LabeledPoint], params: ParamsBayes): NaiveBayesModel = {
		// run training algorithm to build the model
		NaiveBayes.train(trainData, params.lambda)
	}
}
