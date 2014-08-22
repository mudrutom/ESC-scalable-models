package cz.cvut.esc.models.streaming

import cz.cvut.esc.models.CliApp
import cz.cvut.esc.models.classifiers.{ParamsBayes, SparkNaiveBayes}
import org.apache.spark.SparkContext

/**
 * Streaming classifier using Naive Bayes.
 */
object StreamNaiveBayes extends StreamingClassifier[ParamsBayes] with CliApp[ParamsBayes] with Serializable {

	@transient
	private lazy val sparkContext = new SparkContext(sparkConf)

	override def sc = sparkContext

	override def name = "StreamNaiveBayes"

	override def paramsParser(args: Array[String]) = SparkNaiveBayes.paramsParser(args)

}
