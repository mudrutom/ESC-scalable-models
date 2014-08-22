package cz.cvut.esc.models.streaming

import cz.cvut.esc.models.CliApp
import cz.cvut.esc.models.classifiers.{ParamsSVM, SparkSVM}
import org.apache.spark.SparkContext

/**
 * Streaming classifier using Support Vector Machine (SVM) with Stochastic Gradient Descent.
 */
object StreamSVM extends StreamingClassifier[ParamsSVM] with CliApp[ParamsSVM] with Serializable {

	@transient
	private lazy val sparkContext = new SparkContext(sparkConf)

	override def sc = sparkContext

	override def name = "StreamSVM"

	override def paramsParser(args: Array[String]) = SparkSVM.paramsParser(args)

}
