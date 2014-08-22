package cz.cvut.esc.models.streaming

import cz.cvut.esc.models.CliApp
import cz.cvut.esc.models.classifiers.{ParamsLR, SparkLogisticRegression}
import org.apache.spark.SparkContext

object StreamLogisticRegression extends StreamingClassifier[ParamsLR] with CliApp[ParamsLR] with Serializable {

	@transient
	private lazy val sparkContext = new SparkContext(sparkConf)

	override def sc = sparkContext

	override def name = "StreamLogisticRegression"

	override def paramsParser(args: Array[String]) = SparkLogisticRegression.paramsParser(args)

}
