package cz.cvut.esc.models.streaming

import cz.cvut.esc.models.CliApp
import cz.cvut.esc.models.classifiers.{ParamsSVM, SparkSVM}

object StreamSVM extends StreamingClassifier[ParamsSVM] with CliApp[ParamsSVM] with Serializable {

	override def name = "StreamSVM"

	override def paramsParser(args: Array[String]) = SparkSVM.paramsParser(args)

}
