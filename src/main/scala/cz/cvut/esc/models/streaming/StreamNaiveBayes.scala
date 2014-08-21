package cz.cvut.esc.models.streaming

import cz.cvut.esc.models.CliApp
import cz.cvut.esc.models.classifiers.{ParamsBayes, SparkNaiveBayes}

object StreamNaiveBayes extends StreamingClassifier[ParamsBayes] with CliApp[ParamsBayes] with Serializable {

	 override def name = "StreamNaiveBayes"

	 override def paramsParser(args: Array[String]) = SparkNaiveBayes.paramsParser(args)

}
