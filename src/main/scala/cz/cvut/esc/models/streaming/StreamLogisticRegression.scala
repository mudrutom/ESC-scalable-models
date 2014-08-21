package cz.cvut.esc.models.streaming

import cz.cvut.esc.models.CliApp
import cz.cvut.esc.models.classifiers.{ParamsLR, SparkLogisticRegression}

object StreamLogisticRegression extends StreamingClassifier[ParamsLR] with CliApp[ParamsLR] with Serializable {

	 override def name = "StreamLogisticRegression"

	 override def paramsParser(args: Array[String]) = SparkLogisticRegression.paramsParser(args)

 }
