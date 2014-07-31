package cz.cvut.esc.models.classifiers

import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object SparkSVM extends Classifier with Serializable {

	name = "SparkSVM"

	override def trainClassifier(trainData: RDD[LabeledPoint], params: Params): SVMModel = {
			// run training algorithm to build the model
			val model = SVMWithSGD.train(trainData, 10)
			// clear the default threshold.
			model.clearThreshold()
			model
	}

}
