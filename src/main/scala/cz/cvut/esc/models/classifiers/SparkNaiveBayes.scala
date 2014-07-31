package cz.cvut.esc.models.classifiers

import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object SparkNaiveBayes extends Classifier with Serializable {

	name = "SparkNaiveBayes"

	override def trainClassifier(trainData: RDD[LabeledPoint], params: Params): NaiveBayesModel = {
		// run training algorithm to build the model
		NaiveBayes.train(trainData)
	}

}
