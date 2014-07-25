package cz.cvut.esc.models.classifiers

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

trait Classifier {

	val POS_CLASS = "0"

	def parseLabeledData(sc: SparkContext, input: String) = sc.textFile(input).map { line =>
		val parts = line.split(',')
		LabeledPoint(if (parts(0) == POS_CLASS) 1 else 0, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
	}

	def evaluateBinaryClassifier(model: ClassificationModel, test: RDD[LabeledPoint]) {
		// evaluation on the test set
		val prediction = model.predict(test.map(_.features))
		val predictionAndLabel = prediction.zip(test.map(_.label))

		// get evaluation metrics
		val metrics = new BinaryClassificationMetrics(predictionAndLabel)
		val auROC = metrics.areaUnderROC()

		println(s"Area under ROC = $auROC")
	}

}
