package cz.cvut.esc.models.classifiers

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.rdd.RDD

object SparkDecisionTree extends Classifier with Serializable {

	name = "SparkDecisionTree"

	override def trainClassifier(trainData: RDD[LabeledPoint], params: SparkDecisionTree.Params): ClassificationModel = ???

	override protected def run(params: Params) = {
		val sc = new SparkContext(sparkConf)

		// parse and split the input data
		val (train, test) = parseAndSplitData(sc, params)

		// run training algorithm to build the model
		val model = DecisionTree.train(train.cache(), Algo.Classification, Gini, 3)

		// evaluation on the test set
		val prediction = model.predict(test.map(_.features)).zip(test.map(_.label))
		evaluateBinaryClassifier(prediction)
	}

}
