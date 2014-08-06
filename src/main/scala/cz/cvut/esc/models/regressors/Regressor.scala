package cz.cvut.esc.models.regressors

import cz.cvut.esc.models.{InputDataParser, Params}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.{LabeledPoint, RegressionModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 * An abstraction of the (real-valued) regressor.
 */
trait Regressor[P <: Params] extends InputDataParser[P] {

	/** @return classifier name */
	def name: String

	/** @return Spark configuration */
	def sparkConf: SparkConf = new SparkConf().setAppName(name)

	/**
	 * Train a real-valued regressor on the training data.
	 * @param trainData train data
	 * @param params input parameters
	 * @return learned regression model
	 */
	def trainRegressor(trainData: RDD[LabeledPoint], params: P): RegressionModel

	/** Runs learning & evaluation process. */
	def run(params: P) {
		val sc = new SparkContext(sparkConf)

		// parse and split the input data
		val (train, test) = parseAndSplitData(sc, params)

		// train the classifier
		val model = trainRegressor(train.cache(), params)

		// evaluation on the test set
		val prediction = model.predict(test.map(_.features)).zip(test.map(_.label))
		evaluateRealValuedRegressor(prediction)
	}

	/**
	 * Evaluates the results of the regression.
	 * @param prediction prediction and label pairs
	 */
	protected def evaluateRealValuedRegressor(prediction: RDD[(Double, Double)]) {
		// compute the mean squared error
		val mse = prediction.map(x => x._1 - x._2).map(x => x * x).mean()

		println(s"Mean Squared Error = $mse")
	}
}
