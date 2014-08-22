package cz.cvut.esc.models.streaming

import cz.cvut.esc.models.classifiers._
import cz.cvut.esc.models.{InputDataParser, Params}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.{immutable, mutable}

/** State of the stream classification. */
case class State(
									model: ClassificationModel,
									bestModel: Option[ClassificationModel] = None,
									stats: immutable.Seq[(Double, Boolean)] = Nil
									)

/**
 * An abstraction of a (binary) classifier of steaming data.
 */
trait StreamingClassifier[P <: Params] extends InputDataParser[P] {

	/** @return streaming classifier name */
	def name: String

	/** @return Spark configuration */
	def sparkConf: SparkConf = new SparkConf().setAppName(name)

	/** @return Spark context */
	def sc: SparkContext

	/** Runs stream learning & evaluation process. */
	def run(params: P) {
		val ssc = new StreamingContext(sc, Seconds(1L))
		ssc.checkpoint("temp")

		// get the data stream
		val stream = getDataStream(ssc, params)

		// create state-update function
		val updateState = (points: Seq[LabeledPoint], state: Option[State]) => {
			if (points.size < 1) state
			else {
				// do the training
				val data = sc.parallelize(points, 1).cache()
				val model = learnModel(data, params)

				// update the state
				val next = state match {
					case None => State(model = model)
					case Some(s) =>
						val auPR = computeAuPR(Some(s.model), data)
						val bestAuPR = computeAuPR(s.bestModel, data)
						State(
							model = model,
							bestModel = if (auPR > bestAuPR) Some(s.model) else s.bestModel,
						  stats = s.stats :+ (bestAuPR, auPR > bestAuPR))
				}
				Some(next)
			}
		}

		// apply the update function and print progress
		val stateStream = stream.updateStateByKey(updateState)
		stateStream.foreachRDD(_.values.foreach(s => println(s.stats.last)))

		// start processing
		ssc.start()
		ssc.awaitTermination()
	}

	/**
	 * Provides a DStream with input data.
	 * @param ssc Spark Streaming context
	 * @param params input parameters
	 * @return DStream with input data
	 */
	def getDataStream(ssc: StreamingContext, params: P): DStream[(Int, LabeledPoint)] = {
		// parse the input data
		val data = parseData(sc, params)

		// the first value/column is assumed to be a timestamp
		val seqData = data.map(lp => {
			val features = lp.features.toArray
			features.head -> new LabeledPoint(lp.label, new DenseVector(features.tail))
		}).cache()

		// split the data into batches
		val sequence = seqData.keys.distinct().collect()
		val rddQueue = new mutable.Queue[RDD[(Int, LabeledPoint)]]()
		for (i <- sequence) rddQueue.enqueue(seqData.filter(_._1 == i).map{ case (_, lp) => (0, lp) })

		// create the stream form RDDs
		ssc.queueStream(rddQueue)
	}

	/**
	 * Learns a classification model.
	 * @param data training data
	 * @param params input parameters
	 * @return learned classification model
	 */
	def learnModel(data: RDD[LabeledPoint], params: P): ClassificationModel = {
		params match {
			case svm: ParamsSVM => SparkSVM.trainClassifier(data, svm)
			case bayes: ParamsBayes => SparkNaiveBayes.trainClassifier(data, bayes)
			case lr: ParamsLR => SparkLogisticRegression.trainClassifier(data, lr)
		}
	}

	/**
	 * Computes the auPR classification metric.
	 * @param model classification model
	 * @param data testing data
	 * @return area under precision-recall curve
	 */
	def computeAuPR(model: Option[ClassificationModel], data: RDD[LabeledPoint]): Double = {
		model match {
			case None => 0.0
			case Some(m) =>
				val prediction = m.predict(data.map(_.features)).zip(data.map(_.label))
				val metrics = new BinaryClassificationMetrics(prediction)
				metrics.areaUnderPR()
		}
	}
}
