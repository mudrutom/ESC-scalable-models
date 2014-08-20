package cz.cvut.esc.models.streaming

import cz.cvut.esc.models.InputFormat._
import cz.cvut.esc.models.classifiers.{ParamsSVM, SparkSVM}
import cz.cvut.esc.models.{CliApp, InputDataParser, InputFormat, Params}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

import scala.collection.mutable

case class ParamsT(
										input: String = null,
										inputFormat: InputFormat = SVM,
										trainSplit: Double = 0.6
										) extends Params

case class TestState(
											best: Option[ClassificationModel] = None,
											bestAuPR: Double = 0.0,
											current: ClassificationModel
											)

object Test extends CliApp[ParamsT] with InputDataParser[ParamsT] {

	override def paramsParser(args: Array[String]) = {
		val parser = new OptionParser[ParamsT]("test") {
			arg[String]("<input>")
			.required()
			.text("path to the input dataset")
			.action((x, p) => p.copy(input = x))
			opt[String]('f', "format")
			.text("input file format: " + InputFormat.values.mkString(","))
			.action((x, p) => p.copy(inputFormat = InputFormat.withName(x)))
		}
		(parser, new ParamsT())
	}

	val sc = new SparkContext(new SparkConf())

	override def run(params: ParamsT) {
		Logger.getRootLogger.setLevel(Level.WARN)

		val data = parseData(sc, params)

		// the first value is a timestamp
		val seqData = data.map(lp => {
			val features = lp.features.toArray
			features.head -> new LabeledPoint(lp.label, new DenseVector(features.tail))
		}).cache()

		// split data into batches
		val sequence = seqData.keys.distinct().collect()
		val q = new mutable.Queue[RDD[(Double, LabeledPoint)]]()
		for (s <- sequence) q.enqueue(seqData.filter(_._1 == s))

		// init stream
		val ssc = new StreamingContext(sc, Seconds(1L))
		ssc.checkpoint("temp")
		val stream = ssc.queueStream(q)

		def updateState(points: Seq[LabeledPoint], state: Option[TestState]): Option[TestState] = {
			if (points.size < 1) None
			else {
				// do the SVM training
				val data = sc.makeRDD(points).cache()
				val model = SparkSVM.trainClassifier(data, ParamsSVM())

				// update the state
				Some(state match {
					case None => TestState(current = model)
					case Some(s) =>
						val auPR = computeAuPR(Some(s.current), data)
						val bestAuPR = computeAuPR(s.best, data)
						println(bestAuPR + " curr=" + auPR)
						if (auPR > bestAuPR) TestState(current = model, best = Some(s.current), bestAuPR = auPR)
						else s.copy(current = model, bestAuPR = bestAuPR)
				})
			}
		}

		stream.updateStateByKey(updateState).foreachRDD(rdd => println(rdd.count()))

		// start processing
		ssc.start()
		ssc.awaitTermination()
	}

	def computeAuPR(model: Option[ClassificationModel], data: RDD[LabeledPoint]) = {
		model match {
			case None => 0.0
			case Some(m) =>
				val prediction = m.predict(data.map(_.features)).zip(data.map(_.label))
				val metrics = new BinaryClassificationMetrics(prediction)
				metrics.areaUnderPR()
		}
	}

}
