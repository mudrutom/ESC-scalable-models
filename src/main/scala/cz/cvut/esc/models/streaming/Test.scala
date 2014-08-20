package cz.cvut.esc.models.streaming

import cz.cvut.esc.models.InputFormat._
import cz.cvut.esc.models.classifiers.{ParamsSVM, SparkSVM}
import cz.cvut.esc.models.{CliApp, InputDataParser, InputFormat, Params}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.{Milliseconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

import scala.collection.mutable

case class ParamsT(
										input: String = null,
										inputFormat: InputFormat = SVM,
										trainSplit: Double = 0.6
										) extends Params

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

	override def run(params: ParamsT) {
		Logger.getRootLogger.setLevel(Level.WARN)

		val sc = new SparkContext(new SparkConf())
		val data = parseData(sc, params)

		// the first value is a timestamp
		val seqData = data.map(lp => {
			val features = lp.features.toArray
			features.head -> new LabeledPoint(lp.label, new DenseVector(features.tail))
		}).cache()

		// split data into batches
		val sequence = seqData.keys.distinct().collect()
		val q = new mutable.Queue[RDD[LabeledPoint]]()
		for (s <- sequence) q.enqueue(seqData.filter(_._1 == s).values)

		// init stream
		val ssc = new StreamingContext(sc, Milliseconds(100L))
		val stream = ssc.queueStream(q).window(Milliseconds(500L), Milliseconds(500L))

		// do the SVM training
		stream.foreachRDD(rdd => {
			val t = System.currentTimeMillis()
			val model = SparkSVM.trainClassifier(rdd.cache(), ParamsSVM())
			println(model.toString + " " + (System.currentTimeMillis() - t))
		})

		// start processing
		ssc.start()
		ssc.awaitTermination()
	}

}
