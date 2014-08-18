package cz.cvut.esc.models

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

/** Input data format enumeration. */
object InputFormat extends Enumeration {
	type InputFormat = Value
	val SVM, LP, ARFF = Value
}

import cz.cvut.esc.models.InputFormat._

/** Basic CLI input parameters. */
abstract class Params {
	def input: String
	def inputFormat: InputFormat
	def trainSplit: Double
}

/**
 * Parser for input data sets.
 */
trait InputDataParser[P <: Params] {

	/** @return seed for the random number generator */
	def seed: Long = 11L

	/**
	 * Parse and split the input data into train and test sets.
	 * @param sc Spark context
	 * @param params input parameters
	 * @return train and test sets
	 */
	def parseAndSplitData(sc: SparkContext, params: P): (RDD[LabeledPoint], RDD[LabeledPoint]) = {
		// parse the input data
		val data = parseData(sc, params)

		// split data into training and test sets
		val splits = data.randomSplit(Array(params.trainSplit, 1.0 - params.trainSplit), seed = seed)
		(splits(0), splits(1))
	}

	/**
	 * Parse the input data.
	 * @param sc Spark context
	 * @param params input parameters
	 * @return parsed data
	 */
	def parseData(sc: SparkContext, params: P): RDD[LabeledPoint] = {
		params.inputFormat match {
			case SVM => MLUtils.loadLibSVMFile(sc, params.input)
			case LP => MLUtils.loadLabeledData(sc, params.input)
			case ARFF => parseARFF(sc, params.input)
		}
	}

	/**
	 * Parsing of the Attribute-Relation File Format (ARFF).
	 * @param sc Spark context
	 * @param path file or directory path
	 * @return parsed data
	 */
	private def parseARFF(sc: SparkContext, path: String) = {
  	val file = sc.textFile(path)

		// parse header with attributes info
		val attrs = file.filter(_.startsWith("@attribute")).map(_.substring(10).trim).collect()
		val parsers = attrs.map {
			case attr if attr.endsWith("numeric") => (value: String) => value.toDouble
			case attr if attr.matches("^.*[{].*[}]$") =>
				val nominals = attr.substring(attr.indexOf('{') + 1, attr.indexOf('}')).split(",")
				(value: String) => nominals.indexOf(value).toDouble
			case _ => (_: String) => Double.NaN
		}
		val last = sc.broadcast(attrs.length - 1)

		val lines = file.filter(line => !line.isEmpty && !line.startsWith("%") && !line.startsWith("@")).map(_.split(",").map(_.trim))
		val rows = lines.map(_.zip(parsers).map { case (value, parser) => parser(value) })
		rows.map(row => new LabeledPoint(row(last.value), new DenseVector(row.slice(0, last.value))))
	}
}
