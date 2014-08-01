package cz.cvut.esc.models

import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils

/** Input data format enumeration. */
object InputFormat extends Enumeration {
	type InputFormat = Value
	val SVM, LP = Value
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
	def parseAndSplitData(sc: SparkContext, params: P) = {
		// parse the input data
		val data = params.inputFormat match {
			case SVM => MLUtils.loadLibSVMFile(sc, params.input)
			case LP => MLUtils.loadLabeledData(sc, params.input)
		}

		// split data into training and test sets
		val splits = data.randomSplit(Array(params.trainSplit, 1.0 - params.trainSplit), seed = seed)
		(splits(0), splits(1))
	}
}
