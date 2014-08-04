package cz.cvut.esc.models

import scopt.OptionParser

/**
 * CLI (Command Line Interface) application.
 */
trait CliApp[P] {

	/**
	 * Returns option parameter parser with default values
	 * @param args input arguments
	 * @return default params and the parameter parser
	 */
	def paramsParser(args: Array[String]): (OptionParser[P], P)

	def run(params: P): Unit

	/** The main method. */
	def main(args: Array[String]) {
		val (parser, default) = paramsParser(args)
		parser.parse(args, default).map { params =>
			run(params)
		} getOrElse {
			System.exit(1)
		}
	}
}
