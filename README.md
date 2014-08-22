Scalable predictive models
==========================

The aim of the project is predictive modelling, e.g. classification and regression tasks.

The first goal is to apply the existing algorithms on datasets with varying size and to evaluate how scalable those algorithms are.
Furthermore, we want to run the algorithms on [Apache Spark](http://spark.apache.org/) cluster computing engine, which seems
to be an ideal tool for large-scale data processing. Our experiments shall answer the question from which dataset size the
Spark computing engine is advantageous over the traditional approaches.

In the second stage of the project, we will experiment with the streaming feature of Spark, i.e. an ability to process streaming
data in real-time. Here, the goal is to implement a system that can run several models in parallel and adaptively switch models
based on their actual performance. This is especially useful in dynamically changing environments where a static model soon becomes
inaccurate. An example application of such a system could be real-time detection of spam e-mails.

For more info see the [project blog](http://esc-scalable-models.blogspot.cz/).
