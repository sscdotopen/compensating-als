Compensating ALS
================

In this repository, we provide code and instructions to repeat the experiments regarding the compensation of failures during the computation of a low-rank matrix factorization with the ALS algorithm.

We modified an existing parallel ALS implementation from the [Apache Mahout](http://mahout.apache.org) library to simulate failures and apply our proposed compensation function. The source code can be found in the [CompensatingALSFactorizer](https://github.com/sscdotopen/compensating-als/blob/master/src/main/java/io/ssc/compensatingals/CompensatingALSFactorizer.java) class.

In order to repeat our experiments, you first need to download the *movielens-1M* dataset from [http://www.grouplens.org/node/73](http://www.grouplens.org/node/73).

The dataset has to be converted to Mahout's input format for recommenders like this:

`cat ratings.dat |sed -e s/::/,/g| cut -d, -f1,2,3 > ratings.csv`

The code for running the experiments can be found in the [RunExperiment](https://github.com/sscdotopen/compensating-als/blob/master/src/main/java/io/ssc/compensatingals/RunExperiment.java) class. Simply adjust the constants for the dataset location, failing iteration and failing percentage and run the code as a standard Java program.
