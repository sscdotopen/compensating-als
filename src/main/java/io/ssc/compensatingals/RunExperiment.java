package io.ssc.compensatingals;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;

import java.io.File;
import java.io.IOException;

public class RunExperiment {

  private static final String DATASET_LOCATION = "/home/ssc/movielens1M.csv";
  private static final int FAILING_ITERATION = 10;
  private static final double FAILING_PERCENTAGE = 0.15;

  public static void main(String[] args) throws IOException, TasteException {
    DataModel dataModel = new FileDataModel(new File(DATASET_LOCATION));

    CompensatingALSFactorizer factorizer = new CompensatingALSFactorizer(dataModel, 10, 0.065, 25, FAILING_ITERATION,
        FAILING_PERCENTAGE);

    factorizer.factorize();
  }
}
