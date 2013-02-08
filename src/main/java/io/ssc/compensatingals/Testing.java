package io.ssc.compensatingals;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;

import java.io.File;
import java.io.IOException;

public class Testing {

  public static void main(String[] args) throws IOException, TasteException {
    DataModel dataModel = new FileDataModel(new File("/home/ssc/Desktop/plista/movielens1M.csv"));

    CompensatingALSFactorizer factorizer = new CompensatingALSFactorizer(dataModel, 10, 0.065, 25, 10, 0.15);

    factorizer.factorize();

  }
}
