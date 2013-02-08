package io.ssc.compensatingals;

import com.google.common.collect.Lists;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.recommender.svd.ALSWRFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.AbstractFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorization;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.iterator.FixedSizeSamplingIterator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.als.AlternatingLeastSquaresSolver;
import org.apache.mahout.math.als.ImplicitFeedbackAlternatingLeastSquaresSolver;
import org.apache.mahout.math.set.OpenLongHashSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.mahout.math.map.OpenIntObjectHashMap;

import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * factorizes the rating matrix using "Alternating-Least-Squares with Weighted-λ-Regularization" as described in the paper
 * <a href="http://www.hpl.hp.com/personal/Robert_Schreiber/papers/2008%20AAIM%20Netflix/netflix_aaim08(submitted).pdf">
 * "Large-scale Collaborative Filtering for the Netflix Prize"</a>
 *
 *  also supports the implicit feedback variant of this approach as described in "Collaborative Filtering for Implicit Feedback Datasets"
 *  available at http://research.yahoo.com/pub/2433
 */
public class CompensatingALSFactorizer extends AbstractFactorizer {

  private final DataModel dataModel;

  /** number of features used to compute this factorization */
  private final int numFeatures;
  /** parameter to control the regularization */
  private final double lambda;
  /** number of iterations */
  private final int numIterations;

  private final boolean usesImplicitFeedback;
  /** confidence weighting parameter, only necessary when working with implicit feedback */
  private final double alpha;

  private final int numTrainingThreads;

  private static final double DEFAULT_ALPHA = 40;

  private static final Logger log = LoggerFactory.getLogger(ALSWRFactorizer.class);

  public CompensatingALSFactorizer(DataModel dataModel, int numFeatures, double lambda, int numIterations,
      boolean usesImplicitFeedback, double alpha, int numTrainingThreads) throws TasteException {
    super(dataModel);
    this.dataModel = dataModel;
    this.numFeatures = numFeatures;
    this.lambda = lambda;
    this.numIterations = numIterations;
    this.usesImplicitFeedback = usesImplicitFeedback;
    this.alpha = alpha;
    this.numTrainingThreads = numTrainingThreads;
  }

  public CompensatingALSFactorizer(DataModel dataModel, int numFeatures, double lambda, int numIterations,
      boolean usesImplicitFeedback, double alpha) throws TasteException {
    this(dataModel, numFeatures, lambda, numIterations, usesImplicitFeedback, alpha,
        Runtime.getRuntime().availableProcessors());
  }

  public CompensatingALSFactorizer(DataModel dataModel, int numFeatures, double lambda, int numIterations)
      throws TasteException {
    this(dataModel, numFeatures, lambda, numIterations, false, DEFAULT_ALPHA);
  }

  private int failingIteration = 1000;
  private double failingPercentage = 0;

  public CompensatingALSFactorizer(DataModel dataModel, int numFeatures, double lambda, int numIterations, int failingIteration, double failingPercentage)
      throws TasteException {
    this(dataModel, numFeatures, lambda, numIterations, false, DEFAULT_ALPHA);
    this.failingPercentage = failingPercentage;
    this.failingIteration = failingIteration;
  }

  static class Features {

    private final DataModel dataModel;
    private final int numFeatures;

    private final double[][] M;
    private final double[][] U;

    Features(CompensatingALSFactorizer factorizer) throws TasteException {
      dataModel = factorizer.dataModel;
      numFeatures = factorizer.numFeatures;
      Random random = RandomUtils.getRandom();
      M = new double[dataModel.getNumItems()][numFeatures];
      LongPrimitiveIterator itemIDsIterator = dataModel.getItemIDs();
      while (itemIDsIterator.hasNext()) {
        long itemID = itemIDsIterator.nextLong();
        int itemIDIndex = factorizer.itemIndex(itemID);
        M[itemIDIndex][0] = averateRating(itemID);
        for (int feature = 1; feature < numFeatures; feature++) {
          M[itemIDIndex][feature] = random.nextDouble() * 0.1;
        }
      }
      U = new double[dataModel.getNumUsers()][numFeatures];
    }

    double[][] getM() {
      return M;
    }

    double[][] getU() {
      return U;
    }

    Vector getUserFeatureColumn(int index) {
      return new DenseVector(U[index]);
    }

    Vector getItemFeatureColumn(int index) {
      return new DenseVector(M[index]);
    }

    void setFeatureColumnInU(int idIndex, Vector vector) {
      setFeatureColumn(U, idIndex, vector);
    }

    void setFeatureColumnInM(int idIndex, Vector vector) {
      setFeatureColumn(M, idIndex, vector);
    }

    protected void setFeatureColumn(double[][] matrix, int idIndex, Vector vector) {
      for (int feature = 0; feature < numFeatures; feature++) {
        matrix[idIndex][feature] = vector.get(feature);
      }
    }

    protected double averateRating(long itemID) throws TasteException {
      PreferenceArray prefs = dataModel.getPreferencesForItem(itemID);
      RunningAverage avg = new FullRunningAverage();
      for (Preference pref : prefs) {
        avg.addDatum(pref.getValue());
      }
      return avg.getAverage();
    }
  }

  protected double averateItemRating(long itemID) {
    PreferenceArray prefs = null;
    try {
      prefs = dataModel.getPreferencesForItem(itemID);
    } catch (TasteException e) {
      throw new RuntimeException(e);
    }
    RunningAverage avg = new FullRunningAverage();
    for (Preference pref : prefs) {
      avg.addDatum(pref.getValue());
    }
    return avg.getAverage();
  }

  protected double averateUserRating(long userID) {
    PreferenceArray prefs = null;
    try {
      prefs = dataModel.getPreferencesFromUser(userID);
    } catch (TasteException e) {
      throw new RuntimeException(e);
    }
    RunningAverage avg = new FullRunningAverage();
    for (Preference pref : prefs) {
      avg.addDatum(pref.getValue());
    }
    return avg.getAverage();
  }

  @Override
  public Factorization factorize() throws TasteException {
    log.info("starting to compute the factorization...");
    final Features features = new Features(this);

    final Random random = RandomUtils.getRandom();
    StringBuilder events = new StringBuilder();
    StringBuilder convergence = new StringBuilder();

    /* feature maps necessary for solving for implicit feedback */
    OpenIntObjectHashMap<Vector> userY = null;
    OpenIntObjectHashMap<Vector> itemY = null;

    if (usesImplicitFeedback) {
      userY = userFeaturesMapping(dataModel.getUserIDs(), dataModel.getNumUsers(), features.getU());
      itemY = itemFeaturesMapping(dataModel.getItemIDs(), dataModel.getNumItems(), features.getM());
    }

    for (int iteration = 0; iteration < numIterations; iteration++) {
      log.info("iteration {}", iteration);

      final OpenLongHashSet failedItems = new OpenLongHashSet(0);
      final OpenLongHashSet failedUsers = new OpenLongHashSet(0);

      if (iteration == failingIteration) {
        int numFailedItems = (int) (dataModel.getNumItems() * failingPercentage);
        failedItems.ensureCapacity(numFailedItems);
        Iterator<Long> samples = new FixedSizeSamplingIterator<Long>(numFailedItems, dataModel.getItemIDs());
        while (samples.hasNext()) {
          failedItems.add(samples.next());
        }

        int numFailedUsers = (int) (dataModel.getNumUsers() * failingPercentage);
        failedUsers.ensureCapacity(numFailedUsers);
        samples = new FixedSizeSamplingIterator<Long>(numFailedUsers, dataModel.getItemIDs());
        while (samples.hasNext()) {
          failedUsers.add(samples.next());
        }

        events.append("Failure in iteration: " + failingIteration + ", " + numFailedItems +" items and " + numFailedUsers + " users lost");
      }

      /* fix M - compute U */
      ExecutorService queue = createQueue();
      LongPrimitiveIterator userIDsIterator = dataModel.getUserIDs();
      try {

        final ImplicitFeedbackAlternatingLeastSquaresSolver implicitFeedbackSolver = usesImplicitFeedback ?
            new ImplicitFeedbackAlternatingLeastSquaresSolver(numFeatures, lambda, alpha, itemY) : null;

        while (userIDsIterator.hasNext()) {
          final long userID = userIDsIterator.nextLong();
          final LongPrimitiveIterator itemIDsFromUser = dataModel.getItemIDsFromUser(userID).iterator();
          final PreferenceArray userPrefs = dataModel.getPreferencesFromUser(userID);
          queue.execute(new Runnable() {
            @Override
            public void run() {
              List<Vector> featureVectors = Lists.newArrayList();
              while (itemIDsFromUser.hasNext()) {
                long itemID = itemIDsFromUser.nextLong();
                if (!failedItems.contains(itemID)) {
                  featureVectors.add(features.getItemFeatureColumn(itemIndex(itemID)));
                }
              }

              if (!featureVectors.isEmpty()) {
                Vector userFeatures = usesImplicitFeedback ?
                    implicitFeedbackSolver.solve(sparseUserRatingVector(userPrefs)) :
                    AlternatingLeastSquaresSolver.solve(featureVectors, ratingVector(userPrefs, failedItems, true), lambda, numFeatures);
                features.setFeatureColumnInU(userIndex(userID), userFeatures);
              } else {

                Vector reinitializedFeatures = new DenseVector(numFeatures);
                reinitializedFeatures.setQuick(0, averateUserRating(userID));
                for (int n = 1; n < numFeatures; n++) {
                  reinitializedFeatures.setQuick(n, random.nextDouble() * 0.1);
                }

                features.setFeatureColumnInU(userIndex(userID), reinitializedFeatures);
              }
            }
          });
        }
      } finally {
        queue.shutdown();
        try {
          queue.awaitTermination(dataModel.getNumUsers(), TimeUnit.SECONDS);
        } catch (InterruptedException e) {
          log.warn("Error when computing user features", e);
        }
      }

      /* fix U - compute M */
      queue = createQueue();
      LongPrimitiveIterator itemIDsIterator = dataModel.getItemIDs();
      try {

        final ImplicitFeedbackAlternatingLeastSquaresSolver implicitFeedbackSolver = usesImplicitFeedback ?
            new ImplicitFeedbackAlternatingLeastSquaresSolver(numFeatures, lambda, alpha, userY) : null;

        while (itemIDsIterator.hasNext()) {
          final long itemID = itemIDsIterator.nextLong();
          final PreferenceArray itemPrefs = dataModel.getPreferencesForItem(itemID);
          queue.execute(new Runnable() {
            @Override
            public void run() {
              List<Vector> featureVectors = Lists.newArrayList();
              for (Preference pref : itemPrefs) {
                long userID = pref.getUserID();
                if (!failedUsers.contains(userID)) {
                  featureVectors.add(features.getUserFeatureColumn(userIndex(userID)));
                }
              }

              if (!featureVectors.isEmpty()) {
                Vector itemFeatures = usesImplicitFeedback ?
                    implicitFeedbackSolver.solve(sparseItemRatingVector(itemPrefs)) :
                    AlternatingLeastSquaresSolver.solve(featureVectors, ratingVector(itemPrefs, failedUsers, false), lambda, numFeatures);

                features.setFeatureColumnInM(itemIndex(itemID), itemFeatures);
              } else {

                Vector reinitializedFeatures = new DenseVector(numFeatures);
                reinitializedFeatures.setQuick(0, averateItemRating(itemID));
                for (int n = 1; n < numFeatures; n++) {
                  reinitializedFeatures.setQuick(n, random.nextDouble() * 0.1);
                }

                features.setFeatureColumnInM(itemIndex(itemID), reinitializedFeatures);
              }
            }
          });
        }
      } finally {
        queue.shutdown();
        try {
          queue.awaitTermination(dataModel.getNumItems(), TimeUnit.SECONDS);
        } catch (InterruptedException e) {
          log.warn("Error when computing item features", e);
        }
      }

      // Compute training error
      RunningAverage avg = new FullRunningAverage();
      LongPrimitiveIterator userIDs = dataModel.getUserIDs();
      while (userIDs.hasNext()) {
        long userID = userIDs.nextLong();
        for (Preference preference : dataModel.getPreferencesFromUser(userID)) {
          long itemID = preference.getItemID();
          double prediction = features.getItemFeatureColumn(itemIndex(itemID))
              .dot(features.getUserFeatureColumn(userIndex(userID)));
          double err = preference.getValue() - prediction;
          avg.addDatum(err * err);
        }
      }

      System.out.println("RMSE: " + Math.sqrt(avg.getAverage()));
      convergence.append(Math.sqrt(avg.getAverage()) + "\n");
    }

    log.info("finished computation of the factorization...");

    System.out.println(events);
    System.out.println(convergence);

    return createFactorization(features.getU(), features.getM());
  }

  protected ExecutorService createQueue() {
    return Executors.newFixedThreadPool(numTrainingThreads);
  }


  protected static Vector ratingVector(PreferenceArray prefs, OpenLongHashSet toFilter, boolean isItem) {
    double[] buffer = new double[prefs.length()];

    int n = 0;
    for (Preference preference : prefs) {

      if (isItem) {
        if (!toFilter.contains(preference.getItemID())) {
          buffer[n++] = preference.getValue();
        }
      } else {
        if (!toFilter.contains(preference.getUserID())) {
          buffer[n++] = preference.getValue();
        }
      }
    }

    double[] ratings = new double[n];
    System.arraycopy(buffer, 0, ratings, 0, n);

    return new DenseVector(ratings, true);
  }

  //TODO find a way to get rid of the object overhead here
  protected OpenIntObjectHashMap<Vector> itemFeaturesMapping(LongPrimitiveIterator itemIDs, int numItems,
                                                             double[][] featureMatrix) {
    OpenIntObjectHashMap<Vector> mapping = new OpenIntObjectHashMap<Vector>(numItems);
    while (itemIDs.hasNext()) {
      long itemID = itemIDs.next();
      mapping.put((int) itemID, new DenseVector(featureMatrix[itemIndex(itemID)], true));
    }

    return mapping;
  }

  protected OpenIntObjectHashMap<Vector> userFeaturesMapping(LongPrimitiveIterator userIDs, int numUsers,
                                                             double[][] featureMatrix) {
    OpenIntObjectHashMap<Vector> mapping = new OpenIntObjectHashMap<Vector>(numUsers);

    while (userIDs.hasNext()) {
      long userID = userIDs.next();
      mapping.put((int) userID, new DenseVector(featureMatrix[userIndex(userID)], true));
    }

    return mapping;
  }

  protected Vector sparseItemRatingVector(PreferenceArray prefs) {
    SequentialAccessSparseVector ratings = new SequentialAccessSparseVector(Integer.MAX_VALUE, prefs.length());
    for (Preference preference : prefs) {
      ratings.set((int) preference.getUserID(), preference.getValue());
    }
    return ratings;
  }

  protected Vector sparseUserRatingVector(PreferenceArray prefs) {
    SequentialAccessSparseVector ratings = new SequentialAccessSparseVector(Integer.MAX_VALUE, prefs.length());
    for (Preference preference : prefs) {
      ratings.set((int) preference.getItemID(), preference.getValue());
    }
    return ratings;
  }
}
