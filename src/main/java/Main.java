import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {

    public static void main(String[] args) throws Exception {

        long s = System.currentTimeMillis();
        check();
        long e = System.currentTimeMillis();
        System.out.println(e - s);
    }

    private static void check() throws Exception {
        DataSource source1 = new DataSource("results.arff");
        Instances train = source1.getDataSet();
        if (train.classIndex() == -1)
            train.setClassIndex(train.numAttributes() - 1);

        DataSource source2 = new DataSource("check.arff");
        Instances test = source2.getDataSet();
        if (test.classIndex() == -1)
            test.setClassIndex(train.numAttributes() - 1);

        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(train);

        double label = naiveBayes.classifyInstance(test.instance(0));
        test.instance(0).setClassValue(label);
        System.out.println(test.instance(0).stringValue(4));
    }

}
