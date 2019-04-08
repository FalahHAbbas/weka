import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {

    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("weather.arff");
        Instances data = source.getDataSet();
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        String[] options = new String[1];
        options[0] = "-U";
        J48 tree = new J48();
        tree.setOptions(options);
        tree.buildClassifier(data);
        System.out.println(tree.graph());

//        checkkk();
        long s=System.currentTimeMillis();
        check();
        long e=System.currentTimeMillis();

        System.out.println(e-s);
    }

//    private static void checkkk() throws Exception {
//
//        ConverterUtils.DataSource source1 = new ConverterUtils.DataSource("weather.arff");
//        Instances train = source1.getDataSet();
//        // setting class attribute if the data format does not provide this information
//        // For example, the XRFF format saves the class attribute information as well
//        if (train.classIndex() == -1)
//            train.setClassIndex(train.numAttributes() - 1);
//
//        ConverterUtils.DataSource source2 = new ConverterUtils.DataSource("test.arff");
//        Instances test = source2.getDataSet();
//        // setting class attribute if the data format does not provide this information
//        // For example, the XRFF format saves the class attribute information as well
//        if (test.classIndex() == -1)
//            test.setClassIndex(train.numAttributes() - 1);
//
//        // model
//
//        NaiveBayes naiveBayes = new NaiveBayes();
//        naiveBayes.buildClassifier(train);
//
//        // this does the trick
//        double label = naiveBayes.classifyInstance(test.instance(0));
//        test.instance(0).setClassValue(label);
//
//        System.out.println(test.instance(0).stringValue(4));
//    }

    private static void check() throws Exception {
        DataSource source1 = new DataSource("results.arff");
        Instances train = source1.getDataSet();
        if (train.classIndex() == -1)
            train.setClassIndex(train.numAttributes() - 1);

        DataSource source2 = new DataSource("check.arff");
        Instances test = source2.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (test.classIndex() == -1)
            test.setClassIndex(train.numAttributes() - 1);

        // model

        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(train);

        // this does the trick
        double label = naiveBayes.classifyInstance(test.instance(0));
        test.instance(0).setClassValue(label);

        System.out.println(test.instance(0).stringValue(4));
    }
}
