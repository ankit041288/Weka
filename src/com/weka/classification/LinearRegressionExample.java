package com.weka.classification;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.M5P;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.util.Random;

public class LinearRegressionExample {

    public static void main(String[] args) throws Exception {

        // load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("resource/ENB2012_data.csv"));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 2);
      //  System.out.println(data);
        //remove last attribute Y2
        Remove remove = new Remove();
        remove.setOptions(new String[]{"-R", data.numAttributes()+""});
        remove.setInputFormat(data);
        data = Filter.useFilter(data, remove);
       // System.out.println(data);
        // Regression modle
        LinearRegression model = new LinearRegression();
        model.buildClassifier(data);
       // System.out.println(model);


        M5P m5p = new M5P();
        m5p.setOptions(new String[]{""});
        m5p.buildClassifier(data);
        System.out.println(m5p);


        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(
                model, data, 10, new Random(1), new String[]{});
        System.out.println(eval.toSummaryString());




    }
}
