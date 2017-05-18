package com.weka.classification;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

import javax.swing.*;
import java.util.Random;


public class ZooClassification {

    public static void main(String[] args) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("resource/zoo.csv");
        Instances data = source.getDataSet();
        System.out.println(data.toString() + " instances loaded.");
        StringToWordVector stwv = new StringToWordVector();
        stwv.setInputFormat(data);
        try {
            data = Filter.useFilter(data, stwv);
        } catch (Exception e) {
            e.printStackTrace();
        }

       // System.out.println(data.toString());
/*       Remove remove = new Remove();
        String[] opts = new String[]{ "-R", "1"};
        remove.setOptions(opts);
        remove.setInputFormat(data);
        data = Filter.useFilter(data, remove);
        data.setClassIndex(data.numAttributes()-1);*/
	// write your code here


        // Now Feature Selection
        data.setClassIndex(data.numAttributes()-1);
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker search = new Ranker();

        AttributeSelection attSelect = new AttributeSelection();
        attSelect.setEvaluator(eval);
        attSelect.setSearch(search);
        attSelect.SelectAttributes(data);

        int[] indices = attSelect.selectedAttributes();
        System.out.println(Utils.arrayToString(indices));

        //  Learning Algorithm ( j48 is a decision tree algorithm )

        J48 tree = new J48();
        String[] options = new String[1];
        options[0] = "-U";

        tree.setOptions(options);

        tree.buildClassifier(data);

        System.out.println(tree);

        // Test Data



        TreeVisualizer tv = new TreeVisualizer(null, tree.graph(), new PlaceNode2());
        JFrame frame = new javax.swing.JFrame("Tree Visualizer");
        frame.setSize(800, 500);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(tv);
        frame.setVisible(true);
        tv.fitToScreen();

        Classifier cl = new J48();
        Evaluation eval_roc = new Evaluation(data);
        eval_roc.crossValidateModel(cl, data, 10, new Random(1), new Object[] {});
        System.out.println(eval_roc.toSummaryString());




    }
}
