// Filename: p1_logistic_regression.cpp
// Author: Devin DeJongh - utdallas.edu
// Description: 
//      - Read in data from titanic_project.csv
//      - Generate a logistic regression model
//      - Start by timing the optimization process
//      - Generate metrics for model
#include <iostream>
#include <fstream>
#include <cmath>     // exp()
#include <vector>    // reading in from file 
#include <chrono>    // timing optimization
#include <string>    // stof()

using namespace std; 

const int MAX_LEN = 1046;                       // length of data set
const int TRAIN = 900;                          // length of training set 
const int TEST = 146;                           // length of test set 
const double learningRate = 0.001;              // alpha parameter

double weights[2] = {1, 1};                     // w_0 (intercept) is weights[0], w_1 os weights[1]
double* pclassTrain = new double[TRAIN];        // array of pclass for training 
double* survivedTrain = new double[TRAIN];      // array of survived for training 
double* sigmoidVector = new double[TRAIN];      // sigmoid values for opitmization (training)
double* errors = new double[TRAIN];             // array of errors for optimization (training)

double* pclassTest = new double[TEST];          // test set of pclass column 
double* survivedTest = new double[TEST];        // test set of survived column
double* predictions = new double[TEST];         // store model predictions 

// Function prototypes 
void sensitivity(double preds[], double testTargets[]);
void specificity(double preds[], double testTargets[]);
void accuracy(double preds[], double testTargets[]);
void predict(double weights[], double testPredictor[], double preds[]);
void sigmoid(double sig[], double predictor[], double ws[]);
void updateWeights(double predictor[], double errs[], double ws[], double alpha);

int main(int argc, char** argv) {
    // Objects for file reading 
    ifstream inFS; 
    string line; 
    string labelIn, pclassIn, survivedIn, sexIn, ageIn; 
    
    // Vectors for reading in files 
    vector<string> label (MAX_LEN);
    vector<double> pclass(MAX_LEN);
    vector<double> survived(MAX_LEN);
    vector<double> sex(MAX_LEN);
    vector<double> age(MAX_LEN);

    // Begin opening file 
    cout << "Opening titanic_project.csv " << endl; 
    inFS.open("titanic_project.csv"); 
    if(!inFS.is_open()) {
        cout << "Failed to open file!" << endl;
        return 1; // indicate error 
    }

    // Ouput header
    cout << "Header -> "; 
    getline(inFS, line); 
    cout << line << endl; 

    // NOTE: We read all of the values in as vectors
    // Other method resulted in misplaced values in arrays 
    // or junk binary. May change later.  

    int numObservations = 0; 
    while(inFS.good()) {
        getline(inFS, labelIn, ','); 
        getline(inFS, pclassIn, ','); 
        getline(inFS, survivedIn, ','); 
        getline(inFS, sexIn, ','); 
        getline(inFS, ageIn, '\n');

        label.at(numObservations) = labelIn; 
        pclass.at(numObservations) = stof(pclassIn); 
        survived.at(numObservations) = stof(survivedIn); 
        sex.at(numObservations) = stof(sexIn); 
        age.at(numObservations) = stof(ageIn); 

        numObservations++; 
    }

    // Create training and test sets 
    for(int i = 0; i < 900; i++) {
        // cout << "Pclass for " << i << " -> " << pclass.at(i) << endl; 
        pclassTrain[i] = pclass.at(i); 
        survivedTrain[i] = survived.at(i); 
    } 
    int index = 0; 
    for(int i = 900; i < MAX_LEN; i++) {
        pclassTest[index] = pclass.at(i); 
        survivedTest[index] = survived.at(i);
        index++;
        // cout << pclassTest[i] << endl << survivedTest[i] << endl;  
    }

    // Delete vectors now that data is transferred to arrays
    label.clear(); 
    pclass.clear(); 
    survived.clear(); 
    sex.clear(); 
    age.clear(); 
    inFS.close(); 

    // for(int i = 0; i < 900; i++) {
    //     cout << "Pclass (array) at " << i << ": " << pclassTrain[i] << endl; 
    //     cout << "Survived (array) at " << i << ": " << survivedTrain[i] << endl; 
    // }

    cout << "Beginning optimization process..." << endl;
    cout << "This may take some time." << endl << endl; 
    
    // Optimization process begins here
    auto start = chrono::high_resolution_clock::now();
    for(int i = 0; i < 5000; i++) {
        sigmoid(sigmoidVector, pclassTrain, weights);               // generate sigmoid vector 
        for(int j = 0; j < TRAIN; j++) {                            // create error vector  
            errors[j] = survivedTrain[j] - sigmoidVector[j];     
        }
        updateWeights(pclassTrain, errors, weights, learningRate);  // perform weight updates  

        // Output updates to weights 
        if(i == 10) {
            cout << "Weights at iteration " << i << ": " << weights[0] << ", " << weights[1] << endl; 
        } else if(i == 50) {
            cout << "Weights at iteration " << i << ": " << weights[0] << ", " << weights[1] << endl; 
        } else if(i == 100) {
            cout << "Weights at iteration " << i << ": " << weights[0] << ", " << weights[1] << endl; 
        } else if(i == 500) {
            cout << "Weights at iteration " << i << ": " << weights[0] << ", " << weights[1] << endl; 
        } else if(i == 1000) {
            cout << "Weights at iteration " << i << ": " << weights[0] << ", " << weights[1] << endl; 
        }
    } 
    auto stop = chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_sec = stop - start; 
    
    // Optimization complete.
    cout << endl << "Optimization complete." << endl;
    cout << "Weights after 5000 iterations: " << weights[0] << ", " << weights[1] << endl;
    cout << "Time (total): " << elapsed_sec.count() << endl << endl; 

    // Deploy model -> populate predictions[] using new weights;
    predict(weights, pclassTest, predictions); 
     
    // Calculating metrics
    cout << "Calculating metrics..." << endl; 
    accuracy(predictions, survivedTest);
    sensitivity(predictions, survivedTest); 
    specificity(predictions, survivedTest); 

    return 0; // success
}

// calculate sensitivity using predictions and test set of target variable
// sensitivity <- TP / (TP + FN)
void sensitivity(double preds[], double testTargets[]) {
    double truePositive = 0; 
    double falseNegative = 0; 
    double sensitivity = 0; 
    for(int i = 0; i < TEST; i++) {
        if((preds[i] == 1) && (testTargets[i] == 1)) {
            truePositive++; 
        } else if((preds[i] == 0) && (testTargets[i] == 1)) {
            falseNegative++; 
        }
    }

    sensitivity = (truePositive) / (truePositive + falseNegative);

    cout << "Sensitivity: " << sensitivity << endl; 
}

// calculate specificity using predictions and test set of target variable
// specificity <- TN / (TN + FP)
void specificity(double preds[], double testTargets[]) {
    double trueNegative = 0;
    double falsePositive = 0;
    double specificity = 0; 
    for(int i = 0; i < TEST; i++) {
        if((preds[i] == 0) && (testTargets[i] == 0)) {
            trueNegative++; 
        } else if((preds[i] == 1) && (testTargets[i] == 0)) {
            falsePositive++; 
        }
    }

    specificity = (trueNegative) / (trueNegative + falsePositive);

    cout << "Specificity: " << specificity << endl; 
}

// calculate accuracy using predictions and test set of target variable
// accuracy <- Correct / Total
void accuracy(double preds[], double testTargets[]) {
    double correct = 0; 
    double accuracy = 0; 
    for(int i = 0; i < TEST; i++) {
        if(preds[i] == testTargets[i]) {
            correct++; 
        }
    }
    accuracy = correct / TEST; 
    cout << "Accuracy: " << accuracy << endl; 
}

// Apply logistic regression algorithm
// Find log odds -> w_0 + w_1(x)
// Extract probability through logistic function 
void predict(double weights[], double testPredictor[], double preds[]) {
    double modelPredictions = 0; 
    double logOdds = 0;  
    for(int i = 0; i < TEST; i++) {
        logOdds = weights[0] + (weights[1] * testPredictor[i]);
        modelPredictions = (1 / (1 + exp( -1 * logOdds))); 
        if(modelPredictions < 0.5) {
            preds[i] = 0; 
        } else {
            preds[i] = 1; 
        }
    } // for 
} // predict 

// Performs "matrix multiplication" for weights = weights + alpha * (t(data_matrix) %*% errors)
void updateWeights(double predictor[], double errs[], double ws[], double alpha) {    
    double updates[2] = {0, 0}; 
    for(int i = 0; i < TRAIN; i++) {
        updates[0] = updates[0] + errs[i]; 
        updates[1] = updates[1] + (predictor[i] * errs[i]); 
    }
    // cout << "Updates: " << updates[0] << ", " << updates[1] << endl; 
    ws[0] = ws[0] + (alpha * updates[0]);
    ws[1] = ws[1] + (alpha * updates[1]);
}

// Generates sigmoid values for each data entry
// This is meant to mimick matrix multiplication from w^T(x) for 1/(1+e^-w^Tx)
void sigmoid(double sig[], double predictor[], double ws[]) { // fine 
    double intercept = 0; 
    double weightProduct = 0; 
    double param = 0; 
    double result = 0; 
    for(int i = 0; i < TRAIN; i++) {
        intercept = ws[0] * 1.0;
        weightProduct = ws[1] * predictor[i]; 
        param = intercept + weightProduct;
        result = 1.0 / (1.0 + exp(param * -1.0)); 
        sig[i] = result;   
    }
}