# PowerSystem_RiskAssessment
Welcome to the project for the course-" Machine Learning for Energy System Applications".
Workflow on an energy systems application of risk assessment – Data pre-processing, standardization, normalization. – Feature selection – Model selection and training

   
## Workflow
### Data pre-processing (feature_engineering.py)
The data pre-processing consists of three sub tasks:
- Feature removal based on variance
- Create 'reduction' dataset for both regression and classification task
- Create 'statistical' dataset for both regression and classification task

### Regression (regression.py)
The following tasks are performed under this:
- Compare the following models performance based on default parameters
   - Lasso regression
   - Ridge regression
   - SVR- linear
   - SVR- polynomial
   - SVR- gaussian
   - Deep neural network
- Grid search hyper parameter for the best chosen model (Neural network model)
   - This task is done for each type of dataset
- Classify risk state based on risk factor predicted by the regression model

### Classification (classification.py)
The following tasks are performed under this:
Calling the classification pipeline function does everything
- If Choose model is said to True:
   -When calling the classification pipeline function:
   -specify the dataset type to be read
   -specify the chosen model
   - Choose and fine tune any of the models:
   - SVC- Linear
   - SVC- Polynomial
   - SVC-RBF
   - Random Forests Classifier
   - Extreme Trees Classifier
   - MLP Neural Network
   - Get the validation accuracy, and test accuracy, per class precision, recall and f1 score
   for the fine-tuned model, also get the trained model

- If Choose model is said to False:
   -When calling the classification pipeline function:
   -specify the hyperparameters dictionary with the final chosen hyperparameters
   -specify the dataset type to be read
   -specify the chosen model
   -The final model of type 'x' is chosen and the test results are printed, the trained model is returned


## How to run the files?
- Download everything as submitted into a single folder
- Go to 'main.py' file
- Depending on the task un comment and run the pipeline functions.
