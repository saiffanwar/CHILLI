import numpy as np
import matplotlib.pyplot as plt
from lime_utils import lime_tabular
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.utils.extmath import safe_sparse_dot
from pprint import pprint

class CHILLI():

    def __init__(self, model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, features):
        self.model = model
        # These should be scaled numpy arrays
        self.x_train = x_train
        self.y_train = y_train
        self.y_train_pred = y_train_pred
        self.x_test = x_test
        self.y_test = y_test
        self.y_test_pred = y_test_pred
        self.features = features


    def build_explainer(self, categorical_features=None, kernel_width=None, mode='regression'):
#        The explainer is built herem on the training data with the features and type of model specified.
#        y_hat_test = self.model.predict(self.x_test)
        explainer = lime_tabular.LimeTabularExplainer(self.x_train, test_data=self.x_test, test_labels=self.y_test, test_predictions=self.y_test_pred, feature_names=self.features, categorical_features=categorical_features, mode=mode, verbose=False, kernel_width=kernel_width)
        return explainer

    def make_explanation(self, predictor, explainer, instance, num_features=25, num_samples=1000):
        ground_truth = self.y_test[instance]
        instance_prediction = predictor(self.x_test[instance].reshape(1,-1))[0]
        exp, local_model, perturbations, model_perturbation_predictions, exp_perturbation_predictions = explainer.explain_instance(self.x_test[instance], instance_num=instance, predict_fn=predictor, num_features=num_features, num_samples=num_samples)
        self.local_model = local_model
        exp_instance_prediction = local_model.predict(self.x_test[instance].reshape(1,-1))[0]

        explanation_error = mean_squared_error(model_perturbation_predictions, exp_perturbation_predictions, squared=False)
        exp.intercept = self.local_model.intercept_

        return exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error, instance_prediction, exp_instance_prediction

    def exp_sorter(self, exp_list, features):

        explained_features = [i[0] for i in exp_list]
        for e in explained_features:
            if len(e.split('=')) >1:
                explained_features[explained_features.index(e)] = e.split('=')[0]

        feature_contributions = {f:[] for f in features}
        contributions = [e[1] for e in exp_list]

        explained_feature_indices = [features.index(i) for i in explained_features]
        for f in features:
            for num, e in enumerate(explained_features):
                if e == f:
#                    sorted_exp.append(e[1])
                    feature_contributions[f].append(contributions[explained_features.index(e)])
        sorted_exp = [feature_contributions[f][0] for f in features]

        return sorted_exp


    def plot_explanation(self, instance, exp, kernel_width, perturbations, model_perturbation_predictions, exp_perturbation_predictions, targetFeature):
        fontsize = 10
        fig, axes = plt.subplots(1,1,figsize=(5,5))
        plt.tight_layout()

        exp_size = 12
        # Plot the explanation
        exp_list = exp.as_list()

        feature_contributions = self.exp_sorter(exp_list, self.features)
        explained_features = self.features
        explained_features_x_test = self.x_test
#
        instance_x, instance_model_y, instance_exp_y = np.array(perturbations[0]), int(np.array(model_perturbation_predictions[0])), int(np.array(exp_perturbation_predictions[0]))
        perturbations_x, perturbations_model_y, perturbations_exp_y = np.array(perturbations[1:]), np.array(model_perturbation_predictions[1:]), np.array(exp_perturbation_predictions[1:])
        perturbations_model_y = [int(i) for i in perturbations_model_y]
        explanation_error = mean_squared_error(perturbations_model_y, perturbations_exp_y)

        colours = ['green' if x>= 0 else 'red' for x in feature_contributions]

        axes.set_yticks(range(len(explained_features)))
        axes.set_yticklabels(explained_features, rotation=0, fontsize=fontsize)
        axes.barh(explained_features, feature_contributions, color=colours, align='center', label='_nolegend_')
        axes.tick_params(axis='both', which='major', labelsize=14)
        axes.set_title(f'Explanation for instance {instance} \n Explanation Error = {explanation_error:.2f} \n Model Instance Prediction {instance_model_y} \n Explanation Instance Prediction {instance_exp_y}', fontsize=fontsize)
        axes.set_xlabel('Feature Contribution', fontsize=fontsize)
        # expPlot.text(-3000,2.5, 'a)', fontsize=14)
#        expPlot.set_xlim(-2000, 2000)

        fig.savefig(f'Explanations/{instance}_explanation.pdf', bbox_inches='tight')


    def interactive_perturbation_plot(self, instance, exp, kernel_width, perturbations, model_perturbation_predictions, exp_perturbation_predictions, targetFeature):

        exp_list = exp.as_list()

        feature_contributions = self.exp_sorter(exp_list, self.features)
        explained_features = self.features
        explained_features_x_test = self.x_test
#
        instance_x, instance_model_y, instance_exp_y = np.array(perturbations[0]), int(np.array(model_perturbation_predictions[0])), int(np.array(exp_perturbation_predictions[0]))
        perturbations_x, perturbations_model_y, perturbations_exp_y = np.array(perturbations[1:]), np.array(model_perturbation_predictions[1:]), np.array(exp_perturbation_predictions[1:])
        perturbations_model_y = [int(i) for i in perturbations_model_y]
        explanation_error = mean_squared_error(perturbations_model_y, perturbations_exp_y)
        print(instance_model_y, instance_exp_y, explanation_error)


        perturbation_weights = exp.weights[1:]

        num_rows=int(np.ceil(len(explained_features)/4))+1
        num_cols=4

        fig = make_subplots(rows=num_rows, cols=num_cols, column_widths=[0.25, 0.25, 0.25, 0.25], row_heights =[0.33]+[0.16]*(num_rows-1),
                            specs = [
                                [{'colspan':2}, None, {'colspan':2}, None],
                                ]+[[{}, {}, {}, {}] for i in range(num_rows-1)], subplot_titles=['Explanation Prediction Convergence', 'Feature Significance']+explained_features,
                            horizontal_spacing=0.05, vertical_spacing=0.05)

        colours = ['green' if x>= 0 else 'red' for x in feature_contributions]

        # Plot explanation bar chart
        fig.add_trace(go.Bar(x=feature_contributions, y=explained_features, marker_color=colours, orientation='h', showlegend=False), row=1, col=3)

        axes = [[row, col] for row in range(2,num_rows+1) for col in range(1,num_cols+1)]

#
        for i in range(len(self.features)):
            if i==0:
                showlegend=True
            else:
                showlegend=False
            fig.add_trace(go.Scatter(x=explained_features_x_test[:,i],y=self.y_test_pred,
                                     mode='markers', marker = dict(color='lightgrey', size=3, opacity=0.9),
                                     showlegend=showlegend, name='Test data'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=perturbations_x[:,i],y=perturbations_model_y,
                                     mode='markers', marker = dict(color=perturbation_weights, colorscale='Oranges', size=3, opacity=0.9),
                                     showlegend=showlegend, name='Model (f) predictions for perturbations'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=perturbations_x[:,i],y=perturbations_exp_y,
                                     mode='markers', marker = dict(color=perturbation_weights, colorscale='Greens', size=3, opacity=0.9),
                                     showlegend=showlegend, name='Explanation (g) predictions for perturbations'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=[instance_x[i]],y=[instance_model_y],
                                     mode='markers', marker = dict(color='red', size=20),
                                     showlegend=showlegend, name='Instance being explained'),
                          row=axes[i][0], col=axes[i][1])
            fig.add_trace(go.Scatter(x=[instance_x[i]],y=[exp.local_model.predict(instance_x.reshape(1,-1))],
                                     mode='markers', marker = dict(color='blue', size=10, opacity=0.9),
                                     showlegend=showlegend, name='Explanation (g) predictions for perturbations'),
                          row=axes[i][0], col=axes[i][1])


        fig.update_layout(title=dict(text = f' Explanation for instance {instance} <br> Explanation Error = {explanation_error:.2f} <br> Model Instance Prediction {instance_model_y} <br> Explanation Instance Prediction {instance_exp_y}', y=0.99, x=0),
                          font=dict(size=14),
                          legend=dict(yanchor="top", y=1.1, xanchor="right"),
                          height=300*num_rows, )
        fig.write_html(f'Explanations/instance_{instance}_kw={kernel_width}_explanation.html', auto_open=False)

def chilli_explain(model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred,
                   features, targetFeature, instance, kernel_width=None, categorical_features=None, plot_exp=True):
    chilliExplainer = CHILLI(model, x_train, y_train, y_train_pred, x_test, y_test, y_test_pred, features)
    print(features)
    categorical_features = [features.index(c) for c in  categorical_features]
    explainer = chilliExplainer.build_explainer(mode='regression', kernel_width=kernel_width, categorical_features=categorical_features)
    exp, perturbations, model_perturbation_predictions, exp_perturbation_predictions, explanation_error, instance_prediction, exp_instance_prediction = chilliExplainer.make_explanation(model, explainer, instance=instance, num_samples=1000)

    print('Base Model Prediction: ', y_test[instance])
    print('CHILLI Explanation Prediction: ', exp_instance_prediction)
    print(f'CHILLI Error: {abs(y_test[instance]-exp_instance_prediction)}')


    if plot_exp:
        chilliExplainer.plot_explanation(instance, exp, kernel_width, perturbations, model_perturbation_predictions, exp_perturbation_predictions, targetFeature)
#        chilliExplainer.interactive_perturbation_plot(instance, exp, kernel_width, perturbations, model_perturbation_predictions, exp_perturbation_predictions, 'target')


