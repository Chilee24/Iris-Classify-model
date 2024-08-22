import flask
import numpy as np
import torch
import torch.nn as nn
import sklearn.datasets as datasets
import time

app = flask.Flask(__name__)

def eculid_distance(x, y):
    return torch.sqrt(torch.sum((x-y)**2, dim=1))

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.lin(x)

class KNN:
    def __init__(self, k):
        self.k = k

    def predict(self, x_train, y_train, x_test):
        predictions = []
        for i in range(x_test.shape[0]):
            dist = eculid_distance(x_train, x_test[i])
            k_neighbors = y_train[torch.topk(dist, self.k, largest=False).indices]
            y_pred = torch.mode(k_neighbors).values.item()
            predictions.append(y_pred)
        return torch.tensor(predictions)

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_probs = None
        self.feature_means = None
        self.feature_stds = None

    def fit(self, X, y):
        # Tính xác suất prior cho mỗi lớp
        self.classes, class_counts = torch.unique(y, return_counts=True)
        self.class_probs = class_counts.float() / len(y)

        # Tính mean và std của mỗi feature cho mỗi lớp
        self.feature_means = []
        self.feature_stds = []
        for c in self.classes:
            X_c = X[y == c]
            self.feature_means.append(X_c.mean(dim=0))
            self.feature_stds.append(X_c.std(dim=0))
        self.feature_means = torch.stack(self.feature_means)
        self.feature_stds = torch.stack(self.feature_stds)

    def gaussian_likelihood(self, x, mean, std):
        exponent = -0.5 * ((x - mean) / std)**2
        log_coefficient = -torch.log(std) - 0.5 * torch.log(torch.tensor(2 * np.pi))
        return log_coefficient + exponent

    def predict(self, X):
        log_probs = []
        for c in range(len(self.classes)):
            class_log_prob = torch.log(self.class_probs[c])
            feature_log_probs = self.gaussian_likelihood(X, self.feature_means[c], self.feature_stds[c])
            total_log_prob = class_log_prob + feature_log_probs.sum(dim=-1)
            log_probs.append(total_log_prob)
        log_probs = torch.stack(log_probs, dim=-1)
        return self.classes[torch.argmax(log_probs, dim=-1)]

iris = datasets.load_iris()
x = iris.data
y = iris.target
mean = x.mean(axis=0)
std = x.std(axis=0)
x = (x - mean) / std
X_train = torch.tensor(x, dtype=torch.float)
Y_train = torch.tensor(y, dtype=torch.long)

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = flask.request.form.to_dict()
    model = data['model']
    prediction = None
    prediction_time = None
    features = [float(data[f'feature{i}']) for i in range(4)]
    
    x_test = np.array(features).reshape(1, -1)
    x_test = (x_test - mean) / std
    x_test = torch.tensor(x_test, dtype=torch.float32)
    
    prediction_time = 0
    
    if model == 'knn':
        knn = KNN(3)
        start_time = time.time()
        y_pred = knn.predict(X_train, Y_train, x_test)
        end_time = time.time()
        prediction = int(y_pred.item())
        prediction_time = end_time - start_time

    elif model == 'linear':
        linear_model = Linear(4, 3)
        linear_model.load_state_dict(torch.load('linear_model.pth'))
        linear_model.eval()
        start_time = time.time()
        with torch.no_grad():
            y_pred = linear_model(x_test)
        end_time = time.time()
        prediction = torch.argmax(y_pred, dim=1).item()
        prediction_time = end_time - start_time
    
    elif model == 'bayes':
        bayes = GaussianNaiveBayes()
        bayes.fit(X_train, Y_train)
        start_time = time.time()
        prediction = bayes.predict(x_test).item()
        end_time = time.time()
        prediction_time = end_time - start_time

    prediction_time_ms = prediction_time * 1000  

    return flask.render_template('index.html', prediction=prediction, prediction_time=prediction_time_ms)

if __name__ == '__main__':
    app.run(debug=True)