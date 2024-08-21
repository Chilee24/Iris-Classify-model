import flask
import torch
import torch.nn as nn
import sklearn.datasets as datasets

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
    features = [float(data[f'feature{i}']) for i in range(4)]
    x_test = torch.tensor([features], dtype=torch.float32)

    if model == 'knn':
        knn = KNN(3)
        y_pred = knn.predict(X_train, Y_train, x_test)
        prediction = int(y_pred.item())

    elif model == 'linear':
        linear_model = Linear(4, 3)
        linear_model.load_state_dict(torch.load('linear_model.pth'))
        linear_model.eval()
        with torch.no_grad():
            y_pred = linear_model(x_test)
        prediction = torch.argmax(y_pred, dim=1).item()

    return flask.render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
