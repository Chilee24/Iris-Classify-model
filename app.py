import flask
import torch
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split

app = flask.Flask(__name__)


def eculid_distance(x, y):
    return torch.sqrt(torch.sum((x-y)**2, dim = 1))


class KNN:
    def __init__(self, k):
        self.k = k

    def predict(self, x_train, y_train, x_test):
        predictions = []
        for i in range(x_test.shape[0]):
            dist = eculid_distance(x_train, x_test[i])
            k_neighbors = y_train[torch.topk(dist, self.k, largest = False).indices]
            y_pred = torch.mode(k_neighbors).values.item()
            predictions.append(y_pred)
        return torch.tensor(predictions)

iris = datasets.load_iris()
x = iris.data
y = iris.target
mean = x.mean(axis = 0)
std = x.std(axis = 0)
x = (x - mean) / std
X_train = torch.tensor(x, dtype = torch.float)
Y_train = torch.tensor(y, dtype = torch.long)

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = flask.request.form.to_dict()
    features = [float(data[f'feature{i}']) for i in range(4)]
    x_test = torch.tensor([features], dtype=torch.float32)
    k = int(data['k'])

    knn = KNN(k)

    y_pred = knn.predict(X_train, Y_train, x_test)
    prediction = int(y_pred.item())
    return flask.render_template('index.html', prediction = prediction)

if __name__ == '__main__':
    app.run(debug = True)




