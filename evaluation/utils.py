import tensorflow_gan as tfgan
import numpy as np
import torch
import random
import logging
from tqdm import tqdm
from scipy import linalg
from scipy.stats import entropy
from collections import defaultdict
import xgboost

from sklearn import linear_model, ensemble, naive_bayes, svm, tree, discriminant_analysis
from sklearn.metrics import accuracy_score

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
session = tf.compat.v1.InteractiveSession()


def set_logger(f, level="INFO"):
    logger = logging.getLogger()

    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')

    handler = logging.StreamHandler(open(f, "w"))
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(level)


def seed_everthing(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def normalize_data(x_train, x_test):
    mean = np.mean(x_train)
    sdev = np.std(x_train)
    x_train_normed = (x_train - mean) / sdev
    x_test_normed = (x_test - mean) / sdev
    assert not np.any(np.isnan(x_train_normed)) and not np.any(
        np.isnan(x_test_normed))

    return x_train_normed, x_test_normed


def train_test_sklearn_model(model, x_tr, y_tr, x_ts, y_ts, norm_data=False):
    x_tr, x_ts = normalize_data(x_tr, x_ts) if norm_data else (x_tr, x_ts)
    model.fit(x_tr, y_tr)
    y_pred = model.predict(x_ts)
    acc = accuracy_score(y_pred, y_ts)
    return acc


def prep_sklearn_models():
    models = {'logistic_reg': linear_model.LogisticRegression,
              'random_forest': ensemble.RandomForestClassifier,
              'gaussian_nb': naive_bayes.GaussianNB,
              'bernoulli_nb': naive_bayes.BernoulliNB,
              'linear_svc': svm.LinearSVC,
              'decision_tree': tree.DecisionTreeClassifier,
              'lda': discriminant_analysis.LinearDiscriminantAnalysis,
              'adaboost': ensemble.AdaBoostClassifier,
              # 'mlp': neural_network.MLPClassifier,
              'bagging': ensemble.BaggingClassifier,
              'gbm': ensemble.GradientBoostingClassifier,
              'xgboost': xgboost.XGBClassifier}

    model_specs = defaultdict(dict)
    model_specs['logistic_reg'] = {
        'solver': 'lbfgs', 'max_iter': 5000, 'multi_class': 'auto'}
    model_specs['random_forest'] = {
        'n_estimators': 100, 'class_weight': 'balanced'}
    model_specs['linear_svc'] = {
        'max_iter': 10000, 'tol': 1e-8, 'loss': 'hinge'}
    model_specs['bernoulli_nb'] = {'binarize': 0.5}
    model_specs['lda'] = {'solver': 'eigen',
                          'n_components': 9, 'tol': 1e-8, 'shrinkage': 0.5}
    model_specs['decision_tree'] = {'class_weight': 'balanced', 'criterion': 'gini', 'splitter': 'best',
                                    'min_samples_split': 2, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0,
                                    'min_impurity_decrease': 0.0}
    # setting used in neurips2020 submission
    model_specs['adaboost'] = {'n_estimators': 100, 'algorithm': 'SAMME.R'}
    # model_specs['adaboost'] = {'n_estimators': 100, 'learning_rate': 0.1, 'algorithm': 'SAMME.R'}  best so far
    model_specs['bagging'] = {'max_samples': 0.1, 'n_estimators': 20}
    model_specs['gbm'] = {'subsample': 0.1, 'n_estimators': 50}
    model_specs['xgboost'] = {'colsample_bytree': 0.1,
                              'objective': 'multi:softprob', 'n_estimators': 50}

    return models, model_specs


def train_sklearn_models(gen_images, gen_labels, dataset):
    x_gen, y_gen = gen_images, gen_labels
    x_gen = np.reshape(x_gen, (x_gen.shape[0], -1))

    if dataset == "mnist":
        from torchvision.datasets import MNIST
        dataset = MNIST(root="../_data", train=False, download=True)
    elif dataset == "fmnist":
        from torchvision.datasets import FashionMNIST
        dataset = FashionMNIST(root="../_data", train=False, download=True)
    else:
        raise NotImplementedError

    x_real, y_real = dataset.data.numpy(), dataset.targets.numpy()
    x_real = np.reshape(x_real, (x_real.shape[0], -1)) / 255.

    models, model_specs = prep_sklearn_models()
    accs = []
    for key in models:
        logging.info(f"Training {key}......")
        model = models[key](**model_specs[key])
        model.n_jobs = -1
        acc = train_test_sklearn_model(
            model, x_tr=x_gen, y_tr=y_gen, x_ts=x_real, y_ts=y_real)
        accs.append(acc)
    mean_acc = np.mean(accs)
    return mean_acc


def train_classifier(
    model: torch.nn.Module,
    train_dataloader,
    val_dataloader,
    optimizer,
    epochs=50,
):
    loss_fn = torch.nn.CrossEntropyLoss()
    device = next(model.parameters()).device
    best_acc = 0
    best_model_state_dict = model.state_dict().copy()
    for epoch in range(epochs):
        model.train()
        for X, y in tqdm(train_dataloader, desc=f"Epoch{epoch + 1}"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for X, y in tqdm(val_dataloader, desc=f"Epoch{epoch + 1}"):
                X, y = X.to(device), y.to(device)
                output = model(X)
                _, pred = torch.max(output, dim=1)

                total += y.size(0)
                correct += (y == pred).sum().item()
            acc = correct / total
            if acc > best_acc:
                best_acc = acc
                best_model_state_dict = model.state_dict().copy()
    model.load_state_dict(best_model_state_dict)
    return best_acc


@torch.no_grad()
def test_classifier(model: torch.nn.Module, dataloader):
    model.eval()
    device = next(model.parameters()).device
    total = 0
    correct = 0
    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)
        output = model(X)
        _, pred = torch.max(output, dim=1)

        total += y.size(0)
        correct += (y == pred).sum().item()
    acc = correct / total
    return acc


# INCEPTION_TFHUB = tfgan.eval.INCEPTION_TFHUB
INCEPTION_TFHUB = "_models/tfgan_eval_inception_1"

def compute_activation_and_logits(images, batch_size=128):

    images_input = tf.compat.v1.placeholder(
        tf.float32, [None, None, None, 3], name='images_input')

    def get_inception_output(images=images_input):
        size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
        images = tf.compat.v1.image.resize_bilinear(images, [size, size])
        classifier = tfgan.eval.classifier_fn_from_tfhub(
            INCEPTION_TFHUB,
            [tfgan.eval.INCEPTION_OUTPUT, tfgan.eval.INCEPTION_FINAL_POOL],
            False,
        )
        output = classifier(images)
        return output

    func = get_inception_output()

    session = tf.get_default_session()
    n_batches = int(np.ceil(float(images.shape[0]) / batch_size))
    acts = []
    logits = []
    for i in tqdm(range(n_batches)):
        batch = images[i * batch_size:(i + 1) * batch_size]
        output = session.run(func, {images_input: batch})
        acts.append(output[tfgan.eval.INCEPTION_FINAL_POOL])
        logits.append(output[tfgan.eval.INCEPTION_OUTPUT])
    acts = np.concatenate(acts, axis=0)
    logits = np.concatenate(logits, axis=0)
    logits = np.exp(logits) / np.sum(np.exp(logits), 1, keepdims=True)
    return acts, logits


def compute_inception_score_from_logits(logits, splits=1):
    scores = []
    for i in range(splits):
        part = logits[(i * logits.shape[0] // splits):((i + 1) * logits.shape[0] // splits), :]
        py = np.mean(part, axis=0)
        kl = np.mean([entropy(part[i, :], py) for i in range(part.shape[0])])
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def compute_activation_stat(acts):
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma


def compute_fid_score(mu1, sigma1, mu2, sigma2):
    m = np.square(mu1 - mu2).sum()
    s, _ = linalg.sqrtm(np.dot(sigma1, sigma2), disp=False)
    fid = np.real(m + np.trace(sigma1 + sigma2 - s * 2))
    return fid
