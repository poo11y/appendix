import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    feat_sort = np.sort(feature_vector)
    uniq_feat = np.unique(feat_sort)
    if (len(uniq_feat) == 1) or (len(uniq_feat) == 0):
        return np.array([]), np.array([]), -np.inf, -np.inf
    index_sort = np.argsort(feature_vector)
    targ_sort = target_vector[index_sort]
    thresholds = (uniq_feat[:-1] + uniq_feat[1:]) / 2
    
    count_ones = (target_vector == 1).sum()
    count_zeros = (target_vector == 0).sum()

    count_ones_left = np.cumsum(targ_sort == 1)
    count_zer_left = np.cumsum(targ_sort == 0)
    index_tr = np.searchsorted(feat_sort, thresholds, side='right')
    ones_left = count_ones_left[index_tr - 1]
    zeros_left = count_zer_left[index_tr - 1]
    ones_right = count_ones - ones_left
    zeros_right = count_zeros - zeros_left

    R_l = ones_left + zeros_left
    R_r = ones_right + zeros_right

    p_0_left = zeros_left / (R_l)
    p_1_left = ones_left / (R_l)
    p_0_right = zeros_right / (R_r)
    p_1_right = ones_right / (R_r)

    H_l = 1 - p_0_left ** 2 - p_1_left ** 2
    H_r = 1 - p_0_right ** 2 - p_1_right ** 2
    
    ginis = -(R_l / len(target_vector)) * H_l -(R_r / len(target_vector)) * H_r
    if (len(ginis) == 0) or (np.all(np.isnan(ginis))):
        return thresholds, ginis, -np.inf, -np.inf
    else:
        gini_best = np.max(ginis)
        threshold_best = thresholds[np.argmax(ginis)]
    

        return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if (self._max_depth is not None) and (depth >= self._max_depth):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if (self._min_samples_split is not None) and (sub_X.shape[0] < self._min_samples_split):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] =  current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if (len(feature_vector) <= 3) or (len(np.unique(feature_vector)) == 1):
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        len_left_leaf = np.sum(split)
        len_right_leaf = sub_X.shape[0] - len_left_leaf

        if self._min_samples_leaf is not None:
            if (len_left_leaf < self._min_samples_leaf) or (len_right_leaf < self._min_samples_leaf):
                node["type"] = "terminal"
                node["class"] = Counter(sub_y).most_common(1)[0][0]
                return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)


    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']
        feature_best = node['feature_split']
        if self._feature_types[feature_best] == 'real':
            if x[feature_best] < node['threshold']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        elif self._feature_types[feature_best] == 'categorical':
            if x[feature_best] in node['categories_split']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])


    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

class LinearRegressionTree(DecisionTree):
    def __init__(self, feature_types, base_model_type=None, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        pass
