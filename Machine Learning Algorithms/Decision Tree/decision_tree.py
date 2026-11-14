import numpy as np
import heapq


class DecisionTree:
	"""
	A decision tree classifier with sklearn-compatible API.
	Supports multiple criteria, class/sample weights, cost-complexity pruning,
	incremental impurity computation, best-first growth, and more.
	"""

	def __init__(self, criterion='gini', max_depth=10,
	             min_samples_split=2, min_samples_leaf=1,
	             min_weight_fraction_leaf=0.0, max_features='sqrt',
	             random_state=None, max_leaf_nodes=None,
	             min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0):
		self.criterion = criterion
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf
		self.min_weight_fraction_leaf = min_weight_fraction_leaf
		self.max_features = max_features
		self.random_state = random_state
		self.max_leaf_nodes = max_leaf_nodes
		self.min_impurity_decrease = min_impurity_decrease
		self.class_weight = class_weight
		self.ccp_alpha = ccp_alpha
		self._rng = None
		self._tree = None
		self.classes_ = None
		self.n_classes_ = None
		self.n_features_in_ = None
		self.n_outputs_ = 1

	def _impurity(self, counts, total):
		if total == 0:
			return 0.0
		p = counts / total
		p = p[p > 0]
		if self.criterion == 'gini':
			return 1.0 - np.sum(p ** 2)
		return -np.sum(p * np.log2(p))

	def _get_max_features(self):
		mf = self.max_features
		n = self.n_features_in_
		if mf is None:
			return n
		if mf == 'sqrt':
			return max(1, int(np.sqrt(n)))
		if mf == 'log2':
			return max(1, int(np.log2(n)))
		if isinstance(mf, (int, np.integer)):
			return max(1, min(int(mf), n))
		if isinstance(mf, float):
			return max(1, int(mf * n))
		return n

	def _compute_sample_weight(self, y):
		if self.class_weight is None:
			return None
		if isinstance(self.class_weight, dict):
			w = np.array([self.class_weight.get(c, 1.0) for c in self.classes_], dtype=np.float64)
		elif self.class_weight == 'balanced':
			counts = np.bincount(y, minlength=self.n_classes_)
			w = len(y) / (self.n_classes_ * counts.astype(np.float64))
		else:
			raise ValueError(f"class_weight={self.class_weight} not supported")
		return w[y]

	def _make_leaf(self, y, sample_weight):
		counts = np.bincount(y, minlength=self.n_classes_, weights=sample_weight)
		wsum = np.sum(counts)
		return {
			'leaf': True,
			'value': int(np.argmax(counts)),
			'counts': counts,
			'impurity': self._impurity(counts, wsum),
			'weighted_n_node_samples': wsum,
			'n_node_samples': len(y),
		}

	def _find_best_split(self, X, y, sample_weight):
		n, nf = X.shape
		sw = sample_weight
		wsum = np.sum(sw) if sw is not None else n

		n_sub = self._get_max_features()
		feature_idx = self._rng.choice(nf, n_sub, replace=False)

		wpar = np.bincount(y, minlength=self.n_classes_, weights=sw)
		par_imp = self._impurity(wpar, wsum)

		min_wfrac = self.min_weight_fraction_leaf * wsum

		best_feat = None
		best_thresh = None
		best_imp = np.inf

		for j in feature_idx:
			ind = np.argsort(X[:, j], kind='stable')
			Xj = X[ind, j]
			yj = y[ind]
			wj = sw[ind] if sw is not None else np.ones(n)

			uv = np.unique(Xj)
			if len(uv) < 2:
				continue

			left_c = np.zeros(self.n_classes_, dtype=np.float64)
			right_c = np.bincount(yj, minlength=self.n_classes_, weights=wj).astype(np.float64)
			left_w = 0.0
			right_w = np.sum(wj)

			for i in range(n):
				c = yj[i]
				wt = wj[i]
				left_c[c] += wt
				right_c[c] -= wt
				left_w += wt
				right_w -= wt

				if left_w < self.min_samples_leaf or left_w < min_wfrac:
					continue
				if right_w < self.min_samples_leaf or right_w < min_wfrac:
					break

				if i < n - 1 and Xj[i] != Xj[i + 1]:
					l_imp = self._impurity(left_c, left_w)
					r_imp = self._impurity(right_c, right_w)
					w_imp = (left_w / wsum) * l_imp + (right_w / wsum) * r_imp

					if self.min_impurity_decrease > 0.0:
						if par_imp - w_imp < self.min_impurity_decrease:
							continue

					if w_imp < best_imp:
						best_imp = w_imp
						best_feat = j
						best_thresh = (Xj[i] + Xj[i + 1]) / 2.0

		if np.isinf(best_imp):
			return None, None, None
		return best_feat, best_thresh, best_imp

	def _grow_depth_first(self, X, y, sample_weight, depth):
		n = len(y)
		sw = sample_weight
		wsum = np.sum(sw) if sw is not None else n

		if (self.max_depth is not None and depth >= self.max_depth) or \
		   n < self.min_samples_split or \
		   len(np.unique(y)) == 1 or \
		   wsum < 2 * self.min_samples_leaf:
			return self._make_leaf(y, sw)

		feature, threshold, imp = self._find_best_split(X, y, sw)
		if feature is None:
			return self._make_leaf(y, sw)

		mask = X[:, feature] <= threshold
		X_l, X_r = X[mask], X[~mask]
		y_l, y_r = y[mask], y[~mask]
		sw_l = sw[mask] if sw is not None else None
		sw_r = sw[~mask] if sw is not None else None

		return {
			'leaf': False,
			'feature': feature,
			'threshold': threshold,
			'impurity': imp,
			'weighted_n_node_samples': wsum,
			'n_node_samples': n,
			'counts': np.bincount(y, minlength=self.n_classes_, weights=sw),
			'left': self._grow_depth_first(X_l, y_l, sw_l, depth + 1),
			'right': self._grow_depth_first(X_r, y_r, sw_r, depth + 1),
		}

	def _grow_best_first(self, X, y, sample_weight):
		root = self._make_leaf(y, sample_weight)
		leaves = 1

		heap = []
		feat, thresh, imp = self._find_best_split(X, y, sample_weight)
		if feat is not None:
			wsum = np.sum(sample_weight) if sample_weight is not None else len(y)
			parent_imp = root['impurity']
			improvement = parent_imp - imp
			heapq.heappush(heap, (-improvement, 0, id(root), root,
			                      X, y, sample_weight, feat, thresh, 1))

		while heap:
			neg_imp, _, _, node, Xn, yn, swn, feat, thresh, depth = heapq.heappop(heap)

			if self.max_depth is not None and depth >= self.max_depth:
				continue

			if self.max_leaf_nodes is not None and leaves >= self.max_leaf_nodes:
				continue

			mask = Xn[:, feat] <= thresh
			X_l, X_r = Xn[mask], Xn[~mask]
			y_l, y_r = yn[mask], yn[~mask]
			sw_l = swn[mask] if swn is not None else None
			sw_r = swn[~mask] if swn is not None else None

			left = self._make_leaf(y_l, sw_l)
			right = self._make_leaf(y_r, sw_r)
			leaves += 1

			wsum = np.sum(swn) if swn is not None else len(yn)
			node.clear()
			node.update({
				'leaf': False,
				'feature': feat,
				'threshold': thresh,
				'impurity': imp,
				'weighted_n_node_samples': wsum,
				'n_node_samples': len(yn),
				'counts': np.bincount(yn, minlength=self.n_classes_, weights=swn),
				'left': left,
				'right': right,
			})

			for cx, cy, csw, cn in [(X_l, y_l, sw_l, left), (X_r, y_r, sw_r, right)]:
				if len(cy) < self.min_samples_split or len(np.unique(cy)) == 1:
					continue
				f, t, i = self._find_best_split(cx, cy, csw)
				if f is not None:
					cw = np.sum(csw) if csw is not None else len(cy)
					par_i = self._impurity(
						np.bincount(cy, minlength=self.n_classes_, weights=csw), cw)
					improvement = par_i - i
					if improvement > self.min_impurity_decrease:
						heapq.heappush(heap, (-improvement, -(depth + 1), id(cn),
						                      cn, cx, cy, csw, f, t, depth + 1))

		return root

	def _compute_leaf_stats(self, node):
		if node['leaf']:
			node['_n_leaves'] = 1
			node['_sub_imp'] = node['impurity'] * node['weighted_n_node_samples']
		else:
			self._compute_leaf_stats(node['left'])
			self._compute_leaf_stats(node['right'])
			node['_n_leaves'] = node['left']['_n_leaves'] + node['right']['_n_leaves']
			node['_sub_imp'] = node['left']['_sub_imp'] + node['right']['_sub_imp']

	def _prune(self):
		if self.ccp_alpha <= 0.0:
			return
		self._compute_leaf_stats(self._tree)
		while True:
			best_node = None
			best_alpha = np.inf

			def _find(node):
				nonlocal best_node, best_alpha
				if node.get('leaf'):
					return
				_find(node['left'])
				_find(node['right'])
				if node['_n_leaves'] > 1:
					r_t = node['impurity'] * node['weighted_n_node_samples']
					r_tt = node['_sub_imp']
					alpha = (r_t - r_tt) / (node['_n_leaves'] - 1)
					if alpha < best_alpha:
						best_alpha = alpha
						best_node = node

			_find(self._tree)
			if best_alpha > self.ccp_alpha or best_node is None:
				break

			n_samp = int(best_node['n_node_samples'])
			val = int(np.argmax(best_node['counts']))
			best_node.clear()
			best_node.update(self._make_leaf(np.full(n_samp, val), None))
			self._compute_leaf_stats(self._tree)

	def fit(self, X, y, sample_weight=None):
		X = np.asarray(X, dtype=np.float64)
		y = np.asarray(y)
		if X.ndim != 2:
			raise ValueError("X must be 2D")
		if y.ndim != 1:
			raise ValueError("y must be 1D")
		if len(X) != len(y):
			raise ValueError("X and y length mismatch")

		self.n_features_in_ = X.shape[1]
		self.classes_ = np.unique(y)
		self.n_classes_ = len(self.classes_)
		self._rng = np.random.default_rng(self.random_state)

		y_enc = np.searchsorted(self.classes_, y)

		if sample_weight is not None:
			sample_weight = np.asarray(sample_weight, dtype=np.float64)
			if sample_weight.shape != (len(y),):
				raise ValueError("sample_weight shape mismatch")

		cw = self._compute_sample_weight(y_enc)
		if cw is not None:
			sample_weight = cw if sample_weight is None else sample_weight * cw

		if self.max_leaf_nodes is not None:
			self._tree = self._grow_best_first(X, y_enc, sample_weight)
		else:
			self._tree = self._grow_depth_first(X, y_enc, sample_weight, 0)

		self._prune()
		return self

	def _predict_one(self, node, x):
		while not node['leaf']:
			node = node['left'] if x[node['feature']] <= node['threshold'] else node['right']
		return node['value']

	def predict(self, X):
		X = np.asarray(X, dtype=np.float64)
		return np.array([self._predict_one(self._tree, x) for x in X])

	def predict_proba(self, X):
		X = np.asarray(X, dtype=np.float64)
		probas = []
		for x in X:
			node = self._tree
			while not node['leaf']:
				node = node['left'] if x[node['feature']] <= node['threshold'] else node['right']
			counts = node['counts']
			probas.append(counts / np.sum(counts))
		return np.array(probas)

	def apply(self, X):
		X = np.asarray(X, dtype=np.float64)
		leaf_ids = []
		stack = [(self._tree, 0)]
		next_id = 0
		node_to_id = {}

		def _assign_ids(node):
			nonlocal next_id
			if id(node) not in node_to_id:
				node_to_id[id(node)] = next_id
				next_id += 1
			if not node['leaf']:
				_assign_ids(node['left'])
				_assign_ids(node['right'])

		_assign_ids(self._tree)

		for x in X:
			node = self._tree
			while not node['leaf']:
				node = node['left'] if x[node['feature']] <= node['threshold'] else node['right']
			leaf_ids.append(node_to_id[id(node)])
		return np.array(leaf_ids)

	def get_depth(self):
		def _depth(node):
			if node['leaf']:
				return 0
			return 1 + max(_depth(node['left']), _depth(node['right']))
		return _depth(self._tree)

	def get_n_leaves(self):
		def _count(node):
			if node['leaf']:
				return 1
			return _count(node['left']) + _count(node['right'])
		return _count(self._tree)

	@property
	def feature_importances_(self):
		total = 0.0
		contrib = np.zeros(self.n_features_in_)

		def _accum(node, parent_weight):
			nonlocal total
			if node['leaf']:
				return
			w = node['weighted_n_node_samples']
			lw = node['left']['weighted_n_node_samples']
			rw = node['right']['weighted_n_node_samples']
			li = node['left']['impurity']
			ri = node['right']['impurity']
			imp_dec = w * node['impurity'] - lw * li - rw * ri
			contrib[node['feature']] += imp_dec
			total += imp_dec
			_accum(node['left'], w)
			_accum(node['right'], w)

		_accum(self._tree, self._tree['weighted_n_node_samples'])
		if total > 0:
			return contrib / total
		return contrib
