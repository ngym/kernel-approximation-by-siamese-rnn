"""Base class for Neural Networks.
"""

import numpy as np
import os
results_dir='results/' + str(os.getpid())
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
import pickle
import json
import time
import traceback
import signal
from scipy.signal import gaussian
from sklearn.metrics import f1_score
from hyperopt import Trials, STATUS_OK, STATUS_FAIL, tpe, fmin, hp
from hyperopt.pyll import scope


class Algorithm:
    """Base class for Neural Networks.

    :param hyperparams: Hyperparameter configuration
    :type hyperparams: dict
    """

    def __init__(self, hyperparams):
        assert(isinstance(hyperparams, dict))
        self.input_ndims = {'X': 3, 'Y': 2}
        self.output_ndims = {'Y_hat': 2, 'time_mean': 0}
        self.hyperparams = hyperparams
        
    def __del__(self):
        return

    def check_tensors(self, tensors, ndims):
        """Assertions for tensor arguments.

        :param tensors: Input tensors
        :param ndims: Expected dimensions for tensors
        :type tensors: dict
        :type ndims: dict
        """

        assert(isinstance(tensors, dict))
        assert(set(tensors.keys()) <= set(ndims.keys()))
        assert(all([isinstance(tensors[k], np.ndarray) for k in list(tensors.keys())]))
        assert(all([tensors[k].ndim == ndims[k] for k in list(tensors.keys())]))
        # assert(len(set([tensors[k].shape[0] for k in list(tensors.keys())])) == 1)
        
    def check_hyperparams(self, hyperparams):
        """Assertions for hyperparameter configuration.

        :param hyperparams: Hyperparameter configuration
        :type hyperparams: dict
        """

        assert(isinstance(hyperparams, dict))
        assert(set(hyperparams.keys()) <= set(self.hyperparams.keys()))
        assert(all([(isinstance(v, self.hyperparams[k].__class__) or v is None) for (k, v) in hyperparams.items()]))
        
    def set_hyperparams(self, hyperparams):
        """Set hyperparameter configuration.

        :param hyperparams: Hyperparameter configuration
        :type hyperparams: dict
        """

        self.check_hyperparams(hyperparams)
        for (k, v) in hyperparams.items():
            if v is not None:
                self.hyperparams[k] = v
        print(self.hyperparams)
        self.build()

    def build(self):
        """Build algorithm with the current hyperparameter configuration.
        """

        return
        
    def train(self, tensors_train, tensors_val, validate=True):
        """Train algorithm with optional validaton.

        :param tensors_train: Input tensors for training
        :param tensors_val: Input tensors for validation
        :param validate: Whether to apply validation or not
        :type tensors_train: dict
        :type tensors_val: dict
        :type validate: bool
        :returns: Validation results
        :rtype: dict
        """

        return

    def test(self, tensors):
        """Validate algorithm.
        
        :param tensors: Input tensors for testing
        :type tensors: dict
        :returns: Results
        :rtype: dict
        """

        return

    def predict(self, tensors):
        """Compute outputs with algorithm.

        :param tensors: Input tensors for prediction
        :type tensors: dict
        """

        return
        
    def hyperopt_save_trials(self, trials, filename=results_dir + '/trials.p'):
        """Save results log of hyperparameter optimization.

        :param trials: Log of results
        :param filename: Path for saving file
        :type trials: Trials
        :type filename: str
        """

        pickle.dump(trials, open(filename, 'wb'))

    def hyperopt_load_best_trial(self, filename=results_dir + '/trials.p'):
        """Load best hyperparameter configuration from results log optimization.

        :param filename: Path for loading file
        :type filename: str
        :returns: Best hyperparameter configuration
        :rtype: Trials
        """

        trials = pickle.load(open(filename, 'rb'))
        best_idx = \
            np.nanargmin(np.asarray([trial.get('result').get('loss') if (trial.get('result').get('status') == STATUS_OK)
                                     else np.nan for trial in list(trials)]))
        best_trial = list(trials)[best_idx]
        best_result = best_trial.get('result')
        best_hyperparams = best_result.get('hyperparams')
        self.set_hyperparams(best_hyperparams)
        return best_trial

    def hyperopt_parse_node(self, k, v):
        """Parse hyperparameter configuration for one parameter.

        :param k: Name of parameter
        :param v: Configuration of parameter
        :type k: str
        :type v: dict
        :returns: Parsed configuration
        :rtype: Hyperopt expression
        """

        assert (isinstance(v['distribution'], str))
        if v['distribution'] is 'switch':
            # v_new = scope.int(hp.quniform(k, v['min'], v['max'], 1))
            v_new = hp.choice(k, options=list(range(v['min'], v['max'] + 1)))
        else:
            fn = getattr(hp, v['distribution'])
            if v['distribution'] is 'choice':
                v_new = fn(k, v['options'])
            else:
                if 'log' in v['distribution']:
                    v_min, v_max = np.log(v['min']), np.log(v['max'])
                else:
                    v_min, v_max = v['min'], v['max']
                if v['distribution'][0] is 'q':
                    v_new = scope.int(fn(k, v_min, v_max, v['q']))
                else:
                    v_new = fn(k, v_min, v_max)

        def transform(x):
            if 'transform' in v.keys():
                assert ('x' in v['transform'])
                return eval(v['transform'])
            else:
                return x

        return transform(v_new)

    def hyperopt_preprocess_config(self, config):
        """Parse hyperparameter configuration for all parameters.

        :param config: Hyperparameter configuration
        :type config: dict
        :returns: Parsed configuration
        :rtype: dict
        """

        assert(isinstance(config, dict))
        config_new = dict()
        config_new['algo'] = config['algo'] if 'algo' in config.keys() else tpe.suggest
        config_new['max_evals'] = config['max_evals'] if 'max_evals' in config.keys() else 200
        config_new['space'] = dict()
        for (k, v) in config['space'].items():
            if isinstance(v, dict):
                if v['distribution'] is 'switch':
                    config_new['space'][k] = self.hyperopt_parse_node(k, v)
                    for n in range(v['min'], v['max'] + 1):
                        if n != 0:
                            for _k in (set(v.keys()) - set(['distribution', 'min', 'max'])):
                                key = k + str(n) + '_' + _k
                                _v = self.hyperopt_parse_node(key, v[_k])
                                l = [config_new['space'][k]] + (v['max'] + 1) * [None]
                                if 'increment' in v.keys() and v['increment'] is True:
                                    for m in range(n, v['max'] + 1):
                                        l[m + 1] = _v
                                else:
                                    l[n + 1] = _v
                                    config_new['space'][key] = scope.switch(*l)
                else:
                    config_new['space'][k] = self.hyperopt_parse_node(k, v)
            elif isinstance(v, int) or isinstance(v, float):
                config_new['space'][k] = v
            else:
                print((k, v))
                raise NotImplementedError
        return config_new

    def hyperopt_iteration(self, hyperparams, tensors_train, tensors_val, validate=True):
        """Do one iteration of hyperparameter optimization.

        :param hyperparams: Hyperparameter configuration
        :param tensors_train: Input tensors for training
        :param tensors_val: Input tensors for validation
        :param validate: Whether to apply validation or not
        :type hyperparams: dict
        :type tensors_train: dict
        :type tensors_val: dict
        :type validate: bool
        :returns: Validation results
        :rtype: dict
        """

        def signal_handler(signum, frame):
            raise TimeoutError("Timed out!")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(840)  # Timeout seconds
        try:
            self.set_hyperparams(hyperparams)
            result = self.train(tensors_train, tensors_val, validate)
            result['hyperparams'] = self.hyperparams
            result['status'] = STATUS_OK
            print(result)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            traceback.print_exc()
            result = {'exception': str(e), 'time': time.time(), 'status': STATUS_FAIL}
        return result

    def hyperopt(self, tensors_train, tensors_val, tensors_test, config):
        """Hyperparameter optimization.

        :param tensors_train: Input tensors for training
        :param tensors_val: Input tensors for validation
        :param tensors_test: Input tensors for testing
        :param config: Settings for optimization
        :type tensors_train: dict
        :type tensors_val: dict
        :type tensors_test: dict
        :type config: dict
        :returns: Test results
        :rtype: dict
        """

        trials = Trials()
        print(config)
        preprocessed_config = self.hyperopt_preprocess_config(config)
        print(preprocessed_config)
        fmin(lambda hyperparams: self.hyperopt_iteration(hyperparams, tensors_train, tensors_val, validate=True),
             space=preprocessed_config['space'], algo=preprocessed_config['algo'],
             max_evals=preprocessed_config['max_evals'], trials=trials, return_argmin=False)
        self.hyperopt_save_trials(trials)
        # Parse best trial
        self.hyperopt_load_best_trial()
        # Reinitialize, compile and fit model on train + validation
        print("Best model:")
        return self.hyperopt_iteration(
            self.hyperparams,
            dict([(k, np.concatenate([v, tensors_val[k]], axis=0)) for (k, v) in tensors_train.items()]),
            tensors_test, validate=False)

    def load_json(self, filename=results_dir + '/1/run.json'):
        """Load best hyperparameter configuration from Sacred run.json file.

        :param filename: Path for loading file
        :type filename: str
        :returns: Loaded json
        :rtype: dict
        """

        jsn = json.load(open(filename, 'r'))
        self.set_hyperparams(jsn['result']['hyperparams'])
        return jsn
