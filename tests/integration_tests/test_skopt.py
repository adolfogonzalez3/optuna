from mock import call
from mock import patch
import pytest
from skopt.space import space

import optuna
from optuna import distributions
from optuna.structs import FrozenTrial

if optuna.types.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA


def test_conversion_from_distribution_to_dimenstion():
    # type: () -> None

    sampler = optuna.integration.SkoptSampler()
    study = optuna.create_study(sampler=sampler)
    with patch('skopt.Optimizer') as mock_object:
        study.optimize(_objective, n_trials=2, catch=())

        dimensions = [
            # Original: trial.suggest_uniform('p0', -3.3, 5.2)
            space.Real(-3.3, 5.2),

            # Original: trial.suggest_uniform('p1', 2.0, 2.0)
            # => Skipped because `skopt.Optimizer` cannot handle an empty `Real` dimension.

            # Original: trial.suggest_loguniform('p2', 0.0001, 0.3)
            space.Real(0.0001, 0.3, prior='log-uniform'),

            # Original: trial.suggest_loguniform('p3', 1.1, 1.1)
            # => Skipped because `skopt.Optimizer` cannot handle an empty `Real` dimension.

            # Original: trial.suggest_int('p4', -100, 8)
            space.Integer(-100, 8),

            # Original: trial.suggest_int('p5', -20, -20)
            # => Skipped because `skopt.Optimizer` cannot handle an empty `Real` dimension.

            # Original: trial.suggest_discrete_uniform('p6', 10, 20, 2)
            space.Integer(0, 5),

            # Original: trial.suggest_discrete_uniform('p7', 0.1, 1.0, 0.1)
            space.Integer(0, 8),

            # Original: trial.suggest_discrete_uniform('p8', 2.2, 2.2, 0.5)
            # => Skipped because `skopt.Optimizer` cannot handle an empty `Real` dimension.

            # Original: trial.suggest_categorical('p9', ['9', '3', '0', '8'])
            space.Categorical(('9', '3', '0', '8'))
        ]
        assert mock_object.mock_calls[0] == call(dimensions)


def test_skopt_kwargs():
    # type: () -> None

    sampler = optuna.integration.SkoptSampler(skopt_kwargs={'base_estimator': "GBRT"})
    study = optuna.create_study(sampler=sampler)

    with patch('skopt.Optimizer') as mock_object:
        study.optimize(lambda t: t.suggest_int('x', -10, 10), n_trials=2)

        dimensions = [space.Integer(-10, 10)]
        assert mock_object.mock_calls[0] == call(dimensions, base_estimator="GBRT")


def test_skopt_kwargs_dimenstions():
    # type: () -> None

    # User specified `dimensions` argument will be ignored in `SkoptSampler`.
    sampler = optuna.integration.SkoptSampler(skopt_kwargs={'dimensions': []})
    study = optuna.create_study(sampler=sampler)

    with patch('skopt.Optimizer') as mock_object:
        study.optimize(lambda t: t.suggest_int('x', -10, 10), n_trials=2)

        expected_dimensions = [space.Integer(-10, 10)]
        assert mock_object.mock_calls[0] == call(expected_dimensions)


def test_is_compatible():
    # type: () -> None

    sampler = optuna.integration.SkoptSampler()
    study = optuna.create_study(sampler=sampler)

    study.optimize(lambda t: t.suggest_uniform('p0', 0, 10), n_trials=1)
    search_space = optuna.samplers.product_search_space(study)
    assert search_space == {'p0': distributions.UniformDistribution(low=0, high=10)}

    optimizer = optuna.integration.skopt._Optimizer(search_space, {})

    # Compatible.
    trial = _create_frozen_trial({'p0': 5},
                                 {'p0': distributions.UniformDistribution(low=0, high=10)})
    assert optimizer._is_compatible(trial)

    # Compatible.
    trial = _create_frozen_trial({'p0': 5},
                                 {'p0': distributions.UniformDistribution(low=0, high=100)})
    assert optimizer._is_compatible(trial)

    # Compatible.
    trial = _create_frozen_trial({
        'p0': 5,
        'p1': 7
    }, {
        'p0': distributions.UniformDistribution(low=0, high=10),
        'p1': distributions.UniformDistribution(low=0, high=10)
    })
    assert optimizer._is_compatible(trial)

    # Incompatible ('p0' doesn't exist).
    trial = _create_frozen_trial({'p1': 5},
                                 {'p1': distributions.UniformDistribution(low=0, high=10)})
    assert not optimizer._is_compatible(trial)

    # Incompatible (the value of 'p0' is out of range).
    trial = _create_frozen_trial({'p0': 20},
                                 {'p0': distributions.UniformDistribution(low=0, high=100)})
    assert not optimizer._is_compatible(trial)

    # Error (different distribution class).
    trial = _create_frozen_trial({'p0': 5},
                                 {'p0': distributions.IntUniformDistribution(low=0, high=10)})
    with pytest.raises(ValueError):
        optimizer._is_compatible(trial)


def _objective(trial):
    # type: (optuna.trial.Trial) -> float

    p0 = trial.suggest_uniform('p0', -3.3, 5.2)
    p1 = trial.suggest_uniform('p1', 2.0, 2.0)
    p2 = trial.suggest_loguniform('p2', 0.0001, 0.3)
    p3 = trial.suggest_loguniform('p3', 1.1, 1.1)
    p4 = trial.suggest_int('p4', -100, 8)
    p5 = trial.suggest_int('p5', -20, -20)
    p6 = trial.suggest_discrete_uniform('p6', 10, 20, 2)
    p7 = trial.suggest_discrete_uniform('p7', 0.1, 1.0, 0.1)
    p8 = trial.suggest_discrete_uniform('p8', 2.2, 2.2, 0.5)
    p9 = trial.suggest_categorical('p9', ['9', '3', '0', '8'])

    return p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + int(p9)


def _create_frozen_trial(params, param_distributions):
    # type: (Dict[str, Any], Dict[str, distributions.BaseDistribution]) -> FrozenTrial

    params_in_internal_repr = {}
    for param_name, param_value in params.items():
        params_in_internal_repr[param_name] = param_distributions[param_name].to_internal_repr(
            param_value)

    return FrozenTrial(
        number=0,
        value=1.,
        state=optuna.structs.TrialState.COMPLETE,
        user_attrs={},
        system_attrs={},
        params=params,
        params_in_internal_repr=params_in_internal_repr,
        distributions=param_distributions,
        intermediate_values={},
        datetime_start=None,
        datetime_complete=None,
        trial_id=0,
    )
