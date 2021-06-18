import abc
import logging
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

import matplotlib.pyplot as plt
import numpy as np
from opentuner.measurement.interface import MeasurementInterface
from opentuner.search.manipulator import ConfigurationManipulator, EnumParameter

from ._logging import PathLike, override_opentuner_config
from ._pareto import is_pareto_efficient

msg_logger = logging.getLogger(__name__)
KnobsT = Dict[str, str]
TunerConfigT = Dict[int, int]


class ApproxKnob:
    r"""Basic definition of an approximation knob.
    An approximation knob is an instance of a type of approximation;
    for example, Perforated Convolution is a type of approximation,
    while row-perforated convolution with stride 2 is a knob.

    :param name: The name of this approximation knob. Must be unique throughout.
    :param devices: The devices this knob can be applied on.
           Default is `None` which means all devices are supported.
    """

    def __init__(
        self, name: str, devices: List[str] = None, baseline_priority: int = None
    ):
        self.name = name
        self.devices = None if devices is None else set(devices)
        self.baseline_priority = baseline_priority

    def exists_on_device(self, device: str) -> bool:
        """Returns True if this knob can be applied to an `ApproxApp` on device `device`.

        :param device: The device to check for.
        """
        if self.devices is None:
            return True
        return device in self.devices

    def __repr__(self):
        device_str = "" if self.devices is None else str(list(self.devices))
        return f"Knob({self.name}){device_str}"

    @classmethod
    def unique_baseline(cls, knobs: Iterable["ApproxKnob"]) -> "ApproxKnob":
        baselines = set(k for k in knobs if k.baseline_priority is not None)
        if baselines:
            sorted_bases = sorted(
                baselines, key=lambda b: b.baseline_priority, reverse=True
            )
            return sorted_bases[0]
        else:
            return cls("__baseline__")


class ApproxApp(abc.ABC):
    """Generic approximable application with operator & knob enumeration,
    and measures its own QoS and cost given a configuration.
    (A configuration is a dictionary from operator name to a knob name.)
    To use this class, inherit from it and implement `name` and `measure_qos_cost`.

    :param op_knobs: a mapping from each operator (identified by str) to a list of applicable knobs.
    :type op_knobs: Dict[str, List[ApproxKnob]]
    :param target_device: the target device that this application should be tuned on.
           Each knob has a number of devices it is supported on
           (see `ApproxKnob.exists_on_device`)
           and only knobs supported on `target_device` will be used for this application.
    :type target_device: Optional[str]

    :var baseline_knob: The baseline knob of this application.
         This is derived by looking at all knobs defined in `op_knobs`
         and deciding which is the baseline.
    """

    def __init__(
        self, op_knobs: Dict[str, List[ApproxKnob]], target_device: Optional[str] = None
    ) -> None:
        super().__init__()
        self.op_knobs = op_knobs
        if target_device:
            self.op_knobs = self._filter_knob_by_device(self.op_knobs, target_device)
        # Also modifies self.op_knobs in place.
        self.baseline_knob = self._check_get_baseline_knob_(self.op_knobs)

    @property
    def ops(self) -> List[str]:
        """A list of operators in this application.

        :rtype: List[str]
        """
        return list(self.op_knobs)

    @property
    def knobs(self) -> List[ApproxKnob]:
        """A list of all unique knobs (see `ApproxKnob`)
        applicable to operators in this application.

        :rtype: List[ApproxKnob]
        """
        knob_sets = [set(knobs) for knobs in self.op_knobs.values()]
        return list(set.union(*knob_sets))

    def get_tuner(self) -> "ApproxTuner":
        """Sets up an ApproxTuner instance which the user can directly call
        `tune()` on with opentuner parameters."""
        return ApproxTuner(self)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of this application.
        Acts as an identifier in many places, so the user should try to make it unique.

        :rtype: str
        """
        return ""

    @abc.abstractmethod
    def measure_qos_cost(
        self, with_approxes: KnobsT, is_test: bool
    ) -> Tuple[float, float]:
        """Measures the QoS and cost (time, energy, ...) of a given configuration.

        :param with_approxes: The approximation configuration to measure QoS and cost for.
        :param is_test: If True, uses a "test" dataset/mode that is held away from the tuner
               during tuning; otherwise use "tune" dataset.
               How the "tune" and "test" mode behave is up to the user to define.
        """
        pass

    def add_baseline_to_knobs(self, approxes: KnobsT) -> KnobsT:
        """For each operator not appearing in the keys of configuration `approxes`
        (a dictionary), map it to the baseline (see `ApproxApp.baseline_knob`).

        `measure_qos_cost` should call this on the incoming config
        if you wish to be able to abbreviate the configuration
        (for example, you can write `measure_qos_cost({})` to get the baseline QoS).
        This ensures all operators are present when the config is sent to tuner.

        :param approxes: the config to add baseline knobs to.
        """
        return {
            op_name: approxes.get(op_name, self.baseline_knob.name)
            for op_name in self.ops
        }

    @staticmethod
    def _check_get_baseline_knob_(
        op_knobs: Dict[str, List[ApproxKnob]]
    ) -> "ApproxKnob":
        # Modifies op_knobs inplace.
        # Find the baseline knob if the user has one, or get a default one
        knob_sets = [set(knobs) for knobs in op_knobs.values()]
        knobs = list(set.union(*knob_sets))
        baseline_knob = ApproxKnob.unique_baseline(knobs)
        # Start checking if each layer has the baseline knob
        for knobs in op_knobs.values():
            if baseline_knob not in set(knobs):
                knobs.append(baseline_knob)
        return baseline_knob

    @staticmethod
    def _filter_knob_by_device(op_knobs: Dict[str, List[ApproxKnob]], device: str):
        return {
            op: [knob for knob in knobs if knob.exists_on_device(device)]
            for op, knobs in op_knobs.items()
        }


class Config:
    """An approximation configuration with its measurement results, including QoS and cost.

    :param qos: The QoS of this config (measured on tuning mode, see `ApproxApp.measure_qos_cost`).
    :param cost: The *relative* cost (time, energy, etc.) of this config
           compared to the baseline config. This is essentially :math:`1 / speedup`.
    :param knobs: The op-knob mapping in this configuration.
    :param test_qos: The QoS of this config on test mode (see `ApproxApp.measure_qos_cost`).
           This is optional as it is filled in only after the config-testing phase
           (which can be opt out of). See `ApproxTuner.tune`.
    """

    def __init__(
        self, qos: float, cost: float, knobs: KnobsT, test_qos: Optional[float] = None
    ) -> None:
        self.qos = qos
        self.cost = cost
        self.knobs = dict(sorted(knobs.items()))
        self.test_qos: Optional[float] = test_qos

    @property
    def speedup(self):
        return 1 / self.cost


T = TypeVar("T", bound=Config)


# ApproxTuner is generic over the type of the config
# So that the user can use custom Config inherited from Config
# (in which case they need to override `get_all_configs_from_db`).
class ApproxTuner(Generic[T]):
    """Supports tuning and holds all tuning results.
    `ApproxTuner.tune` is the main method for tuning.

    An instance of `ApproxTuner` can be obtained from `ApproxApp.get_tuner`.

    :param app: the application to tune.
    """

    def __init__(self, app: ApproxApp) -> None:
        self.app = app
        self._tuned = False
        self.all_configs = []
        self.kept_configs = []
        self.best_configs_prefilter = []
        self.best_configs = []
        # The following will be filled after self.tune() is called
        self.baseline_tune_qos, self.baseline_test_qos = None, None
        self.tune_keep_threshold, self.test_keep_threshold = None, None

    @property
    def tuned(self) -> bool:
        """Returns True if `tune` has been called at least once."""
        return self._tuned

    def tune(
        self,
        max_iter: int,
        qos_tuner_threshold: float,
        qos_keep_threshold: Optional[float] = None,
        is_threshold_relative: bool = False,
        take_best_n: Optional[int] = None,
        test_configs: bool = True,
        app_kwargs: dict = None
        # TODO: more parameters + opentuner param forwarding
    ) -> List[T]:
        """Runs a tuning session.

        :param max_iter: Number of iterations to use in tuning.
        :param qos_tuner_threshold: The QoS threshold that the tuner should aim for.
               QoS is assumed to be a higher-better quantity.
               This should be slightly tighter than `qos_keep_threshold`
               to account for extra error when running on test dataset.
        :param qos_keep_threshold: The QoS threshold beyond which we will keep the configuration.
               By default it is equal to `qos_keep_threshold`.
        :param is_threshold_relative: If True, the actual thresholds are considered to be
               ``baseline_qos - given_threshold``.
               This applies to `qos_tuner_threshold` and `qos_keep_threshold`.
        :param take_best_n: Take the best :math:`n` configurations after tuning.
               "Best" is defined as the configurations closest to the pareto curve
               of the QoS-cost tradeoff space.
               If `take_best_n` is None, only the configurations strictly on the
               pareto curve are taken.
        :param test_configs: If True, runs the configs on the test dataset,
               filter the taken configs by `qos_keep_threshold`,
               and fill the `test_qos` field of `Config`.
        :param app_kwargs: Additional arguments to pass to
               `ApproxApp.measure_qos_cost` during tuning.
        """

        from opentuner.tuningrunmain import TuningRunMain

        from ._dbloader import read_opentuner_db

        n_ops, n_knobs = len(self.app.ops), len(self.app.knobs)
        msg_logger.info(
            "Started tuning app %s with %d ops and %d unique knob types",
            self.app.name,
            n_ops,
            n_knobs,
        )
        msg_logger.info("Knobs: %s", self.app.knobs)
        msg_logger.info("Baseline knob: %s", self.app.baseline_knob)
        msg_logger.info("At most %d iterations", max_iter)


        import tempfile
        import os
        opentuner_args = opentuner_default_args()
        opentuner_args.database = os.path.join(tempfile.mkdtemp(), 'opentuner.db')


        tuner = self._get_tuner_interface(
            opentuner_args,
            max_iter,
            qos_tuner_threshold,
            qos_keep_threshold,
            is_threshold_relative,
            app_kwargs or {},
        )
        assert self.tune_keep_threshold is not None
        trm = TuningRunMain(tuner, opentuner_args)
        # TuningRunMain.__init__ initializes its own logger, so we'll override it and use ours
        override_opentuner_config()
        msg_logger.info(
            "Estimated size of search space: %d", trm.manipulator.search_space_size()
        )
        # A little bit of hack to get the _real_ progress when duplicated configs exist
        tuner.set_progress_getter(lambda: trm.search_driver.test_count)
        # This is where opentuner runs
        trm.main()
        # Parse and store results
        self._tuned = True
        config_ty = self._get_config_class()
        self.all_configs = [
            config_ty(result.accuracy, result.time, configuration.data)
            for result, configuration in read_opentuner_db(opentuner_args.database)
        ]
        self.kept_configs = [
            cfg for cfg in self.all_configs if cfg.qos > self.tune_keep_threshold
        ]
        self.best_configs_prefilter = self._take_best_configs(
            self.kept_configs, take_best_n
        )
        msg_logger.info(
            "Tuning finished with %d configs in total, "
            "%d configs above keeping threshold, "
            "and %d configs selected on tradeoff curve",
            len(self.all_configs),
            len(self.kept_configs),
            len(self.best_configs_prefilter),
        )
        if test_configs:
            msg_logger.info("Running configurations on test inputs")
            # Also fills in the test QoS of self.best_configs_prefilter
            self.best_configs = self._test_configs_(self.best_configs_prefilter)
        else:
            self.best_configs = self.best_configs_prefilter
        return self.best_configs

    def dump_configs(self, filepath: PathLike, best_only: bool = True):
        """Writes configuration to a JSON file.

        :param filepath: The JSON file to write into.
        :param best_only: If True, only writes the "best" configuration
               (filtered after running on test dataset, if required).
               Otherwise, writes all configurations within the given QoS threshold.
        """

        import os

        from jsonpickle import encode

        if not self.tuned:
            raise RuntimeError(
                f"No tuning session has been run; call self.tune() first."
            )
        filepath = Path(filepath)
        os.makedirs(filepath.parent, exist_ok=True)
        confs = self.best_configs if best_only else self.kept_configs
        with filepath.open("w") as f:
            f.write(encode(confs, indent=2))

    def plot_configs(
        self,
        show_qos_loss: bool = False,
        connect_best_points: bool = False,
    ) -> plt.Figure:
        """Plots 1 or 2 QoS-vs-speedup scatter plot of configurations.

        All kept configurations and all "best" configurations (before test-set filtering if any)
        are always plotted in the first subplot.
        If test-set filtering was used, the second subplot contains the "best" configurations
        plotted twice, with tune-set and test-set QoS loss respectively.

        :param show_qos_loss: If True, uses the loss of QoS (compared to the baseline)
               instead of the absolute QoS in the first graph.
               *This does not apply to the second graph* if it exists,
               which always use QoS loss for ease of comparison.
        """

        if not self.tuned:
            raise RuntimeError(
                f"No tuning session has been run; call self.tune() first."
            )
        # Without `ax` argument, this function returns if we can
        # do the second plot or not.
        dot_format = "-o" if connect_best_points else "o"
        if self.plot_test_phase():
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 6), dpi=300)
            self.plot_kept_and_best(ax0, show_qos_loss)
            self.plot_test_phase(ax1, dot_format)
        else:
            fig, ax0 = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
            self.plot_kept_and_best(ax0, show_qos_loss)
        fig.tight_layout()
        return fig

    def plot_kept_and_best(self, ax: plt.Axes, show_qos_loss: bool):
        kept_confs = self._config_qos_speedups(
            self.kept_configs, "qos", show_qos_loss, False
        )
        best_confs = self._config_qos_speedups(
            self.best_configs_prefilter, "qos", show_qos_loss, False
        )
        ax.plot(kept_confs[0], kept_confs[1], "o", label="Kept Configs")
        ax.plot(best_confs[0], best_confs[1], "o", label="Best Configs")
        self._set_xy_limit(ax, show_qos_loss)
        if show_qos_loss:
            rthres = self.baseline_tune_qos - self.tune_keep_threshold
            self._draw_qos_line(ax, rthres, f"Relative threshold: {rthres:.2f}")
            ax.set_xlabel("QoS Loss (tune dataset)")
        else:
            bqos, thres = self.baseline_tune_qos, self.tune_keep_threshold
            self._draw_qos_line(ax, bqos, f"Baseline QoS: {bqos:.2f}")
            self._draw_qos_line(ax, thres, f"Threshold: {thres:.2f}")
            ax.set_xlabel("QoS (tune dataset)")
        ax.set_ylabel("Speedup (x)")
        ax.legend()

    def plot_test_phase(
        self, ax: plt.Axes = None, dot_format: str = "o", _tune_key: str = "qos"
    ):
        configs = self.best_configs_prefilter
        tested = [conf.test_qos is not None for conf in configs]
        can_plot = all(tested)
        if not ax:
            return can_plot
        assert can_plot
        tune_x, tune_y = self._config_qos_speedups(configs, _tune_key, True, False)
        test_x, test_y = self._config_qos_speedups(configs, "test_qos", True, True)
        ax.plot(tune_x, tune_y, dot_format, label="Tune-set QoS")
        ax.plot(test_x, test_y, dot_format, label="Test-set QoS")
        self._set_xy_limit(ax)
        rthres = self.baseline_tune_qos - self.tune_keep_threshold
        self._draw_qos_line(ax, rthres, f"Relative threshold: {rthres:.2f}")
        ax.set_xlabel("QoS Loss")
        ax.set_ylabel("Speedup (x)")
        ax.legend()

    def _set_xy_limit(self, ax: plt.Axes, show_qos_loss: bool = True):
        xmin, ymin = ax.get_xlim()
        if show_qos_loss:
            ax.set_xlim(xmin=min(0, xmin))
        ax.set_ylim(ymin=min(1, ymin))

    def _config_qos_speedups(
        self,
        configs: List[Config],
        qos_attr: str,
        qos_loss: bool,
        baseline_is_test: bool,
    ):
        def qos_speedup(conf: Config):
            qos = getattr(conf, qos_attr)
            bqos = (
                self.baseline_test_qos if baseline_is_test else self.baseline_tune_qos
            )
            return bqos - qos if qos_loss else qos, conf.speedup

        if not configs:
            return np.zeros((2, 0))
        sorted_points = np.array(
            sorted([qos_speedup(c) for c in configs], key=lambda p: p[0])
        ).T
        return sorted_points

    @staticmethod
    def _draw_qos_line(ax: plt.Axes, qos: float, text: str):
        ymin, ymax = ax.get_ylim()
        ymid = (ymin + ymax) / 2
        ax.axvline(qos)
        ax.annotate(text, (qos, ymid), rotation=90, verticalalignment="center")

    @staticmethod
    def _take_best_configs(configs: List[T], n: Optional[int] = None) -> List[T]:
        points = np.array([(c.qos, c.speedup) for c in configs])
        taken_idx = is_pareto_efficient(points, take_n=n)
        return [configs[i] for i in taken_idx]

    def _test_configs_(self, configs: List[Config]):
        from tqdm import tqdm

        assert self.test_keep_threshold is not None
        if not configs:
            return []
        total_error = 0
        for cfg in tqdm(configs, leave=False):
            assert cfg.test_qos is None
            cfg.test_qos, _ = self.app.measure_qos_cost(cfg.knobs, True)
            msg_logger.debug(f"Test dataset: {cfg.qos:.3f} -> {cfg.test_qos:.3f}")
            total_error += abs(cfg.qos - cfg.test_qos)
        mean_err = total_error / len(configs)
        msg_logger.debug("QoS changed by %f on test dataset (mean abs diff)", mean_err)
        return [cfg for cfg in configs if cfg.test_qos > self.test_keep_threshold]

    def _get_tuner_interface(
        self,
        opentuner_args,
        max_iter: int,
        qos_tuner_threshold: float,
        qos_keep_threshold: Optional[float],
        is_threshold_relative: bool,
        app_kwargs: dict,
    ) -> "TunerInterface":
        # By default, keep_threshold == tuner_threshold
        keep_threshold = qos_keep_threshold or qos_tuner_threshold
        if is_threshold_relative:
            self.baseline_tune_qos, _ = self.app.measure_qos_cost({}, False)
            self.baseline_test_qos, _ = self.app.measure_qos_cost({}, True)
            # Now abs threshold
            qos_tuner_threshold = self.baseline_tune_qos - qos_tuner_threshold
            # These are also abs thresholds
            self.tune_keep_threshold = self.baseline_tune_qos - keep_threshold
            self.test_keep_threshold = self.baseline_test_qos - keep_threshold
            msg_logger.info(
                "Using relative thresholds: baseline QoS = %f (tune set) and %f (test set)",
                self.baseline_tune_qos,
                self.baseline_test_qos,
            )
        else:
            self.tune_keep_threshold = self.test_keep_threshold = keep_threshold
        opentuner_args.test_limit = max_iter
        msg_logger.info(
            "Tuner QoS threshold: %f; keeping configurations with QoS >= %f (tune dataset)",
            qos_tuner_threshold,
            self.tune_keep_threshold,
        )
        return TunerInterface(
            opentuner_args,
            self.app,
            qos_tuner_threshold,
            self.tune_keep_threshold,
            max_iter,
            **app_kwargs,
        )

    @classmethod
    def _get_config_class(cls) -> Type[Config]:
        return Config


def opentuner_default_args():
    from opentuner import default_argparser

    args = default_argparser().parse_args([])
    args.no_dups = True  # Don't print duplicated config warnings
    return args


class TunerInterface(MeasurementInterface):
    def __init__(
        self,
        args,
        app: ApproxApp,
        tuner_thres: float,
        keep_thres: float,
        test_limit: int,
        **app_kwargs,
    ):
        from opentuner.measurement.inputmanager import FixedInputManager
        from opentuner.search.objective import ThresholdAccuracyMinimizeTime
        from tqdm import tqdm

        self.app = app
        self.tune_thres = tuner_thres
        self.keep_thres = keep_thres
        self.pbar = tqdm(total=test_limit, leave=False)
        self.app_kwargs = app_kwargs
        _, self.baseline_cost = app.measure_qos_cost({}, False, **self.app_kwargs)
        msg_logger.debug(f"Baseline cost = {self.baseline_cost}")

        # tune_thres can come in as np.float64 and opentuner doesn't like that
        objective = ThresholdAccuracyMinimizeTime(float(tuner_thres))
        input_manager = FixedInputManager(size=len(self.app.op_knobs))
        super(TunerInterface, self).__init__(
            args,
            program_name=self.app.name,
            input_manager=input_manager,
            objective=objective,
        )

    def set_progress_getter(self, getter: Callable[[], int]):
        self.progress_getter = getter

    def manipulator(self) -> ConfigurationManipulator:
        """Define the search space by creating a ConfigurationManipulator."""
        manipulator = ConfigurationManipulator()
        for op, knobs in self.app.op_knobs.items():
            knob_names = [knob.name for knob in knobs]
            manipulator.add_parameter(EnumParameter(op, knob_names))
        return manipulator

    def run(self, desired_result, input_, limit):
        """Run a given configuration then return cost and QoS."""
        from opentuner.resultsdb.models import Result

        cfg = desired_result.configuration.data
        qos, cost = self.app.measure_qos_cost(cfg, False, **self.app_kwargs)
        # We only want to measure cost in relative.
        # This `cost` is inverse of speedup because opentuner needs a lower-better value.
        cost /= self.baseline_cost
        # Print a debug message for each config in tuning and keep threshold
        self.print_debug_config(qos, cost)
        self.pbar.update(self.progress_getter() - self.pbar.n)
        return Result(time=cost, accuracy=qos)

    def save_final_config(self, config):
        self.pbar.close()

    def print_debug_config(self, qos: float, cost: float):
        gt_tune, gt_keep = qos > self.tune_thres, qos > self.keep_thres
        if not gt_tune and not gt_keep:
            return
        if gt_tune and not gt_keep:
            kind = "tuning"
        elif gt_keep and not gt_tune:
            kind = "keep"
        else:
            kind = "tuning and keep"
        msg_logger.debug(
            f"Found config within {kind} threshold: QoS = {qos}, "
            f"cost = {cost} ({1 / cost} speedup)"
        )
