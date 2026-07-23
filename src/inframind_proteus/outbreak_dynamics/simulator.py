"""Core renewal equation simulator.

Orchestrates one or more vectorised runs of the renewal model by combining:

- A reproduction number model (:class:`~.rt_models.BaseRT` subclass)
- A generation time model (:class:`~.generation_time.BaseGT` subclass)
- A negative-binomial observation (notification) model

Modes
-----
``"calibration"``
    Trajectories are scored against empirical data via WIS.
``"projection"``
    Forward-only run; no data comparison.

Configuration is held in the :class:`SimulationConfig` dataclass (and its
children :class:`TemporalConfig` and :class:`LocationConfig`).
Output is returned as a :class:`SimulationOutput` dataclass.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import NoneType
from typing import Literal, Any

import numba as nb
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from voluptuous import default_factory

from .generation_time import BaseGT, ConstantGammaGT
from .initial_infections import (
    InitialInfectionsConfig,
    build_initial_infec_df,
    parse_initial_infections_config,
)
from .rt_models import BaseRT, LogisticRT, get_rt_model
from .sampling import SamplingConfig, parse_calibration_sampling_config, build_calibration_params_df
from .scoring import nbinom_ppf_cf, wis_score_vectorized, rmse_vectorized, nb_loglikelihood_vectorized, \
    coverages_vectorized
from .utils import parse_timestamp, save_yaml_dict
from .. import BaseConfig


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TemporalConfig(BaseConfig):
    """Temporal settings for a simulation run.

    Attributes
    ----------
    zero_date:
        Reference date; t = 0 in float-based time stamping.
    sim_start:
        Date at which the simulation begins (may differ from ``zero_date``).
    step_dt:
        Duration of each simulation step in days (default: 7 = weekly).
    calibration_start:
        Start of the calibration window (calibration mode only).
    calibration_end:
        End of the calibration window (calibration mode only).
    """

    zero_date: pd.Timestamp
    sim_start: pd.Timestamp
    step_dt: int = 7
    calibration_start: pd.Timestamp | None = None
    calibration_end: pd.Timestamp | None = None


@dataclass
class LocationConfig(BaseConfig):
    """Location identification for a simulation run.

    Attributes
    ----------
    location_id_variable:
        Name of the ID variable as used in mosqlimate data.
        Examples: ``"uf"``, ``"geocode"``, ``"regional_geocode"``,
        ``"macrorregional_geocode"``, ``"uf_code"``.
    location_id:
        Value of the location identifier.
    population_size:
        Population size for the simulated location. Required if using relative
        scaling scheme for the observation model.
    """

    location_id_variable: str
    location_id: str | int
    population_size: int = int(1E5)


@dataclass
class ObservationModelConfig(BaseConfig):
    """Observation model configuration.

    Attributes
    ----------
    model:
        Name of the observation model. Currently only ``"negative_binomial"``
        is supported.
    params:
        Model-specific parameters. For negative-binomial model, this includes:
        - ``notif_nb_overdispersion``: Overdispersion parameter
        - ``notif_scaling_factor``: Scaling factor from infections to cases
        - ``notif_relative_scale``: Optional relative scaling factor
    reference_population_size:
        Denominator for normalizing incidences per population size. Default
        is 100k (1E5), only change this if there is a clear reason.
    """

    model: str = "negative_binomial"
    params: dict[str, Any] = field(default_factory=dict)
    reference_population_size: int = int(1E5)


@dataclass
class ScoringConfig(BaseConfig):
    """Scoring configuration for calibration mode.

    Attributes
    ----------
    case_beam_quantiles:
        Quantiles used to build the deterministic case prediction beam.
    """

    metrics: list[str] = field(
        default_factory=lambda: ["wis", "rmse", "coverages", "nb_loglikelihood"]
    )
    case_beam_quantiles: list[float] = field(
        default_factory=lambda: [0.025, 0.25, 0.5, 0.75, 0.975]
    )

@dataclass
class OutputConfig(BaseConfig):
    """Output/export configuration.

    Attributes
    ----------
    main_dir:
        Main output directory under which all results will be saved.
    keep_rt_trajectories:
        Whether to keep the full R(t) trajectories for all simulations
        in the output data structure. Uses more memory if true.
    """

    main_dir: Path = Path("outputs/_DEFAULT")

    keep_rt_trajectories: bool = False


@dataclass
class SimulationConfig(BaseConfig):
    """Top-level configuration for the renewal simulator.

    Attributes
    ----------
    mode:
        ``"calibration"`` — compare trajectories against empirical data;
        ``"projection"`` — run forward with no data comparison.
    num_simulations:
        Number of parallel simulation trajectories.
    num_time_steps:
        Number of simulation time steps (excluding the warm-up window).
    gt_max:
        Maximum generation time in days (determines warm-up window size).
    temporal:
        Temporal settings (:class:`TemporalConfig`).
    location:
        Location settings (:class:`LocationConfig`), including population size.
    observation_model:
        Observation model configuration (:class:`ObservationModelConfig`).
    scoring:
        Scoring configuration (:class:`ScoringConfig`).
    sampling:
        Optional sampling settings parsed from ``sampling`` plus fixed
        parameter dictionaries used to build calibration ``params_df``.
    initial_infections:
        Initial infection seeding configuration.
    rng_seed:
        Global RNG seed for the observation model sampling.
    """

    mode: Literal["calibration", "projection"] = "projection"
    num_simulations: int = 1000
    num_time_steps: int = 50
    gt_max: int = 49  # days
    temporal: TemporalConfig = field(
        default_factory=lambda: TemporalConfig(
            zero_date=pd.Timestamp("2023-10-01"),
            sim_start=pd.Timestamp("2023-10-01"),
        )
    )
    location: LocationConfig = field(
        default_factory=lambda: LocationConfig(
            location_id_variable="uf",
            location_id="DF",
        )
    )
    observation_model: ObservationModelConfig = field(
        default_factory=lambda: ObservationModelConfig(
            model="negative_binomial",
            params={
                "notif_nb_overdispersion": 10.0,
                "notif_scaling_factor": 1.0,
            },
        )
    )
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    sampling: SamplingConfig | None = None
    initial_infections: InitialInfectionsConfig = field(
        default_factory=InitialInfectionsConfig
    )
    output: OutputConfig = field(default_factory=OutputConfig)
    rng_seed: int = 0


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimulationScoring:
    """
    Attributes
    ----------
    wis_array:
        Per-simulation WIS scores over the calibration window.  ``None`` in
        projection mode.  Shape ``(num_simulations, n_cal)`` where ``n_cal``
        is the number of observation timestamps that fall within
        ``[calibration_start, calibration_end]``.
    summary:
        Summary scores for all simulations, with one scalar for each score
        and for each simulation.
        Data frame shape is ``(num_simulations, num_scores)``,
        where ``num_scores`` is the number of calculated scores.
    """
    # wis_array: np.ndarray  # Deprecated. Removed to save RAM.
    summary: pd.DataFrame

    @classmethod
    def concat(
            cls, objs: list[SimulationScoring]
    ) -> SimulationScoring:
        """Concatenate multiple SimulationScoring instances into one, joining
        through all simulations.

        """
        return cls(
            # wis_array=np.concatenate([o.wis_array for o in objs], axis=0),
            summary=pd.concat([o.summary for o in objs], axis=0),
        )


@dataclass
class SimulationOutput:
    """Outputs from a simulation run.

    Attributes
    ----------
    rt_df:
        Reproduction number trajectories for all simulations.
        ``None`` if ``config.output.keep_rt_trajectories`` is ``False``.
    infec_df:
        Abstract infection counts.
        Index = ``i_simulation``, columns = time step values in days.
    mean_cases_df:
        Expectancy of reported case counts.
        Same shape as ``infec_df``.
    case_beam_df:
        Deterministic quantile case beam.
        MultiIndex ``(quantile, i_simulation)``, same time-step columns.
    scoring:
        Scoring results for calibration mode; ``None`` in projection mode.
    config:
        The :class:`SimulationConfig` used to produce these outputs.
    """
    infec_df: pd.DataFrame
    mean_cases_df: pd.DataFrame
    case_beam_df: pd.DataFrame
    scoring: SimulationScoring | None
    config: SimulationConfig
    rt_df: pd.DataFrame | None = None

    @classmethod
    def concat(
            cls,
            objs: list[SimulationOutput],
    ) -> SimulationOutput:
        """Concatenate multiple SimulationOutput instances into one.

        This is useful for combining results from sequential chunk runs.

        NOTE: This method does not modify the simulation indices. This means
        that if the input SimulationOutput instances have overlapping simulation indices,
        the resulting concatenated SimulationOutput will also do, possibly leading
        to errors later.

        Returns
        -------
        SimulationOutput
            A new instance with concatenated data frames and arrays.
        """
        rt_dfs = [o.rt_df for o in objs if o.rt_df is not None]
        scoring_objs = [o.scoring for o in objs if o.scoring is not None]
        return cls(
            infec_df=pd.concat([o.infec_df for o in objs], axis=0),
            mean_cases_df=pd.concat([o.mean_cases_df for o in objs], axis=0),
            case_beam_df=pd.concat([o.case_beam_df for o in objs], axis=0),
            scoring=SimulationScoring.concat(scoring_objs) if scoring_objs else None,
            rt_df=pd.concat(rt_dfs, axis=0) if rt_dfs else None,
            config=objs[0].config,  # Assumes all outputs share the same config
        )

    def save(
            self,
            subdir_name="simulation",
            root_dir: Path | str = Path("."),
    ):
        """Save/export all eligible data structures to files.

        Follows the directives in config.output
        """
        cfg = self.config
        out_cfg = cfg.output
        root_dir = Path(root_dir)

        # TODO: Continue this function, formalize exporting procedures.
        raise NotImplementedError

        out_main_dir = out_cfg.main_dir
        out_dir = out_main_dir / subdir_name

        # out_dir = _root / Path(config_dict["output"]["main_dir"]) / "calibration_results" / f"{location_id}_{year}"
        out_dir.mkdir(exist_ok=True, parents=True)

        # save_yaml_dict(config_dict, out_dir / "config.yaml")

        # params_df.to_csv(out_dir / "params.csv.gz")  # Also kinda heavy!

        # results.scoring.summary.to_parquet(out_dir / "scoring.parquet")
        self.scoring.summary.to_csv(out_dir / "scoring.csv.gz")

        # results.case_beam_df.to_csv(out_dir / "case_beam_df.csv")  # TOOOOOOOOO heavy!

        # # Export only selected trajectories (case beams) to save space
        # self.case_beam_df.reset_index().set_index("i_simulation").loc[selected_wis_sr.index].to_csv(
        #     out_dir / "case_beam_selected_df.csv.gz")
        # print(f"Done: {out_dir}")




# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def _validate_population_size(population_size):
    if population_size is None:
        raise ValueError(
            "population_size must be provided when using "
            "notif_relative_scale"
        )
    if population_size < 0:
        raise ValueError(f"population_size must be non-negative. Got {population_size}.")


class RenewalSimulator:
    """Vectorised renewal equation simulator.

    Parameters
    ----------
    rt_model:
        An instance of a :class:`~.rt_models.BaseRT` subclass.
    gt_model:
        An instance of a :class:`~.generation_time.BaseGT` subclass.
    config:
        A :class:`SimulationConfig` instance.
    """

    def __init__(
        self,
        rt_model: BaseRT,
        gt_model: BaseGT,
        config: SimulationConfig,
    ) -> None:
        self.rt_model = rt_model
        self.gt_model = gt_model
        self.config = config

        self._step_dt = config.temporal.step_dt
        self._gt_max_steps = int(np.ceil(config.gt_max / self._step_dt))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        params_df: pd.DataFrame,
        initial_infec_df: pd.DataFrame,
        observations_sr: pd.Series | None = None,
    ) -> SimulationOutput:
        """Run the simulator.

        Parameters
        ----------
        params_df:
            Parameter table.  One row per simulation.  Must contain all
            columns required by ``rt_model``, plus
            ``notif_nb_overdispersion`` and ``notif_scaling_factor`` when
            those should vary across simulations. May also contain the optional
            ``notif_relative_scale`` column for relative scaling.
        initial_infec_df:
            Seed infection values for the warm-up window.
            Shape ``(num_simulations, warmup_steps)``.
        observations_sr:
            Observed case counts indexed by time step.  Required in
            calibration mode; ignored in projection mode.

        Returns
        -------
        SimulationOutput
        """
        cfg = self.config
        # num_sim = cfg.num_simulations  # OLD.
        num_sim = params_df.shape[0]  # Infer from params_df
        num_steps = cfg.num_time_steps
        gt_steps = self._gt_max_steps
        initial_steps = initial_infec_df.shape[1]
        step_dt = self._step_dt

        # ------------------------------------------------------------------
        # 1. Validate inputs
        # ------------------------------------------------------------------
        if params_df.shape[0] != num_sim:
            raise ValueError(
                f"params_df has {params_df.shape[0]} rows; expected {num_sim}"
            )
        # if num_sim != cfg.num_simulations:
        #     warnings.warn(
        #         "Number of rows in `params_df` does not match "
        #         "config.num_simulations. Will run the number on params_df."
        #     )
        if initial_infec_df.shape[0] != num_sim:
            raise ValueError(
                f"initial_infec_df has {initial_infec_df.shape[0]} rows; "
                f"expected {num_sim}"
            )
        if initial_infec_df.shape[1] < gt_steps:
            raise ValueError(
                f"initial_infec_df must have at least {gt_steps} columns; "
                f"got {initial_infec_df.shape[1]}"
            )
        if initial_steps != cfg.initial_infections.num_steps:
            raise ValueError(
                f"initial_infec_df has {initial_infec_df.shape[1]} columns, "
                f"but config.initial_infections.num_steps is "
                f"{cfg.initial_infections.num_steps}. Unless you are providing "
                f"a custom initial_infec_df (remove this error if so), these "
                f"should match for the model to be consistent."
            )

        # if not params_df.index.isin(initial_infec_df.index).all():  # contains
        if not (initial_infec_df.index == params_df.index).all():  # exact match
            raise ValueError(
                "Indices of initial_infec_df and params_df must match exactly"
            )
        if cfg.mode == "calibration" and observations_sr is None:
            raise ValueError("observations_sr is required in calibration mode")
        if cfg.mode == "calibration" and (
            cfg.temporal.calibration_start is None
            or cfg.temporal.calibration_end is None
        ):
            raise ValueError(
                "TemporalConfig.calibration_start and calibration_end must both be "
                "set when mode='calibration'"
            )

        _params: pd.DataFrame = params_df.copy()

        # Fill config-default observation params when not in params_df
        # NOTE: Disabled since we want to enforce presence of these parameters.
        # if "notif_nb_overdispersion" not in _params.columns:
        #     _params["notif_nb_overdispersion"] = cfg.notif_nb_overdispersion
        # if "notif_scaling_factor" not in _params.columns:
        #     _params["notif_scaling_factor"] = cfg.notif_scaling_factor

        self.rt_model.validate_params(_params)

        # ------------------------------------------------------------------
        # 2. GT PMF — shape (num_steps, gt_steps), reversed-lag convention
        # ------------------------------------------------------------------
        gt_pmf = self.gt_model.get_pmf(
            gt_max_steps=gt_steps,
            num_time_steps=num_steps,
            step_dt=step_dt,
        )

        # ------------------------------------------------------------------
        # 3. R(t) — shape (num_sim, gt_steps + num_steps)
        #
        # The RT time grid is in days from zero_date.  The warm-up window
        # starts at (sim_start − initial_steps × step_dt) days from zero_date,
        # so R(t) parameters such as rt_logist_start are expressed as days
        # from zero_date independently of the warm-up size.
        # ------------------------------------------------------------------
        sim_start_day = float((cfg.temporal.sim_start - cfg.temporal.zero_date).days)
        # initial_steps = cfg.initial_infections.num_steps
        t_start = sim_start_day - initial_steps * step_dt

        rt_vec = self.rt_model.generate(
            params_df=_params,
            num_time_steps=initial_steps + num_steps,
            step_dt=step_dt,
            t_start=t_start,
        )

        # ------------------------------------------------------------------
        # 4. Assemble infection array (warm-up pre-filled, rest zero)
        # ------------------------------------------------------------------
        infec_vec = np.concatenate(
            [
                initial_infec_df.to_numpy(dtype=float),
                np.zeros((num_sim, num_steps), dtype=float),
            ],
            axis=1,
        )

        # ------------------------------------------------------------------
        # 5. Core renewal loop
        # ------------------------------------------------------------------
        infec_vec = self._run_renewal_loop(
            infec_vec, rt_vec, gt_pmf, gt_steps, initial_steps, num_steps
        )

        # ------------------------------------------------------------------
        # 6. Observation model (crop warm-up first)
        # ------------------------------------------------------------------
        rng = np.random.default_rng(cfg.rng_seed)
        infec_sim = infec_vec[:, initial_steps:]  # (num_sim, num_steps)

        mean_cases_vec, case_beam_df = self._apply_observation_model(
            infec_sim, _params, rng,
            population_size=cfg.location.population_size,
            reference_population_size=cfg.observation_model.reference_population_size,
        )

        # ------------------------------------------------------------------
        # 7. Assign timestamp columns
        # ------------------------------------------------------------------
        sim_timestamps = pd.date_range(
            start=cfg.temporal.sim_start,
            periods=num_steps,
            freq=pd.tseries.offsets.Day(step_dt),
        )

        infec_df = pd.DataFrame(infec_sim, columns=sim_timestamps)
        # infec_df.index.name = "i_simulation"
        infec_df.index = _params.index
        infec_df.columns.name = "t"

        mean_cases_df = pd.DataFrame(mean_cases_vec, columns=sim_timestamps)
        # mean_cases_df.index.name = "i_simulation"
        mean_cases_df.index = _params.index
        mean_cases_df.columns.name = "t"

        case_beam_df.columns = sim_timestamps
        case_beam_df.columns.name = "t"

        # ------------------------------------------------------------------
        # 8. Scoring (calibration mode only)
        # ------------------------------------------------------------------
        wis_array = None
        if cfg.mode == "calibration":
            observations_sr: pd.Series
            scoring = self.score_simulations(
                cfg, case_beam_df, observations_sr, params_df
            )

        else:
            scoring = None

        # Etc
        # =============
        if cfg.output.keep_rt_trajectories:
            # Reproduce the time grid from the RT logistic model
            rt_datetime_grid = pd.date_range(
                start=(cfg.temporal.sim_start - pd.Timedelta(initial_steps * step_dt, unit="D")),
                periods=initial_steps + num_steps,
                freq=pd.Timedelta(step_dt, unit="D"),
                name="date",
            )
            rt_df = pd.DataFrame(
                rt_vec,
                index=_params.index,
                columns=rt_datetime_grid,
            )
        else:
            rt_df = None


        return SimulationOutput(
            infec_df=infec_df,
            mean_cases_df=mean_cases_df,
            case_beam_df=case_beam_df,
            scoring=scoring,
            config=cfg,
            rt_df=rt_df,
        )

    def run_sequential_chunks(
            self,
            params_df: pd.DataFrame,
            initial_infec_df: pd.DataFrame,
            observations_sr: pd.Series | None = None,
            max_chunk_size: int = 10000,
    ):
        """"""

        _iter_factory = lambda: range(0, params_df.shape[0], max_chunk_size)

        params_df_chunks = [
            params_df.iloc[i:i + max_chunk_size]
            for i in _iter_factory()
        ]
        initial_infec_df_chunks = [
            initial_infec_df.iloc[i:i + max_chunk_size]
            for i in _iter_factory()
        ]

        results: list[SimulationOutput] = list()
        for params_chunk, infec_chunk in zip(params_df_chunks, initial_infec_df_chunks):
            results.append(
                self.run(
                    params_df=params_chunk,
                    initial_infec_df=infec_chunk,
                    observations_sr=observations_sr,
                )
            )

        return SimulationOutput.concat(results)

    def build_initial_infec_df(self) -> pd.DataFrame:
        """Build the warm-up infection matrix using the config settings."""
        return build_initial_infec_df(
            num_simulations=self.config.num_simulations,
            gt_max_steps=self._gt_max_steps,
            step_dt=self._step_dt,
            initial_config=self.config.initial_infections,
        )

    def build_simulation_data(
            self,
            config: SimulationConfig = None,
            sampling_kwargs: dict | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build auxiliary data frames for simulations.

        This method replaces commonly used code for setting up
        a simulation.
        """
        # ---
        config = config or self.config
        sampling_kwargs = sampling_kwargs or dict()

        # Data frame with all model parameters
        params_df = build_calibration_params_df(
            config.num_simulations, config.sampling, **sampling_kwargs
        )
        # Re-assign num_simulations, since prev. step may change it
        config.num_simulations = params_df.shape[0]

        # Initial infections for the warm-up window
        initial_infec_df = build_initial_infec_df(
            config.num_simulations,
            0,  # No longer in use
            config.temporal.step_dt,
            config.initial_infections
        )

        return params_df, initial_infec_df

    @staticmethod
    def _run_renewal_loop(
        infec_vec: np.ndarray,
        rt_vec: np.ndarray,
        gt_pmf: np.ndarray,
        gt_max_steps: int,
        num_initial_steps: int,
        num_time_steps: int,
    ) -> np.ndarray:
        """Core renewal equation time loop (numba-compatible structure).

        Advances ``infec_vec`` in-place through ``num_time_steps`` steps.

        Parameters
        ----------
        infec_vec:
            Full infection array of shape
            ``(num_simulations, gt_max_steps + num_time_steps)``.
            The first ``gt_max_steps`` columns are pre-filled (warm-up).
        rt_vec:
            R(t) array of the same shape as ``infec_vec``.
        gt_pmf:
            Generation time PMF of shape ``(num_time_steps, gt_max_steps)``.
            Axis 1 follows the *reversed-lag* convention: index 0 is the
            largest lag (oldest), index ``-1`` is lag 1 (most recent step).
            This ordering aligns directly with the look-back window slices
            so no further reversal is needed inside the loop.
        gt_max_steps:
            Maximum generation time in number of steps.
        num_initial_steps:
            Size of the warm-up / look-back window, which are skipped at the
            simulation loop.
        num_time_steps:
            Number of steps to advance.

        Returns
        -------
        np.ndarray
            Updated ``infec_vec`` (modified in-place and returned).

        Notes
        -----
        This method intentionally avoids pandas objects and Python-level
        data structures so that a future ``@numba.njit`` decoration requires
        only minimal changes.

        Renewal equation at simulation step ``i`` (0-based)::

            I(t_i) = Σ_s  R(t_{i-s}) · I(t_{i-s}) · w_i(s)

        where the sum runs over ``s = 1..gt_max_steps`` (the look-back
        window), and ``w_i`` is the GT PMF row for step ``i``.
        """
        # Delegate to external function that can be numba-compiled
        return _run_renewal_loop_numba(
            infec_vec=infec_vec,
            rt_vec=rt_vec,
            gt_pmf=gt_pmf,
            gt_max_steps=gt_max_steps,
            num_initial_steps=num_initial_steps,
            num_time_steps=num_time_steps,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_observation_model(
        self,
        infec_vec: np.ndarray,
        params_df: pd.DataFrame,
        rng: np.random.Generator,
        population_size: int | None = None,
        reference_population_size: int = int(1E5)
    ) -> tuple[np.ndarray, pd.DataFrame]:
    # ) -> pd.DataFrame:
        """Apply the negative-binomial observation (notification) model.

        Parameters
        ----------
        infec_vec:
            Abstract infection counts.
            Shape ``(num_simulations, num_time_steps)`` (warm-up excluded).
        params_df:
            Parameter table with ``notif_nb_overdispersion`` and
            ``notif_scaling_factor`` columns. Optionally may contain
            ``notif_relative_scale`` for population-relative scaling, in which
            case ``notif_scaling_factor`` is overridden by the internal
            algorithm.
        rng:
            NumPy random generator.

        Returns
        -------
        mean_cases_vec : np.ndarray
            Expectancy of the number of cases.
            Shape ``(num_simulations, num_time_steps)``.
            dtype int64.
        case_beam_df : pd.DataFrame
            Deterministic case beam.
            MultiIndex ``(quantile, i_simulation)``, integer columns
            ``0..num_time_steps-1`` (renamed to timestamps by the caller).
        """
        required_cols = ["notif_nb_overdispersion", "notif_scaling_factor"]
        missing_cols = [c for c in required_cols if c not in params_df.columns]
        if missing_cols:
            raise ValueError(
                "_apply_observation_model requires params_df columns "
                f"{required_cols}; missing: {missing_cols}"
            )


        overdisp = params_df["notif_nb_overdispersion"].to_numpy()[:, np.newaxis]
        # Get alternative scaling scheme
        if "notif_relative_scale" in params_df.columns:
            # Population-relative scaling factor
            _validate_population_size(population_size)
            scale_f = (
                    params_df["notif_relative_scale"].to_numpy()[:, np.newaxis]
                    * population_size
                    / reference_population_size
            )
        else:
            # Directly provided scaling factor
            scale_f = params_df["notif_scaling_factor"].to_numpy()[:, np.newaxis]

        # Expected reported cases; clip to avoid negative expectancies
        expectancy: np.ndarray = np.clip(infec_vec * scale_f, 0.0, None)

        if np.isnan(expectancy).any():
            expectancy[np.isnan(expectancy)] = 0.
            print("WARNING: NaN values found in expectancy; replaced with 0.")

        # NB success probability: p = n / (n + μ)
        p = overdisp / (overdisp + expectancy)

        # # Stochastic sample (integer counts) (DISABLED)
        # cases_vec = rng.negative_binomial(n=overdisp, p=p)
        # Store the mean for future trajectory sampling
        mean_cases_vec = expectancy

        # Deterministic quantile beam via Cornish-Fisher approximation
        beam_frames = [
            pd.DataFrame(
                nbinom_ppf_cf(q=q, n=overdisp, p=p, continuity=False),
                index=params_df.index,
            )
            for q in self.config.scoring.case_beam_quantiles
        ]

        case_beam_df = pd.concat(
            beam_frames,
            keys=self.config.scoring.case_beam_quantiles,
            names=["quantile", "i_simulation"],
        )

        # Mean cases, allowing to sample trajectories outside this function
        mean_cases_vec = expectancy

        # DEBUG - Case beam quantile anomalies
        # r = case_beam_df.xs(0.5, level="quantile") / mean_cases_vec
        r = case_beam_df.xs(0.75, level="quantile") / mean_cases_vec

        # return cases_vec, case_beam_df
        return mean_cases_vec, case_beam_df

    @staticmethod
    def score_simulations(
            cfg: SimulationConfig,
            case_beam_df: DataFrame,
            observations_sr: Series,
            params_df: pd.DataFrame,
    ) -> SimulationScoring:
        """Score simulated trajectories against observations via WIS.

        Computes per-simulation score metrics over the
        declared calibration window using the deterministic case beam
        quantiles produced by the observation model.

        Parameters
        ----------
        case_beam_df:
            Deterministic case prediction beam with MultiIndex
            ``(quantile, i_simulation)`` and timestamp columns.
        cfg:
            Simulation configuration. Uses
            ``cfg.temporal.calibration_start`` and
            ``cfg.temporal.calibration_end`` to define the scoring window.
        observations_sr:
            Observed case counts indexed by timestamp.
        params_df:
            Data frame with parameters for the simulations, one per row.

        Returns
        -------
        SimulationScoring
            Scoring container with ``wis_array`` of shape
            ``(num_simulations, n_cal)``, where ``n_cal`` is the number of
            observation timestamps inside the calibration window.

        Raises
        ------
        ValueError
            If no observations fall within the calibration window.
        ValueError
            If any calibration timestamp in ``observations_sr`` is absent
            from the simulation timestamps in ``case_beam_df``.
        """
        cal_start = cfg.temporal.calibration_start
        cal_end = cfg.temporal.calibration_end

        # Slice observations to the declared calibration window
        obs_cal = observations_sr.loc[
            (observations_sr.index >= cal_start)
            & (observations_sr.index <= cal_end)
            ]
        if obs_cal.empty:
            raise ValueError(
                f"No observation data within the calibration window "
                f"[{cal_start.date()}, {cal_end.date()}]"
            )

        # Every calibration timestamp must align exactly with a simulation step
        missing = obs_cal.index.difference(case_beam_df.columns)
        if not missing.empty:
            raise ValueError(
                f"{len(missing)} calibration timestamp(s) are absent from the "
                f"simulation period. Missing: {missing[:5].tolist()}"
            )

        # ========

        simulations_df = case_beam_df[obs_cal.index]
        simulations_median_df = simulations_df.xs(0.5, level="quantile")
        summary_df = pd.DataFrame(
            {},
            # index=simulations_df.index.get_level_values("i_simulation").unique(),
            index=params_df.index,
        )
        # ^ Shape: summary_df[i_simulation, score_name] = summary score scalar value
        # Expects that `simulations_df` is sorted by `i_simulation`.
        # Warning: Resulting WIS values may be randomized if this is not satisfied.

        # ====

        # Weighted Interval Score (WIS)
        if "wis" in cfg.scoring.metrics:
            # Weighted Interval Scores (WIS)
            wis_array = wis_score_vectorized(
                simulations_df=simulations_df,
                observations_sr=obs_cal,
            )
            summary_df["wis"] = wis_array.sum(axis=1)

        # Root Mean Squared Error - individual components
        if "rmse" in cfg.scoring.metrics:
            rmse_array = rmse_vectorized(
                simulations_df=simulations_median_df,
                observations_sr=observations_sr,
            )
            summary_df["rmse"] = rmse_array

        # Negative-binomial loglikelihood
        if "nb_loglikelihood" in cfg.scoring.metrics:
            summary_df["nb_loglikelihood"] = nb_loglikelihood_vectorized(
                simulations_df=simulations_median_df,
                observations_sr=observations_sr,
                overdisp=params_df["notif_nb_overdispersion"].to_numpy(),
            )

        # Coverages of selected prediction intervals
        if "coverages" in cfg.scoring.metrics:
            coverages_df = coverages_vectorized(
                simulations_df=simulations_df,
                observations_sr=obs_cal,
            )
            summary_df = pd.concat([summary_df, coverages_df], axis=1)

        scoring = SimulationScoring(
            # wis_array=wis_array,
            summary=summary_df,
        )

        return scoring

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config_dict(
        cls,
        config_dict: dict,
    ) -> "RenewalSimulator":
        """Construct a simulator from a raw config dictionary (e.g. from YAML).

        Parameters
        ----------
        config_dict:
            Parsed YAML dictionary.

        Returns
        -------
        RenewalSimulator
        """
        defaults = SimulationConfig()

        sim_cfg = config_dict.get("simulation", {}) or {}
        temporal_cfg = config_dict.get("temporal", {}) or {}
        location_cfg = config_dict.get("location", {}) or {}
        scoring_cfg = config_dict.get("scoring", {}) or {}
        rt_cfg = config_dict.get("reproduction_number", {}) or {}
        gt_cfg = config_dict.get("generation_time", {}) or {}
        sampling_cfg = parse_calibration_sampling_config(config_dict)
        initial_infections_cfg = parse_initial_infections_config(config_dict)
        output_cfg = config_dict.get("output", {}) or {}

        rt_model_name = str(rt_cfg.get("model", "logistic")).strip().lower()
        rt_model = get_rt_model(rt_model_name)

        gt_model_name = str(gt_cfg.get("model", "constant_gamma")).strip().lower()
        gt_params = gt_cfg.get("params", {}) or {}
        if gt_model_name == "constant_gamma":
            gt_model = ConstantGammaGT(
                shape=float(gt_params.get("shape", 10.0)),
                scale=float(gt_params.get("scale", 1.8)),
            )
        else:
            raise ValueError(
                "Unsupported generation_time.model: "
                f"{gt_cfg.get('model')!r}. Supported models: ['constant_gamma']"
            )

        mode = sim_cfg.get("mode", defaults.mode)
        if mode not in ("calibration", "projection"):
            raise ValueError(
                f"simulation.mode must be 'calibration' or 'projection'; got {mode!r}"
            )

        def _to_timestamp(
            value: str | int | float | pd.Timestamp,
            zero_date: pd.Timestamp | None = None,
        ) -> pd.Timestamp:
            if isinstance(value, pd.Timestamp):
                return value
            return parse_timestamp(value, zero_date=zero_date)

        zero_date = _to_timestamp(
            temporal_cfg.get("zero_date", defaults.temporal.zero_date)
        )
        sim_start = _to_timestamp(
            temporal_cfg.get("sim_start", defaults.temporal.sim_start),
            zero_date=zero_date,
        )

        calibration_start_raw = temporal_cfg.get("calibration_start")
        calibration_end_raw = temporal_cfg.get("calibration_end")
        calibration_start = (
            _to_timestamp(calibration_start_raw, zero_date=zero_date)
            if calibration_start_raw is not None
            else defaults.temporal.calibration_start
        )
        calibration_end = (
            _to_timestamp(calibration_end_raw, zero_date=zero_date)
            if calibration_end_raw is not None
            else defaults.temporal.calibration_end
        )

        # Parse observation model configuration
        obs_model_cfg_dict = config_dict.get("observation_model", {}) or {}
        obs_model_name = obs_model_cfg_dict.get("model", defaults.observation_model.model)
        obs_model_params = obs_model_cfg_dict.get("params", {}) or {}
        
        # Convert params to float dict, using defaults when params is empty
        if obs_model_params:
            obs_params_parsed = {
                str(k): float(v) for k, v in obs_model_params.items()
            }
        else:
            # Use defaults when no params provided
            obs_params_parsed = defaults.observation_model.params.copy()
        
        # Get reference_population_size (currently not in YAML, using default)
        reference_population_size = int(
            obs_model_cfg_dict.get(
                "reference_population_size",
                defaults.observation_model.reference_population_size
            )
        )

        # Parse scoring configuration
        case_beam_quantiles = [
            float(q)
            for q in scoring_cfg.get(
                "case_beam_quantiles",
                defaults.scoring.case_beam_quantiles,
            )
        ]
        scoring_metrics = scoring_cfg.get(
            "metrics",
            defaults.scoring.metrics
        )

        # Parse population_size from location section (with fallback to simulation for transition)
        population_size = int(
            location_cfg.get(
                "population_size",
                sim_cfg.get("population_size", defaults.location.population_size)
            )
        )

        # Output config
        out_main_dir = Path(output_cfg.get("main_dir", defaults.output.main_dir))
        keep_rt_trajectories = output_cfg.get("keep_rt_trajectories", defaults.output.keep_rt_trajectories)

        config = SimulationConfig(
            mode=mode,
            num_simulations=int(sim_cfg.get("num_simulations", defaults.num_simulations)),
            num_time_steps=int(sim_cfg.get("num_time_steps", defaults.num_time_steps)),
            gt_max=int(sim_cfg.get("gt_max", defaults.gt_max)),
            temporal=TemporalConfig(
                zero_date=zero_date,
                sim_start=sim_start,
                step_dt=int(temporal_cfg.get("step_dt", defaults.temporal.step_dt)),
                calibration_start=calibration_start,
                calibration_end=calibration_end,
            ),
            location=LocationConfig(
                location_id_variable=location_cfg.get(
                    "location_id_variable", defaults.location.location_id_variable
                ),
                location_id=location_cfg.get("location_id", defaults.location.location_id),
                population_size=population_size,
            ),
            observation_model=ObservationModelConfig(
                model=obs_model_name,
                params=obs_params_parsed,
                reference_population_size=reference_population_size,
            ),
            scoring=ScoringConfig(
                metrics=scoring_metrics,
                case_beam_quantiles=case_beam_quantiles,
            ),
            sampling=sampling_cfg,
            output=OutputConfig(
                main_dir=out_main_dir,
                keep_rt_trajectories=keep_rt_trajectories,
            ),
            initial_infections=initial_infections_cfg,
            rng_seed=int(sim_cfg.get("rng_seed", defaults.rng_seed)),
        )

        return cls(rt_model=rt_model, gt_model=gt_model, config=config)


def sample_negative_binomial_trajectories(
        expectancy: np.ndarray,
        overdisp: np.ndarray,
        rng: np.random.Generator,
) -> np.ndarray:
    """Apply the negative-binomial observation model to infection counts.

    This samples actual numbers of cases for each time and each abstract
    infection trajectory, rather than specifying prediction intervals.

    Parameters
    ----------
    expectancy: np.ndarray
        Expected number of cases (mean of the negative binomial) at each
        time (column index) for each trajectory (row index).
        Expected shape: (num_simulations, num_time_steps).
    overdisp: np.ndarray
        Overdispersion parameter of the negative binomial for each trajectory.
        Expected shape: (num_simulations,).
    rng: np.random.Generator
        A pre-initialized NumPy random generator, or data to initialize it.
    """
    # Prep work
    # ---------
    # -()- Strict shape checks. Could be made more flexible (e.g. array and scalar)
    if expectancy.ndim != 2:
        raise ValueError(f"expectancy must be 2D; got shape {expectancy.shape}")
    if overdisp.ndim != 1:
        raise ValueError(f"overdisp must be 1D; got shape {overdisp.shape}")
    if expectancy.shape[0] != overdisp.shape[0]:
        raise ValueError(
            f"expectancy.shape[0] ({expectancy.shape[0]}) must match "
            f"overdisp.shape[0] ({overdisp.shape[0]})"
        )

    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    # -------
    _expectancy = expectancy
    _overdisp = overdisp[:, np.newaxis]
    p = _overdisp / (_overdisp + _expectancy)

    cases_vec: np.ndarray = rng.negative_binomial(
        n=_overdisp, p=p, size=_expectancy.shape
    )

    return cases_vec

_nb_readonly_arr = nb.types.Array(nb.types.float64, 2, 'A', readonly=True)
@nb.njit(
    nb.float64[:,:](
        nb.float64[:,:],
        nb.float64[:,:],
        _nb_readonly_arr,
        nb.int64,
        nb.int64,
        nb.int64,
    ),
)
def _run_renewal_loop_numba(
    infec_vec: np.ndarray,
    rt_vec: np.ndarray,
    gt_pmf: np.ndarray,
    gt_max_steps: int,
    num_initial_steps: int,
    num_time_steps: int,
) -> np.ndarray:
    """Core renewal equation time loop (numba-compatible structure).

    Advances ``infec_vec`` in-place through ``num_time_steps`` steps.

    Parameters
    ----------
    infec_vec:
        Full infection array of shape
        ``(num_simulations, gt_max_steps + num_time_steps)``.
        The first ``gt_max_steps`` columns are pre-filled (warm-up).
    rt_vec:
        R(t) array of the same shape as ``infec_vec``.
    gt_pmf:
        Generation time PMF of shape ``(num_time_steps, gt_max_steps)``.
        Axis 1 follows the *reversed-lag* convention: index 0 is the
        largest lag (oldest), index ``-1`` is lag 1 (most recent step).
        This ordering aligns directly with the look-back window slices
        so no further reversal is needed inside the loop.
    gt_max_steps:
        Maximum generation time in number of steps.
    num_initial_steps:
        Size of the warm-up / look-back window, which are skipped at the
        simulation loop.
    num_time_steps:
        Number of steps to advance.

    Returns
    -------
    np.ndarray
        Updated ``infec_vec`` (modified in-place and returned).

    Notes
    -----
    This method intentionally avoids pandas objects and Python-level
    data structures so that a future ``@numba.njit`` decoration requires
    only minimal changes.

    Renewal equation at simulation step ``i`` (0-based)::

        I(t_i) = Σ_s  R(t_{i-s}) · I(t_{i-s}) · w_i(s)

    where the sum runs over ``s = 1..gt_max_steps`` (the look-back
    window), and ``w_i`` is the GT PMF row for step ``i``.
    """
    for i_sim_step in range(num_time_steps):
        i_full = num_initial_steps + i_sim_step
        # Shape of each slice: (num_simulations, gt_max_steps)
        infec_vec[:, i_full] = np.sum(
            rt_vec[:, i_full - gt_max_steps : i_full]
            * infec_vec[:, i_full - gt_max_steps : i_full]
            * gt_pmf[i_sim_step],
            axis=1,
        )
    return infec_vec
