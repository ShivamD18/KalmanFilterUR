"""
The 'less uncertainty' figure, shared by the live dashboard and the offline demo.

FusionRecorder collects ground truth, raw readings and filter estimates and
scores them against each other. FusionFigure draws four panels from it:

  1. distance over time: truth, raw readings, fused estimate with its 2-sigma band
  2. the 1-sigma uncertainty each filter reports, over time
  3. error vs distance: single-reading noise (model lines) vs filter RMSE (dots)
  4. a scorecard: uncertainty now, RMSE, consistency (NEES) and gated outliers
"""

from collections import deque
import math

from matplotlib.ticker import FuncFormatter
import numpy as np

from .fusion import FILTER_SENSORS

# Validated categorical slots 1-3 plus neutral chrome.
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
SHADE = '#f0efec'
FUSED = '#2a78d6'
ULTRASONIC = '#eb6834'
INFRARED = '#1baf7a'

SENSOR_COLORS = {'ultrasonic': ULTRASONIC, 'infrared': INFRARED}
SENSOR_LABELS = {'ultrasonic': 'Ultrasonic reading', 'infrared': 'IR reading'}
FILTER_COLORS = {'fused': FUSED, 'ultrasonic_only': ULTRASONIC, 'infrared_only': INFRARED}
FILTER_LABELS = {
    'fused': 'Fused KF (US + IR)',
    'ultrasonic_only': 'Ultrasonic-only KF',
    'infrared_only': 'IR-only KF',
}
DRAW_ORDER = ('ultrasonic_only', 'infrared_only', 'fused')

OUTLIER_SIGMAS = 5.0   # raw readings further than this from the truth count as outliers
SIGMA_AXIS_MM = (0.3, 300.0)


class BinnedError:
    """Running RMSE of an error signal, binned by the true distance."""

    def __init__(self, edges: np.ndarray):
        self.edges = edges
        self.centers = 0.5 * (edges[:-1] + edges[1:])
        self.sum_sq = np.zeros(len(self.centers))
        self.count = np.zeros(len(self.centers))

    def add(self, truth: np.ndarray, error: np.ndarray):
        idx = np.searchsorted(self.edges, truth, side='right') - 1
        ok = (idx >= 0) & (idx < len(self.centers))
        np.add.at(self.sum_sq, idx[ok], error[ok] ** 2)
        np.add.at(self.count, idx[ok], 1.0)

    def rmse(self, min_count: int = 10) -> np.ndarray:
        with np.errstate(invalid='ignore', divide='ignore'):
            rmse = np.sqrt(self.sum_sq / self.count)
        rmse[self.count < min_count] = np.nan
        return rmse


class FusionRecorder:
    """
    Rolling store of truth, readings and estimates, plus error statistics.

    Time-series data is kept for `window` seconds; the per-distance error
    statistics accumulate over the whole run.
    """

    def __init__(self, models: dict, window: float = 40.0, bin_width: float = 0.05,
                 max_distance: float = 1.0):
        self.models = models
        self.window = window
        # Bin centres on multiples of bin_width, so dwell distances like
        # 0.15 m or 0.45 m sit in the middle of a bin rather than on an edge.
        self.edges = np.arange(bin_width / 2, max_distance + bin_width, bin_width)
        self.reset()

    def reset(self):
        self.latest = -math.inf
        self.truth = deque()
        self.readings = {s: deque() for s in self.models}
        self.estimates = {f: deque() for f in FILTER_SENSORS}
        self.binned = {f: BinnedError(self.edges) for f in FILTER_SENSORS}
        self.binned_raw = {s: BinnedError(self.edges) for s in self.models}
        self.outliers = dict.fromkeys(self.models, 0)
        self._pending_raw = {s: [] for s in self.models}
        self._pending_est = {f: [] for f in FILTER_SENSORS}

    # -- input ---------------------------------------------------------------
    def add_truth(self, t: float, distance: float):
        if math.isfinite(distance):
            self.truth.append((t, distance))
            self.latest = max(self.latest, t)

    def add_reading(self, sensor: str, t: float, z: float):
        self.readings[sensor].append((t, z))
        self._pending_raw[sensor].append((t, z))
        self.latest = max(self.latest, t)

    def add_estimate(self, filter_name: str, t: float, distance: float, variance: float,
                     rejected: bool):
        self.estimates[filter_name].append((t, distance, variance, float(rejected)))
        self._pending_est[filter_name].append((t, distance))
        self.latest = max(self.latest, t)

    # -- bookkeeping -----------------------------------------------------------
    def update(self):
        """Score samples against the truth and drop data that left the window."""
        if len(self.truth) >= 2:
            tt, td = np.array(self.truth).T
            for sensor, pending in self._pending_raw.items():
                ready, self._pending_raw[sensor] = self._split_ready(pending, tt[-1])
                if ready.size:
                    self._score_readings(sensor, ready, tt, td)
            for name, pending in self._pending_est.items():
                ready, self._pending_est[name] = self._split_ready(pending, tt[-1])
                if ready.size:
                    keep = ready[:, 0] >= tt[0]
                    truth = np.interp(ready[keep, 0], tt, td)
                    self.binned[name].add(truth, ready[keep, 1] - truth)

        cutoff = self.latest - self.window
        for series in (self.truth, *self.readings.values(), *self.estimates.values()):
            while series and series[0][0] < cutoff:
                series.popleft()

    @staticmethod
    def _split_ready(pending: list, truth_end: float):
        ready = [p for p in pending if p[0] <= truth_end]
        waiting = [p for p in pending if p[0] > truth_end]
        return np.array(ready, dtype=float).reshape(-1, 2), waiting

    def _score_readings(self, sensor: str, ready: np.ndarray, tt, td):
        t, z = ready[:, 0], ready[:, 1]
        keep = (t >= tt[0]) & np.isfinite(z)
        t, z = t[keep], z[keep]
        truth = np.interp(t, tt, td)
        model = self.models[sensor]
        in_range = (truth >= model.min_range) & (truth <= model.max_range)
        sigma = np.array([model.sigma(d) for d in truth])
        error = z - truth
        outlier = np.abs(error) > OUTLIER_SIGMAS * sigma
        self.outliers[sensor] += int(np.sum(outlier & in_range))
        good = in_range & ~outlier
        self.binned_raw[sensor].add(truth[good], error[good])

    # -- queries ----------------------------------------------------------------
    def truth_array(self) -> np.ndarray:
        return np.array(self.truth, dtype=float).reshape(-1, 2)

    def reading_array(self, sensor: str) -> np.ndarray:
        return np.array(self.readings[sensor], dtype=float).reshape(-1, 2)

    def estimate_array(self, filter_name: str) -> np.ndarray:
        return np.array(self.estimates[filter_name], dtype=float).reshape(-1, 4)

    def current_distance(self) -> float:
        return self.truth[-1][1] if self.truth else math.nan

    def filter_stats(self, filter_name: str) -> dict:
        """Scorecard numbers for one filter over the current window."""
        est = self.estimate_array(filter_name)
        truth = self.truth_array()
        stats = {'sigma_now': math.nan, 'rms_sigma': math.nan, 'rmse': math.nan,
                 'nees': math.nan, 'gated': 0, 'samples': len(est)}
        if not len(est):
            return stats
        t, d, var, rejected = est.T
        stats['sigma_now'] = math.sqrt(var[-1])
        stats['rms_sigma'] = math.sqrt(np.mean(var))
        stats['gated'] = int(rejected.sum())
        if len(truth) >= 2:
            inside = (t >= truth[0, 0]) & (t <= truth[-1, 0])
            if np.any(inside):
                error = d[inside] - np.interp(t[inside], truth[:, 0], truth[:, 1])
                stats['rmse'] = math.sqrt(np.mean(error ** 2))
                stats['nees'] = float(np.mean(error ** 2 / var[inside]))
        return stats

    def uncertainty_reduction(self) -> tuple[float, float]:
        """
        How much smaller the fused sigma is than the best single-sensor sigma.

        Returns (average over the window, latest value) of
        1 - sigma_fused / min(sigma_ultrasonic_only, sigma_infrared_only).
        Every reading produces an estimate from all three filters with the same
        timestamp, so the sigmas are compared at identical instants.
        """
        sigma = {}
        for name in FILTER_SENSORS:
            est = self.estimate_array(name)
            sigma[name] = dict(zip(est[:, 0], np.sqrt(est[:, 2])))
        common = sorted(set(sigma['fused']) & set(sigma['ultrasonic_only'])
                        & set(sigma['infrared_only']))
        if not common:
            return math.nan, math.nan
        reduction = np.array([
            1.0 - sigma['fused'][t] / min(sigma['ultrasonic_only'][t], sigma['infrared_only'][t])
            for t in common])
        return float(np.mean(reduction)), float(reduction[-1])

    def raw_noise(self, sensor: str) -> float:
        """Std of raw reading errors (outliers excluded) over the whole run."""
        binned = self.binned_raw[sensor]
        count = binned.count.sum()
        return math.sqrt(binned.sum_sq.sum() / count) if count else math.nan


def _style(ax, title: str):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle='-')
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=3, width=0.6)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.set_title(title, loc='left', fontsize=10, color=INK, pad=6)


def _legend(ax, ncol: int):
    if not ax.get_legend_handles_labels()[0]:
        return   # nothing drawn yet
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=ncol, frameon=False,
              fontsize=8, labelcolor=INK_SECONDARY, handlelength=1.6, columnspacing=1.2,
              borderaxespad=0.2)


def _plain_log_ticks(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))


def _spans(t: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """Contiguous [start, end] time spans where mask is True."""
    spans, start = [], None
    for ti, m in zip(t, mask):
        if m and start is None:
            start = ti
        elif not m and start is not None:
            spans.append((start, ti))
            start = None
    if start is not None:
        spans.append((start, t[-1]))
    return spans


class FusionFigure:

    def __init__(self, figure, models: dict, title: str):
        self.fig = figure
        self.models = models
        figure.patch.set_facecolor(SURFACE)
        grid = figure.add_gridspec(3, 3, height_ratios=(1.25, 1.0, 1.2), hspace=0.5,
                                   wspace=0.28, left=0.06, right=0.985, top=0.9, bottom=0.08)
        self.ax_distance = figure.add_subplot(grid[0, :])
        self.ax_sigma = figure.add_subplot(grid[1, :], sharex=self.ax_distance)
        self.ax_bins = figure.add_subplot(grid[2, :2])
        self.ax_card = figure.add_subplot(grid[2, 2])
        figure.suptitle(title, x=0.06, ha='left', fontsize=13, color=INK, fontweight='bold')

    def draw(self, rec: FusionRecorder):
        for ax in (self.ax_distance, self.ax_sigma, self.ax_bins, self.ax_card):
            ax.clear()
        truth = rec.truth_array()
        ir_max = self.models['infrared'].max_range
        if len(truth):
            t_end = truth[-1, 0]
            self.ax_distance.set_xlim(max(t_end - rec.window, truth[0, 0]), t_end + 0.5)
        self._draw_distance(rec, truth)
        self._draw_sigma(rec, truth, ir_max)
        self._draw_bins(rec, ir_max)
        self._draw_card(rec)

    def _draw_distance(self, rec, truth):
        ax = self.ax_distance
        _style(ax, 'Distance to the object')
        if len(truth):
            ax.plot(truth[:, 0], truth[:, 1], color=INK, lw=1.4, label='True distance', zorder=4)
        for sensor in ('ultrasonic', 'infrared'):
            raw = rec.reading_array(sensor)
            raw = raw[np.isfinite(raw[:, 1])]
            ax.plot(raw[:, 0], raw[:, 1], ls='none', marker='o', ms=2.8, mec='none', alpha=0.55,
                    color=SENSOR_COLORS[sensor], label=SENSOR_LABELS[sensor], zorder=2)
        est = rec.estimate_array('fused')
        if len(est):
            t, d, sigma = est[:, 0], est[:, 1], np.sqrt(est[:, 2])
            ax.fill_between(t, d - 2 * sigma, d + 2 * sigma, color=FUSED, alpha=0.2, lw=0,
                            label='Fused ±2σ', zorder=3)
            ax.plot(t, d, color=FUSED, lw=1.6, label='Fused estimate', zorder=5)
        top = truth[:, 1].max() + 0.15 if len(truth) else 1.0
        ax.set_ylim(0.0, top)
        ax.set_ylabel('distance [m]')
        ax.tick_params(labelbottom=False)
        _legend(ax, ncol=5)

    def _draw_sigma(self, rec, truth, ir_max):
        ax = self.ax_sigma
        _style(ax, 'Uncertainty each filter reports (1σ, lower is better)')
        ax.set_yscale('log')
        if len(truth) >= 2:
            for i, (a, b) in enumerate(_spans(truth[:, 0], truth[:, 1] > ir_max)):
                ax.axvspan(a, b, color=SHADE, lw=0, zorder=0)
                if i == 0:
                    ax.text(a + 0.3, SIGMA_AXIS_MM[1] * 0.7, 'IR out of range', fontsize=8,
                            color=INK_MUTED, va='top')
        for name in DRAW_ORDER:
            est = rec.estimate_array(name)
            if len(est):
                ax.plot(est[:, 0], 1000 * np.sqrt(est[:, 2]), color=FILTER_COLORS[name],
                        lw=1.4 if name == 'fused' else 1.2, label=FILTER_LABELS[name])
        ax.set_ylim(*SIGMA_AXIS_MM)
        _plain_log_ticks(ax)
        ax.set_ylabel('σ [mm]')
        ax.set_xlabel('simulation time [s]')
        _legend(ax, ncol=3)

    def _draw_bins(self, rec, ir_max):
        ax = self.ax_bins
        _style(ax, 'Error vs distance: one reading (lines) vs Kalman filter (dots)')
        ax.set_yscale('log')
        us, ir = self.models['ultrasonic'], self.models['infrared']
        d = np.linspace(0.10, rec.edges[-1], 300)
        sigma_us = np.array([us.sigma(x) for x in d])
        sigma_ir = np.array([ir.sigma(x) for x in d])
        ir_ok = (d >= ir.min_range) & (d <= ir.max_range)
        fused = np.where(ir_ok, 1.0 / np.sqrt(1.0 / sigma_us ** 2 + 1.0 / sigma_ir ** 2), sigma_us)
        ax.axvspan(ir_max, d[-1], color=SHADE, lw=0, zorder=0)
        ax.text(ir_max + 0.01, SIGMA_AXIS_MM[1] * 0.7, 'IR blind', fontsize=8,
                color=INK_MUTED, va='top')
        ax.plot(d, 1000 * sigma_us, color=ULTRASONIC, lw=1.2, ls='--', label='Ultrasonic reading')
        ax.plot(d[ir_ok], 1000 * sigma_ir[ir_ok], color=INFRARED, lw=1.2, ls='--',
                label='IR reading')
        ax.plot(d, 1000 * fused, color=FUSED, lw=1.2, ls='--', label='Both readings fused')
        for name in DRAW_ORDER:
            binned = rec.binned[name]
            ax.plot(binned.centers, 1000 * binned.rmse(), ls='none', marker='o', ms=7,
                    mec=SURFACE, mew=1.2, color=FILTER_COLORS[name], label=FILTER_LABELS[name])
        now = rec.current_distance()
        if math.isfinite(now):
            ax.axvline(now, color=INK_SECONDARY, lw=1.0)
            ax.text(now + 0.008, SIGMA_AXIS_MM[0] * 1.3, 'now', fontsize=8, color=INK_SECONDARY)
        ax.set_xlim(d[0], d[-1])
        ax.set_ylim(*SIGMA_AXIS_MM)
        _plain_log_ticks(ax)
        ax.set_xlabel('true distance [m]')
        ax.set_ylabel('1σ / RMSE [mm]')
        ax.legend(loc='upper left', ncol=2, frameon=False, fontsize=7.5,
                  labelcolor=INK_SECONDARY, handlelength=1.6)

    def _draw_card(self, rec):
        ax = self.ax_card
        ax.axis('off')
        stats = {name: rec.filter_stats(name) for name in FILTER_SENSORS}
        # Headline: the error over the whole window compared with the best filter
        # you could have built from one sensor. The instant-by-instant sigma
        # comparison is smaller wherever one sensor dominates, so it is the subtext.
        single = [stats[n]['rmse'] for n in ('ultrasonic_only', 'infrared_only')]
        single = [rmse for rmse in single if math.isfinite(rmse)]
        fused_rmse = stats['fused']['rmse']
        error_cut = 1.0 - fused_rmse / min(single) if single else math.nan
        _, sigma_cut_now = rec.uncertainty_reduction()
        ax.text(0.0, 1.0, _percent(error_cut), fontsize=30, color=INK, va='top',
                transform=ax.transAxes)
        ax.text(0.0, 0.73, 'less error than the best single-sensor filter\n'
                f'over the last {rec.window:.0f} s (σ right now: '
                f'{_percent(sigma_cut_now)} lower)',
                fontsize=9, color=INK_SECONDARY, va='top', transform=ax.transAxes)

        columns = (('σ now', 0.58), ('RMSE', 0.74), ('NEES', 0.88), ('gated', 1.0))
        y = 0.40
        for label, x in columns:
            ax.text(x, y, label, fontsize=8, color=INK_MUTED, ha='right', transform=ax.transAxes)
        ax.text(0.0, y, '[mm]', fontsize=8, color=INK_MUTED, transform=ax.transAxes)
        for name in ('fused', 'ultrasonic_only', 'infrared_only'):
            y -= 0.12
            s = stats[name]
            ax.plot([0.015], [y + 0.025], marker='s', ms=7, color=FILTER_COLORS[name],
                    transform=ax.transAxes, clip_on=False)
            ax.text(0.05, y, FILTER_LABELS[name].replace(' KF', ''), fontsize=8.5, color=INK,
                    transform=ax.transAxes)
            values = (_mm(s['sigma_now']), _mm(s['rmse']), _num(s['nees']), str(s['gated']))
            for (_, x), value in zip(columns, values):
                ax.text(x, y, value, fontsize=8.5, color=INK, ha='right', transform=ax.transAxes)
        ax.text(0.0, y - 0.13, 'NEES ≈ 1: the σ a filter reports matches its real error.\n'
                f'Stats over the last {rec.window:.0f} s; gated = readings rejected as outliers.',
                fontsize=7.5, color=INK_MUTED, va='top', transform=ax.transAxes)


def _mm(value: float) -> str:
    if not math.isfinite(value):
        return '–'
    mm = 1000.0 * value
    return f'{mm:.1f}' if mm < 100 else f'{mm:.0f}'


def _percent(value: float) -> str:
    return f'{100.0 * value:.0f}%' if math.isfinite(value) else '–'


def _num(value: float) -> str:
    return f'{value:.2f}' if math.isfinite(value) else '–'
