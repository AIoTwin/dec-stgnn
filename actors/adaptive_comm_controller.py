import math
import os
from collections import deque

class AdaptiveCommController:
    """
    Simple adaptive controller for % removal of cross-cloudlet nodes
    using a 'SUDDEN_EVENT_RATE' metric which measures how accurate the model predicted sudden change in speed (i.e. traffic jam, or recovery from traffic jam)
    (values [0,1])

    - Learns a target RATE from warm-up: target_rate = base * (1 - target_budget).
    - Uses rolling window mean/median to decide.
    - Waits `settle_epochs` validations after any change.
    - Enforces dynamic bounds so you never remove too many nodes.
    """

    def __init__(self,
                 start_frac=0.10,        # start removal fraction, e.g., 10%
                 p_min=0.10,             # never remove less than 10%
                 user_cap=0.70,          # hard cap (safety)
                 min_scoring_nodes=16,   # must remain after removal
                 min_remaining_frac=0.25,# or at least 25% of pool remains
                 warmup_epochs=2,        # learn target during warm-up (if 0, then use only inital train data)
                 target_budget=0.15,     # allow (+0.03 | 3%) worse than warm-up aggregation
                 window=3,               # aggregation over last W(window) MAEs
                 up_margin=0.00,         # agg <= target*(1+up_margin) -> increase
                 down_margin=0.03,       # agg >= target*(1+down_margin) -> decrease
                 step_up=0.05,           # +5% when quality is fine
                 step_down=0.05,         # -5% when quality slips
                 settle_epochs=3):       # wait this many validations after any change
        self.drop_frac = start_frac

        # safety bounds
        self.p_min = p_min
        self.user_cap = user_cap
        self.min_scoring_nodes = min_scoring_nodes
        self.min_remaining_frac = min_remaining_frac

        # target learning
        self.warmup_epochs = warmup_epochs
        self.target_budget = target_budget
        self._warm = deque(maxlen=max(1, warmup_epochs + 1))
        self.target_rate = None

        # decision window
        self.window = window
        self._vals = deque(maxlen=window)
        self.up_margin = up_margin
        self.down_margin = down_margin

        # steps & settling
        self.step_up = step_up
        self.step_down = step_down
        self.settle_epochs = settle_epochs
        self._since_change = 1_000_000  # large so first decision is allowed when warm-up ends

    # ---- helpers ----
    def _pmax_given_pool(self, n_pool: int) -> float:
        """Dynamic max removable fraction given current cross-cloudlet pool."""
        if n_pool <= 0:
            return 0.0
        must_remain = max(self.min_scoring_nodes,
                          math.ceil(self.min_remaining_frac * n_pool)) # maximum number of cross cloudlet nodes that can remain in the cloudlet (n_pool is the number of cross cloudlet nodes a cloudlet has)
        return max(0.0, min(1.0 - must_remain / max(1, n_pool), 1.0))

    def current_fraction(self, n_pool: int) -> float:
        """Get clamped fraction to use this epoch when sampling nodes to remove."""
        upper = min(self._pmax_given_pool(n_pool), self.user_cap)
        self.drop_frac = min(max(self.drop_frac, self.p_min), upper)
        return self.drop_frac

    # ---- main update ----
    def update_after_validation(self, epoch: int, rate: float, n_pool: int,
                                logs_folder=None, cln_id=None, agg: str = "mean"):
        """
        Call once per epoch AFTER computing masked SUDDEN_EVENT_RATE on the next slice.

        Parameters
        ----------
        rate : float in [0, 1]
            How accurate the model predicted sudden change in speed (i.e. traffic jam, or recovery from traffic jam) - (higher is better).
        agg : {"mean","median"}, default "mean"
            Aggregation to compute the window statistic and warm-up target.
        """

        def _aggregate(vals, mode: str):
            # assumes len(vals) > 0
            if mode == "median":
                s = sorted(vals) # create a sorted list (to prepare for median)
                n = len(s) # total number of elements
                mid = n // 2 # get the element in the middle
                return s[mid] if (n % 2 == 1) else 0.5 * (s[mid - 1] + s[mid])
            # default: mean
            return sum(vals) / len(vals)
    
        # 0) warm-up: collect baseline
        if epoch <= self.warmup_epochs:
            if math.isfinite(rate) and rate >= 0:
                self._warm.append(rate)
                if epoch == self.warmup_epochs:
                    base = _aggregate(self._warm, agg)
                    # target is a LOWER BOUND; allow a drop of target_budget
                    self.target_rate = max(0.0, min(1.0, base * (1.0 - self.target_budget)))
            return

        # 1) update window & ensure we have target + enough points
        if not (math.isfinite(rate) and rate >= 0):
            return # if rate has wrong value, exit out of the function
        self._vals.append(rate) # add MAE in _errs deque
        
        if self.target_rate is None: # if, for some reason, target_vals hasn't been set, then set it now with current rate
            tmp = list(self._vals) if len(self._vals) > 0 else [rate]
            base = _aggregate(tmp, agg)
            self.target_rate = max(0.0, min(1.0, base * (1.0 - self.target_budget)))
        
        if len(self._vals) < self.window: # if we have less rate's is _vals deque (size of window), then exit out of the function
            self._since_change += 1
            return

        # 2) compute aggregation over last W
        statW = _aggregate(self._vals, agg)
        ratio = statW / max(self.target_rate, 1e-8) # ratio between aggregation of last "W (window)" received "rate" and calculated "target rate"

        # 3) respect dynamic bounds
        upper = min(self._pmax_given_pool(n_pool), self.user_cap) # upper limit of number of cross cloudlet nodes
        lower = self.p_min # lower limit of number of cross cloudlet nodes (we'll always remove at least 5% of cross cloudlet nodes)

        old = self.drop_frac # previous value for current % of how many cross cloudlet nodes a cloudlet can remove
        changed = False

        # 4) wait 'settle_epochs' validations after any change
        if self._since_change < self.settle_epochs:
            self._since_change += 1
        else:
            # 5) decisions (HIGHER is better):
            if ratio >= (1.0 + self.up_margin):
                # quality comfortably above target → increase removal
                if self.drop_frac + self.step_up <= upper + 1e-9:
                    self.drop_frac = min(upper, self.drop_frac + self.step_up)
                    changed = True
            elif ratio <= (1.0 - self.down_margin):
                # quality below target by margin → decrease removal
                if self.drop_frac - self.step_down >= lower - 1e-9:
                    self.drop_frac = max(lower, self.drop_frac - self.step_down)
                    changed = True

            if changed:
                self._since_change = 0
            else:
                self._since_change += 1

        # 6) final clamp
        self.drop_frac = min(max(self.drop_frac, lower), upper)

        # 7) tiny log (TXT)
        if logs_folder is not None:
            try:
                path = os.path.join(logs_folder, f"debugging/cloudlet_{cln_id}")
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, "drop_frac.txt"), "a") as f:
                    f.write(
                        f"epoch={epoch}, rate={rate:.4f}, statW={statW:.4f}, agg={agg}, "
                        f"target_rate={self.target_rate:.4f}, ratio={ratio:.4f}, "
                        f"n_pool={n_pool}, bounds=[{lower:.2f},{upper:.2f}], "
                        f"since_change={self._since_change}, drop_frac: {old:.2f}→{self.drop_frac:.2f}\n"
                    )
            except Exception:
                pass