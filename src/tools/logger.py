"""
Experiment logger for phi-auto.
Logs results to TSV and updates DEVLOG.md.
"""

import os
import time
import json
from datetime import datetime


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results.tsv')
DEVLOG_PATH = os.path.join(PROJECT_ROOT, 'DEVLOG.md')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')


class ExperimentLogger:
    """Logs experiments to results.tsv and individual JSON files."""

    TSV_HEADER = "exp_id\tdate\tconfig\tval_loss\tval_bpb\tbest_loss\ttok_sec\tram_mb\tsteps\ttime_sec\tstatus\tnotes\n"

    def __init__(self):
        os.makedirs(LOGS_DIR, exist_ok=True)
        self._ensure_tsv()
        self._exp_count = self._count_experiments()

    def _ensure_tsv(self):
        if not os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, 'w') as f:
                f.write(self.TSV_HEADER)

    def _count_experiments(self):
        try:
            with open(RESULTS_PATH) as f:
                return max(0, sum(1 for _ in f) - 1)
        except Exception:
            return 0

    def next_id(self):
        self._exp_count += 1
        return self._exp_count

    def log_experiment(self, exp_id, config, results, status="completed", notes=""):
        """Log experiment result to TSV."""
        row = (
            f"{exp_id}\t"
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}\t"
            f"{self._config_str(config)}\t"
            f"{results.get('val_loss', 0):.4f}\t"
            f"{results.get('val_bpb', 0):.4f}\t"
            f"{results.get('best_loss', 0):.4f}\t"
            f"{results.get('tok_sec', 0):.0f}\t"
            f"{results.get('ram_mb', 0)}\t"
            f"{results.get('steps', 0)}\t"
            f"{results.get('time_sec', 0):.0f}\t"
            f"{status}\t"
            f"{notes}\n"
        )
        with open(RESULTS_PATH, 'a') as f:
            f.write(row)

        # also save detailed JSON
        detail_path = os.path.join(LOGS_DIR, f"exp_{exp_id:04d}.json")
        detail = {
            'exp_id': exp_id,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'results': results,
            'status': status,
            'notes': notes,
        }
        with open(detail_path, 'w') as f:
            json.dump(detail, f, indent=2, default=str)

        return detail_path

    def _config_str(self, config):
        """Compact config string for TSV."""
        if isinstance(config, dict):
            parts = []
            for key in ['n_embd', 'n_head', 'n_layer', 'seq_len', 'lr', 'batch_size']:
                if key in config:
                    val = config[key]
                    if isinstance(val, float) and val < 0.01:
                        parts.append(f"{key}={val:.0e}")
                    else:
                        parts.append(f"{key}={val}")
            return ','.join(parts)
        return str(config)

    def get_best_result(self):
        """Get the best val_bpb from results."""
        best_bpb = float('inf')
        best_exp = None
        try:
            with open(RESULTS_PATH) as f:
                header = f.readline()
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 5:
                        try:
                            bpb = float(parts[4])
                            if bpb < best_bpb:
                                best_bpb = bpb
                                best_exp = int(parts[0])
                        except ValueError:
                            continue
        except Exception:
            pass
        return best_exp, best_bpb

    def get_all_results(self):
        """Get all experiment results as list of dicts."""
        results = []
        try:
            with open(RESULTS_PATH) as f:
                header = f.readline().strip().split('\t')
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= len(header):
                        results.append(dict(zip(header, parts)))
        except Exception:
            pass
        return results


def update_devlog(entry, section="Notes & Observations"):
    """Append an entry to DEVLOG.md under the specified section."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    new_entry = f"\n- [{timestamp}] {entry}"

    try:
        with open(DEVLOG_PATH, 'r') as f:
            content = f.read()

        marker = f"## {section}"
        if marker in content:
            idx = content.index(marker) + len(marker)
            # find end of line
            nl = content.index('\n', idx)
            content = content[:nl + 1] + new_entry + content[nl + 1:]
        else:
            content += f"\n\n## {section}\n{new_entry}\n"

        with open(DEVLOG_PATH, 'w') as f:
            f.write(content)
    except Exception as e:
        print(f"Warning: Could not update DEVLOG: {e}")


if __name__ == "__main__":
    logger = ExperimentLogger()
    print(f"Experiments logged: {logger._exp_count}")
    best_exp, best_bpb = logger.get_best_result()
    if best_exp:
        print(f"Best: exp {best_exp} with val_bpb={best_bpb:.4f}")
    else:
        print("No experiments yet.")
