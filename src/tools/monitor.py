"""
System resource monitor for phi-auto.
Tracks RAM, CPU, battery, temperature on Android/Termux.
"""

import os
import time
import subprocess


def get_ram_usage():
    """Get RAM usage in MB. Returns (used, total, percent)."""
    try:
        with open('/proc/meminfo', 'r') as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(':')] = int(parts[1])
        total = info.get('MemTotal', 0) / 1024  # MB
        available = info.get('MemAvailable', info.get('MemFree', 0)) / 1024
        used = total - available
        pct = (used / total * 100) if total > 0 else 0
        return round(used), round(total), round(pct, 1)
    except Exception:
        return 0, 0, 0


def get_cpu_usage(interval=0.5):
    """Get CPU usage percentage over a short interval."""
    try:
        def read_stat():
            with open('/proc/stat', 'r') as f:
                line = f.readline()
            parts = line.split()[1:]
            vals = [int(x) for x in parts]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            total = sum(vals)
            return idle, total

        idle1, total1 = read_stat()
        time.sleep(interval)
        idle2, total2 = read_stat()

        idle_d = idle2 - idle1
        total_d = total2 - total1
        if total_d == 0:
            return 0.0
        return round((1.0 - idle_d / total_d) * 100, 1)
    except Exception:
        return 0.0


def get_battery():
    """Get battery level and charging status on Android."""
    try:
        # Termux battery status
        result = subprocess.run(
            ['termux-battery-status'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            return {
                'percentage': data.get('percentage', -1),
                'status': data.get('status', 'unknown'),
                'temperature': data.get('temperature', 0),
            }
    except Exception:
        pass

    # fallback: read from sysfs
    try:
        base = '/sys/class/power_supply/battery'
        pct = int(open(f'{base}/capacity').read().strip())
        status = open(f'{base}/status').read().strip()
        try:
            temp = int(open(f'{base}/temp').read().strip()) / 10.0
        except Exception:
            temp = 0
        return {'percentage': pct, 'status': status, 'temperature': temp}
    except Exception:
        return {'percentage': -1, 'status': 'unknown', 'temperature': 0}


def get_cpu_temp():
    """Get CPU temperature in Celsius."""
    thermal_zones = [
        '/sys/class/thermal/thermal_zone0/temp',
        '/sys/class/thermal/thermal_zone1/temp',
        '/sys/class/thermal/thermal_zone2/temp',
    ]
    for path in thermal_zones:
        try:
            temp = int(open(path).read().strip())
            if temp > 1000:
                temp /= 1000.0
            if 10 < temp < 120:
                return round(temp, 1)
        except Exception:
            continue
    return 0.0


def get_disk_free(path=None):
    """Get free disk space in MB."""
    if path is None:
        path = os.path.expanduser('~')
    try:
        stat = os.statvfs(path)
        free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        return round(free_mb)
    except Exception:
        return 0


def get_system_snapshot():
    """Get complete system status snapshot."""
    ram_used, ram_total, ram_pct = get_ram_usage()
    battery = get_battery()
    cpu_temp = get_cpu_temp()
    disk_free = get_disk_free()

    return {
        'timestamp': time.time(),
        'ram_used_mb': ram_used,
        'ram_total_mb': ram_total,
        'ram_pct': ram_pct,
        'cpu_temp_c': cpu_temp,
        'battery_pct': battery['percentage'],
        'battery_status': battery['status'],
        'battery_temp_c': battery['temperature'],
        'disk_free_mb': disk_free,
    }


def format_snapshot(snap):
    """Format snapshot as a one-line status string."""
    parts = [
        f"RAM: {snap['ram_used_mb']}/{snap['ram_total_mb']}MB ({snap['ram_pct']}%)",
        f"CPU: {snap['cpu_temp_c']}°C",
        f"Bat: {snap['battery_pct']}% ({snap['battery_status']})",
        f"Disk: {snap['disk_free_mb']}MB free",
    ]
    return " | ".join(parts)


class ResourceGuard:
    """Monitors resources and can halt training if limits exceeded."""

    def __init__(self, max_ram_pct=85, min_battery_pct=15, max_cpu_temp=75):
        self.max_ram_pct = max_ram_pct
        self.min_battery_pct = min_battery_pct
        self.max_cpu_temp = max_cpu_temp
        self.history = []

    def check(self, verbose=False):
        """Check if resources are OK. Returns (ok, reason)."""
        snap = get_system_snapshot()
        self.history.append(snap)

        if verbose:
            print(f"  [{format_snapshot(snap)}]")

        if snap['ram_pct'] > self.max_ram_pct:
            return False, f"RAM usage too high: {snap['ram_pct']}% > {self.max_ram_pct}%"

        if snap['battery_pct'] > 0 and snap['battery_pct'] < self.min_battery_pct:
            if snap['battery_status'].lower() not in ('charging', 'full'):
                return False, f"Battery low: {snap['battery_pct']}% < {self.min_battery_pct}%"

        if snap['cpu_temp_c'] > 0 and snap['cpu_temp_c'] > self.max_cpu_temp:
            return False, f"CPU too hot: {snap['cpu_temp_c']}°C > {self.max_cpu_temp}°C"

        return True, "OK"

    def wait_for_cooldown(self, target_temp=60, timeout=300):
        """Wait for CPU to cool down."""
        print(f"  Waiting for CPU to cool below {target_temp}°C...")
        start = time.time()
        while time.time() - start < timeout:
            temp = get_cpu_temp()
            if temp <= target_temp or temp == 0:
                print(f"  CPU cooled to {temp}°C")
                return True
            time.sleep(10)
        print(f"  Cooldown timeout after {timeout}s")
        return False


if __name__ == "__main__":
    print("=== phi-auto System Monitor ===\n")
    snap = get_system_snapshot()
    print(format_snapshot(snap))
    print(f"\nCPU usage: {get_cpu_usage()}%")
    print(f"\nFull snapshot: {snap}")
