from __future__ import annotations
from dataclasses import dataclass,field
_MAX_LATENCY_SAMPLES=1000
@dataclass
class ToolStats:
    invocations:int=0
    successes:int=0
    failures:int=0
    retries:int=0
    latencies:list[float]=field(default_factory=list)
    def record(self,latency_ms:float,success:bool)->None:
        self.invocations+=1
        if success:
            self.successes+=1
        else:
            self.failures+=1
        self.latencies.append(latency_ms)
        if len(self.latencies)>_MAX_LATENCY_SAMPLES:
            self.latencies=self.latencies[-_MAX_LATENCY_SAMPLES:]
    @property
    def avg_latency_ms(self)->float:
        return sum(self.latencies)/len(self.latencies) if self.latencies else 0.0
    @property
    def p99_latency_ms(self)->float:
        if not self.latencies:
            return 0.0
        s=sorted(self.latencies)
        return s[max(0,int(len(s)*0.99)-1)]
    @property
    def success_rate(self)->float:
        return self.successes/self.invocations if self.invocations else 0.0
class ToolMetrics:
    def __init__(self)->None:
        self._stats:dict[str,ToolStats]={}
    def get(self,name:str)->ToolStats:
        if name not in self._stats:
            self._stats[name]=ToolStats()
        return self._stats[name]
    def record(self,name:str,latency_ms:float,success:bool,retries:int=0)->None:
        s=self.get(name)
        s.record(latency_ms,success)
        s.retries+=retries
    def all_stats(self)->dict[str,ToolStats]:
        return dict(self._stats)
    def format_table(self)->str:
        if not self._stats:
            return "No tool invocations recorded yet."
        col=18
        rows=[
            f"{'Tool':<{col}} | {'Calls':>5} | {'OK':>4} | {'Fail':>4} | {'Retry':>5} | {'Avg ms':>8} | {'P99 ms':>8} | {'OK%':>6}",
            "-"*(col+60),
        ]
        for name,s in sorted(self._stats.items()):
            rows.append(
                f"{name:<{col}} | {s.invocations:>5} | {s.successes:>4} | {s.failures:>4} | "
                f"{s.retries:>5} | {s.avg_latency_ms:>8.1f} | {s.p99_latency_ms:>8.1f} | {s.success_rate*100:>5.1f}%"
            )
        return "\n".join(rows)
