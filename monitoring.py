# monitoring.py - System monitoring and metrics
import psutil
import time
import logging
from datetime import datetime
from typing import Dict, Any
import asyncio
import aiofiles
import json

class SystemMonitor:
    """System performance and health monitoring"""
    
    def __init__(self, log_file: str = "logs/metrics.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("system_monitor")
        
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
                "memory_total": memory.total,
                "memory_available": memory.available,
                "memory_percent": memory.percent,
                "disk_total": disk.total,
                "disk_free": disk.free,
                "disk_percent": disk.percent
            },
            "process": {
                "pid": process.pid,
                "memory_rss": process_memory.rss,
                "memory_vms": process_memory.vms,
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "create_time": process.create_time()
            }
        }
        
        return metrics
    
    async def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to file"""
        async with aiofiles.open(self.log_file, 'a') as f:
            await f.write(json.dumps(metrics) + '\n')
    
    async def check_health(self) -> Dict[str, str]:
        """Perform health checks"""
        health_status = {
            "overall": "healthy",
            "components": {}
        }
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            health_status["components"]["cpu"] = "critical"
            health_status["overall"] = "unhealthy"
        elif cpu_percent > 70:
            health_status["components"]["cpu"] = "warning" 
        else:
            health_status["components"]["cpu"] = "healthy"
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            health_status["components"]["memory"] = "critical"
            health_status["overall"] = "unhealthy"
        elif memory.percent > 80:
            health_status["components"]["memory"] = "warning"
        else:
            health_status["components"]["memory"] = "healthy"
        
        # Check disk space
        disk = psutil.disk_usage('/')
        if disk.percent > 95:
            health_status["components"]["disk"] = "critical"
            health_status["overall"] = "unhealthy"
        elif disk.percent > 85:
            health_status["components"]["disk"] = "warning"
        else:
            health_status["components"]["disk"] = "healthy"
        
        return health_status

class PerformanceProfiler:
    """Performance profiling for document processing"""
    
    def __init__(self):
        self.processing_times = []
        self.error_rates = {}
        self.throughput_metrics = {}
    
    def record_processing_time(self, operation: str, duration: float, success: bool):
        """Record processing time and success rate"""
        timestamp = time.time()
        
        self.processing_times.append({
            "operation": operation,
            "duration": duration,
            "success": success,
            "timestamp": timestamp
        })
        
        # Update error rates
        if operation not in self.error_rates:
            self.error_rates[operation] = {"total": 0, "errors": 0}
        
        self.error_rates[operation]["total"] += 1
        if not success:
            self.error_rates[operation]["errors"] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.processing_times:
            return {"message": "No performance data available"}
        
        # Calculate averages by operation
        operations = {}
        for record in self.processing_times:
            op = record["operation"]
            if op not in operations:
                operations[op] = []
            operations[op].append(record["duration"])
        
        summary = {}
        for op, times in operations.items():
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            
            error_rate = 0
            if op in self.error_rates:
                total = self.error_rates[op]["total"]
                errors = self.error_rates[op]["errors"]
                error_rate = (errors / total) * 100 if total > 0 else 0
            
            summary[op] = {
                "avg_duration": round(avg_time, 2),
                "max_duration": round(max_time, 2),
                "min_duration": round(min_time, 2),
                "total_operations": len(times),
                "error_rate_percent": round(error_rate, 2)
            }
        
        return summary
