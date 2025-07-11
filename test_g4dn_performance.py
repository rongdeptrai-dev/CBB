#!/usr/bin/env python3
"""
GPU Performance Test for AWS G4dn.xlarge
T4 GPU (16GB VRAM) + 16GB RAM + 4 vCPUs

This script tests:
- GPU availability and memory
- Model loading performance
- Inference speed and throughput
- Memory usage optimization
- Concurrent request handling
"""

import os
import sys
import time
import torch
import psutil
import asyncio
import concurrent.futures
from typing import List, Dict, Any
from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline
import gc

class G4dnPerformanceTester:
    def __init__(self):
        self.device = None
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.test_results = {}
        
    def check_system_specs(self):
        """Check if system meets G4dn.xlarge specifications"""
        print("🔍 Checking System Specifications...")
        
        # CPU check
        cpu_count = psutil.cpu_count()
        print(f"CPU Cores: {cpu_count}")
        if cpu_count < 4:
            print("⚠️ Warning: Expected 4+ CPU cores for G4dn.xlarge")
        
        # RAM check
        memory = psutil.virtual_memory()
        total_ram_gb = memory.total / (1024**3)
        print(f"Total RAM: {total_ram_gb:.1f}GB")
        if total_ram_gb < 15:
            print("⚠️ Warning: Expected 16GB+ RAM for G4dn.xlarge")
        
        # GPU check
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
                
                if "T4" in gpu_name and gpu_memory > 15:
                    print("✅ T4 GPU detected with sufficient VRAM")
                    self.device = torch.device(f"cuda:{i}")
                else:
                    print("⚠️ Warning: Expected T4 GPU with 16GB VRAM")
        else:
            print("❌ CUDA not available")
            self.device = torch.device("cpu")
            
        return {
            "cpu_cores": cpu_count,
            "total_ram_gb": total_ram_gb,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    
    def test_model_loading(self):
        """Test loading different models and measure performance"""
        print("\n🧠 Testing Model Loading Performance...")
        
        models_to_test = [
            ("VietAI/vit5-base", "t5"),
            ("VietAI/vit5-large", "t5"),
            ("microsoft/DialoGPT-medium", "gpt"),
        ]
        
        loading_results = {}
        
        for model_name, model_type in models_to_test:
            print(f"\n📥 Loading {model_name}...")
            start_time = time.time()
            
            try:
                # Clear GPU memory
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Load model
                if model_type == "t5":
                    model = T5ForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        device_map="auto" if self.device.type == "cuda" else None,
                        low_cpu_mem_usage=True
                    )
                    
                    # Create pipeline
                    test_pipeline = pipeline(
                        "text2text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        device=0 if self.device.type == "cuda" else -1,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.8,
                        batch_size=4 if self.device.type == "cuda" else 1
                    )
                
                loading_time = time.time() - start_time
                
                # Test memory usage
                if self.device.type == "cuda":
                    gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                else:
                    gpu_memory_used = 0
                    gpu_memory_reserved = 0
                
                system_memory = psutil.virtual_memory()
                ram_used = system_memory.used / (1024**3)
                
                loading_results[model_name] = {
                    "loading_time": loading_time,
                    "gpu_memory_used": gpu_memory_used,
                    "gpu_memory_reserved": gpu_memory_reserved,
                    "ram_used": ram_used,
                    "success": True
                }
                
                print(f"✅ Loaded in {loading_time:.2f}s")
                print(f"   GPU VRAM: {gpu_memory_used:.1f}GB used, {gpu_memory_reserved:.1f}GB reserved")
                print(f"   System RAM: {ram_used:.1f}GB used")
                
                # Keep the largest model for inference testing
                if "large" in model_name and model_type == "t5":
                    self.model = model
                    self.tokenizer = tokenizer
                    self.pipeline = test_pipeline
                
                # Clean up
                del model, tokenizer, test_pipeline
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                loading_results[model_name] = {
                    "error": str(e),
                    "success": False
                }
        
        return loading_results
    
    def test_inference_speed(self):
        """Test inference speed with different batch sizes and lengths"""
        print("\n⚡ Testing Inference Speed...")
        
        if not self.pipeline:
            print("❌ No model loaded for testing")
            return {}
        
        test_messages = [
            "Xin chào, tôi cần hỗ trợ về tài khoản TikTok.",
            "Video của tôi không hiển thị trên For You Page, tại sao vậy?",
            "Làm thế nào để tăng follower trên TikTok một cách hiệu quả?",
            "Tôi bị hack tài khoản, cần làm gì để lấy lại?",
            "Cách sử dụng filter và effect trong TikTok như thế nào?"
        ]
        
        inference_results = {}
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8] if self.device.type == "cuda" else [1]
        
        for batch_size in batch_sizes:
            print(f"\n🧪 Testing batch size: {batch_size}")
            
            # Prepare batch
            batch_messages = (test_messages * ((batch_size // len(test_messages)) + 1))[:batch_size]
            
            # Single inference test
            start_time = time.time()
            try:
                responses = self.pipeline(batch_messages)
                inference_time = time.time() - start_time
                
                # Calculate metrics
                throughput = batch_size / inference_time
                avg_time_per_message = inference_time / batch_size
                
                print(f"✅ Batch {batch_size}: {inference_time:.2f}s total, {avg_time_per_message:.2f}s per message")
                print(f"   Throughput: {throughput:.2f} messages/second")
                
                inference_results[f"batch_{batch_size}"] = {
                    "batch_size": batch_size,
                    "total_time": inference_time,
                    "avg_time_per_message": avg_time_per_message,
                    "throughput": throughput,
                    "success": True
                }
                
            except Exception as e:
                print(f"❌ Batch {batch_size} failed: {e}")
                inference_results[f"batch_{batch_size}"] = {
                    "error": str(e),
                    "success": False
                }
        
        return inference_results
    
    def test_concurrent_requests(self, num_concurrent=10):
        """Test handling concurrent requests"""
        print(f"\n🔄 Testing {num_concurrent} Concurrent Requests...")
        
        if not self.pipeline:
            print("❌ No model loaded for testing")
            return {}
        
        test_message = "Tôi cần hỗ trợ về vấn đề bảo mật tài khoản TikTok."
        
        async def single_request():
            start_time = time.time()
            try:
                # Simulate async processing
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    response = await loop.run_in_executor(
                        executor, 
                        lambda: self.pipeline(test_message)
                    )
                processing_time = time.time() - start_time
                return {"success": True, "time": processing_time}
            except Exception as e:
                return {"success": False, "error": str(e), "time": time.time() - start_time}
        
        async def run_concurrent_test():
            start_time = time.time()
            tasks = [single_request() for _ in range(num_concurrent)]
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            successful_requests = [r for r in results if r["success"]]
            failed_requests = [r for r in results if not r["success"]]
            
            if successful_requests:
                avg_request_time = sum(r["time"] for r in successful_requests) / len(successful_requests)
                max_request_time = max(r["time"] for r in successful_requests)
                min_request_time = min(r["time"] for r in successful_requests)
            else:
                avg_request_time = max_request_time = min_request_time = 0
            
            return {
                "total_requests": num_concurrent,
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "total_time": total_time,
                "avg_request_time": avg_request_time,
                "max_request_time": max_request_time,
                "min_request_time": min_request_time,
                "requests_per_second": num_concurrent / total_time if total_time > 0 else 0
            }
        
        # Run the concurrent test
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            concurrent_results = loop.run_until_complete(run_concurrent_test())
            loop.close()
            
            print(f"✅ Concurrent test completed:")
            print(f"   Success rate: {concurrent_results['successful_requests']}/{concurrent_results['total_requests']}")
            print(f"   Avg response time: {concurrent_results['avg_request_time']:.2f}s")
            print(f"   Throughput: {concurrent_results['requests_per_second']:.2f} req/s")
            
            return concurrent_results
            
        except Exception as e:
            print(f"❌ Concurrent test failed: {e}")
            return {"error": str(e)}
    
    def test_memory_optimization(self):
        """Test memory usage patterns and optimization"""
        print("\n💾 Testing Memory Optimization...")
        
        if not self.pipeline:
            print("❌ No model loaded for testing")
            return {}
        
        memory_results = {}
        
        # Baseline memory
        if self.device.type == "cuda":
            baseline_gpu = torch.cuda.memory_allocated(0) / (1024**3)
        baseline_ram = psutil.virtual_memory().used / (1024**3)
        
        print(f"Baseline - GPU: {baseline_gpu if self.device.type == 'cuda' else 'N/A'}GB, RAM: {baseline_ram:.1f}GB")
        
        # Test with increasing message lengths
        test_lengths = [50, 100, 200, 500]
        
        for length in test_lengths:
            test_message = "Tôi cần hỗ trợ về TikTok. " * (length // 30)
            
            # Before inference
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            # Run inference
            try:
                start_time = time.time()
                response = self.pipeline(test_message)
                inference_time = time.time() - start_time
                
                # After inference
                if self.device.type == "cuda":
                    current_gpu = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_increase = current_gpu - baseline_gpu
                else:
                    current_gpu = 0
                    gpu_increase = 0
                
                current_ram = psutil.virtual_memory().used / (1024**3)
                ram_increase = current_ram - baseline_ram
                
                memory_results[f"length_{length}"] = {
                    "message_length": length,
                    "inference_time": inference_time,
                    "gpu_memory_used": current_gpu,
                    "ram_memory_used": current_ram,
                    "gpu_increase": gpu_increase,
                    "ram_increase": ram_increase
                }
                
                print(f"Length {length}: {inference_time:.2f}s, GPU+{gpu_increase:.2f}GB, RAM+{ram_increase:.2f}GB")
                
            except Exception as e:
                print(f"❌ Length {length} failed: {e}")
                memory_results[f"length_{length}"] = {"error": str(e)}
        
        return memory_results
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        print("\n📊 Generating Performance Report...")
        
        report = {
            "system_specs": self.test_results.get("system_specs", {}),
            "model_loading": self.test_results.get("model_loading", {}),
            "inference_speed": self.test_results.get("inference_speed", {}),
            "concurrent_requests": self.test_results.get("concurrent_requests", {}),
            "memory_optimization": self.test_results.get("memory_optimization", {}),
            "recommendations": []
        }
        
        # Generate recommendations
        if self.device and self.device.type == "cuda":
            report["recommendations"].append("✅ GPU acceleration enabled")
        else:
            report["recommendations"].append("⚠️ Consider using GPU for better performance")
        
        # Memory recommendations
        if report["system_specs"].get("total_ram_gb", 0) < 16:
            report["recommendations"].append("⚠️ Consider upgrading to 16GB+ RAM")
        
        # Performance recommendations
        inference_results = report.get("inference_speed", {})
        best_batch = None
        best_throughput = 0
        
        for batch_key, batch_result in inference_results.items():
            if batch_result.get("success") and batch_result.get("throughput", 0) > best_throughput:
                best_throughput = batch_result["throughput"]
                best_batch = batch_result["batch_size"]
        
        if best_batch:
            report["recommendations"].append(f"✅ Optimal batch size: {best_batch} (throughput: {best_throughput:.2f} msg/s)")
        
        return report
    
    def run_all_tests(self):
        """Run all performance tests"""
        print("🚀 Starting G4dn.xlarge Performance Tests...")
        print("=" * 60)
        
        # System specs
        self.test_results["system_specs"] = self.check_system_specs()
        
        # Model loading
        self.test_results["model_loading"] = self.test_model_loading()
        
        # Inference speed
        self.test_results["inference_speed"] = self.test_inference_speed()
        
        # Concurrent requests
        self.test_results["concurrent_requests"] = self.test_concurrent_requests()
        
        # Memory optimization
        self.test_results["memory_optimization"] = self.test_memory_optimization()
        
        # Generate report
        report = self.generate_report()
        
        print("\n" + "=" * 60)
        print("📋 PERFORMANCE REPORT SUMMARY")
        print("=" * 60)
        
        for recommendation in report["recommendations"]:
            print(recommendation)
        
        print(f"\nDevice: {self.device}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        return report

def main():
    """Main function to run performance tests"""
    tester = G4dnPerformanceTester()
    
    try:
        report = tester.run_all_tests()
        
        # Save report to file
        import json
        with open("g4dn_performance_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Full report saved to: g4dn_performance_report.json")
        print("🎉 Performance testing completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
