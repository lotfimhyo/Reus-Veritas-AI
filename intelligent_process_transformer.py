#!/usr/bin/env python3
"""
Intelligent Process Transformer for Reus Veritas
نظام تحويل العمليات الذكية لمشروع Reus Veritas

هذا المكون مسؤول عن تحويل العمليات الإجرائية التقليدية إلى وحدات عمل ديناميكية وذكية
مع التحميل الديناميكي للأدوات والتكامل مع محرك المعرفة.

Author: Lotfi mahiddine
Date: 2025
"""

import json
import logging
import importlib
import inspect
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import concurrent.futures
import pickle
import hashlib

# إعداد نظام السجلات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process_transformer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProcessStatus(Enum):
    """حالات العملية"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class ProcessPriority(Enum):
    """أولويات العملية"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class ToolType(Enum):
    """أنواع الأدوات"""
    ANALYZER = "analyzer"
    TRANSFORMER = "transformer"
    VALIDATOR = "validator"
    EXECUTOR = "executor"
    MONITOR = "monitor"
    OPTIMIZER = "optimizer"

@dataclass
class ProcessStep:
    """خطوة في العملية"""
    id: str
    name: str
    description: str
    tool_name: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    timeout: Optional[int]
    retry_count: int
    status: ProcessStatus
    result: Optional[Any]
    error: Optional[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]

@dataclass
class DynamicProcess:
    """عملية ديناميكية"""
    id: str
    name: str
    description: str
    steps: List[ProcessStep]
    priority: ProcessPriority
    status: ProcessStatus
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    created_by: str

@dataclass
class ToolDefinition:
    """تعريف أداة"""
    name: str
    type: ToolType
    description: str
    module_path: str
    class_name: str
    parameters_schema: Dict[str, Any]
    dependencies: List[str]
    version: str
    author: str
    is_active: bool

class BaseTool(ABC):
    """الفئة الأساسية للأدوات"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"tool.{name}")
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """تنفيذ الأداة"""
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """التحقق من صحة المعاملات"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """معلومات الأداة"""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "config": self.config
        }

class TextAnalyzerTool(BaseTool):
    """أداة تحليل النصوص"""
    
    def __init__(self, name: str = "text_analyzer", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
    
    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """تحليل النص"""
        text = parameters.get("text", "")
        
        if not text:
            return {"success": False, "error": "النص مطلوب"}
        
        # تحليل أساسي للنص
        analysis = {
            "word_count": len(text.split()),
            "char_count": len(text),
            "line_count": len(text.split('\n')),
            "language": self._detect_language(text),
            "sentiment": self._analyze_sentiment(text),
            "keywords": self._extract_keywords(text)
        }
        
        return {
            "success": True,
            "analysis": analysis,
            "processed_at": datetime.now().isoformat()
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """التحقق من المعاملات"""
        errors = []
        
        if "text" not in parameters:
            errors.append("معامل 'text' مطلوب")
        elif not isinstance(parameters["text"], str):
            errors.append("معامل 'text' يجب أن يكون نص")
        
        return len(errors) == 0, errors
    
    def _detect_language(self, text: str) -> str:
        """كشف لغة النص"""
        # تحليل مبسط للغة
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        english_chars = sum(1 for c in text if c.isalpha() and c.isascii())
        
        if arabic_chars > english_chars:
            return "arabic"
        elif english_chars > 0:
            return "english"
        else:
            return "unknown"
    
    def _analyze_sentiment(self, text: str) -> str:
        """تحليل المشاعر"""
        # تحليل مبسط للمشاعر
        positive_words = ["جيد", "ممتاز", "رائع", "good", "excellent", "great"]
        negative_words = ["سيء", "فظيع", "bad", "terrible", "awful"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """استخراج الكلمات المفتاحية"""
        # استخراج مبسط للكلمات المفتاحية
        words = text.split()
        # إزالة الكلمات الشائعة
        stop_words = {"في", "من", "إلى", "على", "the", "and", "or", "but", "is", "are"}
        keywords = [word.strip(".,!?") for word in words 
                   if len(word) > 3 and word.lower() not in stop_words]
        
        # إرجاع أكثر الكلمات تكراراً
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:10]

class DataTransformerTool(BaseTool):
    """أداة تحويل البيانات"""
    
    def __init__(self, name: str = "data_transformer", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
    
    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """تحويل البيانات"""
        data = parameters.get("data")
        transformation_type = parameters.get("type", "normalize")
        
        if data is None:
            return {"success": False, "error": "البيانات مطلوبة"}
        
        try:
            if transformation_type == "normalize":
                result = self._normalize_data(data)
            elif transformation_type == "aggregate":
                result = self._aggregate_data(data)
            elif transformation_type == "filter":
                filter_criteria = parameters.get("filter", {})
                result = self._filter_data(data, filter_criteria)
            elif transformation_type == "sort":
                sort_key = parameters.get("sort_key", "value")
                result = self._sort_data(data, sort_key)
            else:
                return {"success": False, "error": f"نوع التحويل غير مدعوم: {transformation_type}"}
            
            return {
                "success": True,
                "transformed_data": result,
                "transformation_type": transformation_type,
                "processed_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            return {"success": False, "error": f"خطأ في تحويل البيانات: {e}"}
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """التحقق من المعاملات"""
        errors = []
        
        if "data" not in parameters:
            errors.append("معامل 'data' مطلوب")
        
        valid_types = ["normalize", "aggregate", "filter", "sort"]
        if "type" in parameters and parameters["type"] not in valid_types:
            errors.append(f"نوع التحويل يجب أن يكون أحد: {valid_types}")
        
        return len(errors) == 0, errors
    
    def _normalize_data(self, data: Any) -> Any:
        """تطبيع البيانات"""
        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            # تطبيع الأرقام إلى نطاق 0-1
            min_val = min(data)
            max_val = max(data)
            if max_val == min_val:
                return [0.5] * len(data)
            return [(x - min_val) / (max_val - min_val) for x in data]
        
        return data
    
    def _aggregate_data(self, data: Any) -> Dict[str, Any]:
        """تجميع البيانات"""
        if isinstance(data, list):
            if all(isinstance(x, (int, float)) for x in data):
                return {
                    "count": len(data),
                    "sum": sum(data),
                    "average": sum(data) / len(data) if data else 0,
                    "min": min(data) if data else None,
                    "max": max(data) if data else None
                }
            else:
                return {
                    "count": len(data),
                    "types": list(set(type(x).__name__ for x in data))
                }
        
        return {"error": "البيانات يجب أن تكون قائمة"}
    
    def _filter_data(self, data: Any, criteria: Dict[str, Any]) -> Any:
        """تصفية البيانات"""
        if isinstance(data, list):
            if "min_value" in criteria or "max_value" in criteria:
                min_val = criteria.get("min_value", float('-inf'))
                max_val = criteria.get("max_value", float('inf'))
                return [x for x in data if isinstance(x, (int, float)) and min_val <= x <= max_val]
        
        return data
    
    def _sort_data(self, data: Any, sort_key: str) -> Any:
        """ترتيب البيانات"""
        if isinstance(data, list):
            try:
                if sort_key == "value":
                    return sorted(data)
                elif sort_key == "reverse":
                    return sorted(data, reverse=True)
            except TypeError:
                # في حالة عدم إمكانية الترتيب
                pass
        
        return data

class ProcessValidatorTool(BaseTool):
    """أداة التحقق من العمليات"""
    
    def __init__(self, name: str = "process_validator", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
    
    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """التحقق من العملية"""
        process_data = parameters.get("process")
        validation_rules = parameters.get("rules", [])
        
        if not process_data:
            return {"success": False, "error": "بيانات العملية مطلوبة"}
        
        validation_results = []
        
        for rule in validation_rules:
            result = self._apply_validation_rule(process_data, rule)
            validation_results.append(result)
        
        # التحقق العام
        general_validation = self._general_validation(process_data)
        validation_results.extend(general_validation)
        
        is_valid = all(result["passed"] for result in validation_results)
        
        return {
            "success": True,
            "is_valid": is_valid,
            "validation_results": validation_results,
            "validated_at": datetime.now().isoformat()
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """التحقق من المعاملات"""
        errors = []
        
        if "process" not in parameters:
            errors.append("معامل 'process' مطلوب")
        
        return len(errors) == 0, errors
    
    def _apply_validation_rule(self, process_data: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
        """تطبيق قاعدة التحقق"""
        rule_type = rule.get("type")
        field = rule.get("field")
        expected = rule.get("expected")
        
        try:
            if rule_type == "required":
                passed = field in process_data and process_data[field] is not None
                message = f"الحقل '{field}' مطلوب" if not passed else f"الحقل '{field}' موجود"
            
            elif rule_type == "type_check":
                value = process_data.get(field)
                expected_type = expected
                passed = isinstance(value, expected_type) if value is not None else False
                message = f"الحقل '{field}' يجب أن يكون من نوع {expected_type.__name__}"
            
            elif rule_type == "range":
                value = process_data.get(field)
                min_val = expected.get("min")
                max_val = expected.get("max")
                passed = min_val <= value <= max_val if isinstance(value, (int, float)) else False
                message = f"الحقل '{field}' يجب أن يكون بين {min_val} و {max_val}"
            
            else:
                passed = False
                message = f"نوع قاعدة غير معروف: {rule_type}"
            
            return {
                "rule": rule,
                "passed": passed,
                "message": message
            }
        
        except Exception as e:
            return {
                "rule": rule,
                "passed": False,
                "message": f"خطأ في تطبيق القاعدة: {e}"
            }
    
    def _general_validation(self, process_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """التحقق العام"""
        results = []
        
        # التحقق من وجود معرف
        if "id" in process_data:
            results.append({
                "rule": {"type": "general", "check": "id_format"},
                "passed": isinstance(process_data["id"], str) and len(process_data["id"]) > 0,
                "message": "المعرف يجب أن يكون نص غير فارغ"
            })
        
        # التحقق من التوقيت
        if "created_at" in process_data:
            try:
                datetime.fromisoformat(process_data["created_at"])
                results.append({
                    "rule": {"type": "general", "check": "timestamp_format"},
                    "passed": True,
                    "message": "تنسيق التوقيت صحيح"
                })
            except:
                results.append({
                    "rule": {"type": "general", "check": "timestamp_format"},
                    "passed": False,
                    "message": "تنسيق التوقيت غير صحيح"
                })
        
        return results

class ToolRegistry:
    """سجل الأدوات"""
    
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self.loaded_tools: Dict[str, BaseTool] = {}
        self.tool_instances: Dict[str, BaseTool] = {}
        
        # تسجيل الأدوات الأساسية
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """تسجيل الأدوات المدمجة"""
        builtin_tools = [
            ToolDefinition(
                name="text_analyzer",
                type=ToolType.ANALYZER,
                description="أداة تحليل النصوص",
                module_path="__main__",
                class_name="TextAnalyzerTool",
                parameters_schema={"text": {"type": "string", "required": True}},
                dependencies=[],
                version="1.0.0",
                author="Reus Veritas",
                is_active=True
            ),
            ToolDefinition(
                name="data_transformer",
                type=ToolType.TRANSFORMER,
                description="أداة تحويل البيانات",
                module_path="__main__",
                class_name="DataTransformerTool",
                parameters_schema={"data": {"type": "any", "required": True}},
                dependencies=[],
                version="1.0.0",
                author="Reus Veritas",
                is_active=True
            ),
            ToolDefinition(
                name="process_validator",
                type=ToolType.VALIDATOR,
                description="أداة التحقق من العمليات",
                module_path="__main__",
                class_name="ProcessValidatorTool",
                parameters_schema={"process": {"type": "object", "required": True}},
                dependencies=[],
                version="1.0.0",
                author="Reus Veritas",
                is_active=True
            )
        ]
        
        for tool_def in builtin_tools:
            self.tools[tool_def.name] = tool_def
    
    def register_tool(self, tool_definition: ToolDefinition):
        """تسجيل أداة جديدة"""
        self.tools[tool_definition.name] = tool_definition
        logger.info(f"تم تسجيل الأداة: {tool_definition.name}")
    
    def load_tool(self, tool_name: str) -> Optional[BaseTool]:
        """تحميل أداة"""
        if tool_name in self.tool_instances:
            return self.tool_instances[tool_name]
        
        if tool_name not in self.tools:
            logger.error(f"الأداة غير مسجلة: {tool_name}")
            return None
        
        tool_def = self.tools[tool_name]
        
        if not tool_def.is_active:
            logger.error(f"الأداة غير نشطة: {tool_name}")
            return None
        
        try:
            # تحميل الأدوات المدمجة
            if tool_def.module_path == "__main__":
                if tool_def.class_name == "TextAnalyzerTool":
                    tool_instance = TextAnalyzerTool()
                elif tool_def.class_name == "DataTransformerTool":
                    tool_instance = DataTransformerTool()
                elif tool_def.class_name == "ProcessValidatorTool":
                    tool_instance = ProcessValidatorTool()
                else:
                    logger.error(f"فئة الأداة غير معروفة: {tool_def.class_name}")
                    return None
            else:
                # تحميل أدوات خارجية
                module = importlib.import_module(tool_def.module_path)
                tool_class = getattr(module, tool_def.class_name)
                tool_instance = tool_class()
            
            self.tool_instances[tool_name] = tool_instance
            logger.info(f"تم تحميل الأداة: {tool_name}")
            
            return tool_instance
        
        except Exception as e:
            logger.error(f"فشل في تحميل الأداة {tool_name}: {e}")
            return None
    
    def get_available_tools(self) -> List[ToolDefinition]:
        """الحصول على قائمة الأدوات المتاحة"""
        return [tool for tool in self.tools.values() if tool.is_active]
    
    def unload_tool(self, tool_name: str):
        """إلغاء تحميل أداة"""
        if tool_name in self.tool_instances:
            del self.tool_instances[tool_name]
            logger.info(f"تم إلغاء تحميل الأداة: {tool_name}")

class ProcessExecutor:
    """منفذ العمليات"""
    
    def __init__(self, tool_registry: ToolRegistry, max_workers: int = 4):
        self.tool_registry = tool_registry
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.running_processes: Dict[str, DynamicProcess] = {}
        self.process_results: Dict[str, Dict[str, Any]] = {}
    
    async def execute_process(self, process: DynamicProcess) -> Dict[str, Any]:
        """تنفيذ عملية"""
        logger.info(f"بدء تنفيذ العملية: {process.id}")
        
        process.status = ProcessStatus.RUNNING
        process.updated_at = datetime.now()
        self.running_processes[process.id] = process
        
        try:
            # تنفيذ الخطوات بالتسلسل
            for step in process.steps:
                if process.status == ProcessStatus.CANCELLED:
                    break
                
                step_result = await self._execute_step(step, process.context)
                
                if not step_result.get("success", False):
                    step.status = ProcessStatus.FAILED
                    step.error = step_result.get("error", "خطأ غير معروف")
                    process.status = ProcessStatus.FAILED
                    break
                else:
                    step.status = ProcessStatus.COMPLETED
                    step.result = step_result
                    
                    # تحديث السياق بنتيجة الخطوة
                    if "result" in step_result:
                        process.context[f"step_{step.id}_result"] = step_result["result"]
            
            # تحديد حالة العملية النهائية
            if process.status != ProcessStatus.FAILED and process.status != ProcessStatus.CANCELLED:
                process.status = ProcessStatus.COMPLETED
            
            process.updated_at = datetime.now()
            
            # حفظ النتائج
            self.process_results[process.id] = {
                "process_id": process.id,
                "status": process.status.value,
                "steps": [asdict(step) for step in process.steps],
                "context": process.context,
                "completed_at": datetime.now().isoformat()
            }
            
            logger.info(f"انتهاء تنفيذ العملية: {process.id} - الحالة: {process.status.value}")
            
            return self.process_results[process.id]
        
        except Exception as e:
            logger.error(f"خطأ في تنفيذ العملية {process.id}: {e}")
            process.status = ProcessStatus.FAILED
            return {
                "process_id": process.id,
                "status": ProcessStatus.FAILED.value,
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            }
        
        finally:
            if process.id in self.running_processes:
                del self.running_processes[process.id]
    
    async def _execute_step(self, step: ProcessStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """تنفيذ خطوة"""
        logger.info(f"تنفيذ الخطوة: {step.id}")
        
        step.status = ProcessStatus.RUNNING
        step.start_time = datetime.now()
        
        try:
            # تحميل الأداة
            tool = self.tool_registry.load_tool(step.tool_name)
            if not tool:
                return {"success": False, "error": f"فشل في تحميل الأداة: {step.tool_name}"}
            
            # التحقق من المعاملات
            is_valid, errors = tool.validate_parameters(step.parameters)
            if not is_valid:
                return {"success": False, "error": f"معاملات غير صحيحة: {errors}"}
            
            # تنفيذ الأداة
            if step.timeout:
                result = await asyncio.wait_for(
                    tool.execute(step.parameters, context),
                    timeout=step.timeout
                )
            else:
                result = await tool.execute(step.parameters, context)
            
            step.end_time = datetime.now()
            return result
        
        except asyncio.TimeoutError:
            step.end_time = datetime.now()
            return {"success": False, "error": "انتهت المهلة الزمنية"}
        
        except Exception as e:
            step.end_time = datetime.now()
            logger.error(f"خطأ في تنفيذ الخطوة {step.id}: {e}")
            return {"success": False, "error": str(e)}
    
    def cancel_process(self, process_id: str) -> bool:
        """إلغاء عملية"""
        if process_id in self.running_processes:
            self.running_processes[process_id].status = ProcessStatus.CANCELLED
            logger.info(f"تم إلغاء العملية: {process_id}")
            return True
        return False
    
    def get_process_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """الحصول على حالة العملية"""
        if process_id in self.running_processes:
            process = self.running_processes[process_id]
            return {
                "process_id": process.id,
                "status": process.status.value,
                "progress": self._calculate_progress(process),
                "current_step": self._get_current_step(process),
                "updated_at": process.updated_at.isoformat()
            }
        elif process_id in self.process_results:
            return self.process_results[process_id]
        
        return None
    
    def _calculate_progress(self, process: DynamicProcess) -> float:
        """حساب تقدم العملية"""
        if not process.steps:
            return 0.0
        
        completed_steps = sum(1 for step in process.steps 
                            if step.status == ProcessStatus.COMPLETED)
        return completed_steps / len(process.steps)
    
    def _get_current_step(self, process: DynamicProcess) -> Optional[str]:
        """الحصول على الخطوة الحالية"""
        for step in process.steps:
            if step.status == ProcessStatus.RUNNING:
                return step.id
            elif step.status == ProcessStatus.PENDING:
                return step.id
        return None

class IntelligentProcessTransformer:
    """نظام تحويل العمليات الذكية"""
    
    def __init__(self, config_path: str = "process_transformer_config.json"):
        self.config = self._load_config(config_path)
        self.tool_registry = ToolRegistry()
        self.process_executor = ProcessExecutor(self.tool_registry)
        
        self.processes: Dict[str, DynamicProcess] = {}
        self.process_templates: Dict[str, Dict[str, Any]] = {}
        
        # تحميل قوالب العمليات
        self._load_process_templates()
        
        logger.info("تم تهيئة نظام تحويل العمليات الذكية")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """تحميل التكوين"""
        default_config = {
            "max_concurrent_processes": 10,
            "default_timeout": 300,
            "enable_process_caching": True,
            "auto_retry_failed_steps": True,
            "max_retry_attempts": 3
        }
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            logger.warning(f"ملف التكوين غير موجود: {config_path}. استخدام التكوين الافتراضي.")
            return default_config
    
    def _load_process_templates(self):
        """تحميل قوالب العمليات"""
        # قوالب أساسية للعمليات الشائعة
        self.process_templates = {
            "text_analysis_pipeline": {
                "name": "خط أنابيب تحليل النصوص",
                "description": "تحليل شامل للنصوص",
                "steps": [
                    {
                        "name": "تحليل النص",
                        "tool": "text_analyzer",
                        "parameters": {"text": "{input_text}"}
                    },
                    {
                        "name": "التحقق من النتائج",
                        "tool": "process_validator",
                        "parameters": {"process": "{step_1_result}"}
                    }
                ]
            },
            "data_processing_pipeline": {
                "name": "خط أنابيب معالجة البيانات",
                "description": "معالجة وتحويل البيانات",
                "steps": [
                    {
                        "name": "تحويل البيانات",
                        "tool": "data_transformer",
                        "parameters": {"data": "{input_data}", "type": "normalize"}
                    },
                    {
                        "name": "تجميع البيانات",
                        "tool": "data_transformer",
                        "parameters": {"data": "{step_1_result}", "type": "aggregate"}
                    }
                ]
            }
        }
    
    def create_process_from_template(self, template_name: str, 
                                   parameters: Dict[str, Any]) -> Optional[DynamicProcess]:
        """إنشاء عملية من قالب"""
        if template_name not in self.process_templates:
            logger.error(f"القالب غير موجود: {template_name}")
            return None
        
        template = self.process_templates[template_name]
        process_id = str(uuid.uuid4())
        
        # إنشاء خطوات العملية
        steps = []
        for i, step_template in enumerate(template["steps"]):
            step_id = f"step_{i+1}"
            
            # استبدال المتغيرات في المعاملات
            step_parameters = self._substitute_parameters(
                step_template["parameters"], 
                parameters
            )
            
            step = ProcessStep(
                id=step_id,
                name=step_template["name"],
                description=step_template.get("description", ""),
                tool_name=step_template["tool"],
                parameters=step_parameters,
                dependencies=step_template.get("dependencies", []),
                timeout=step_template.get("timeout"),
                retry_count=0,
                status=ProcessStatus.PENDING,
                result=None,
                error=None,
                start_time=None,
                end_time=None
            )
            
            steps.append(step)
        
        # إنشاء العملية
        process = DynamicProcess(
            id=process_id,
            name=template["name"],
            description=template["description"],
            steps=steps,
            priority=ProcessPriority.MEDIUM,
            status=ProcessStatus.PENDING,
            context=parameters.copy(),
            metadata={"template": template_name},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="system"
        )
        
        self.processes[process_id] = process
        logger.info(f"تم إنشاء عملية من القالب {template_name}: {process_id}")
        
        return process
    
    def create_custom_process(self, name: str, description: str, 
                            steps_config: List[Dict[str, Any]]) -> DynamicProcess:
        """إنشاء عملية مخصصة"""
        process_id = str(uuid.uuid4())
        
        # إنشاء خطوات العملية
        steps = []
        for i, step_config in enumerate(steps_config):
            step_id = f"step_{i+1}"
            
            step = ProcessStep(
                id=step_id,
                name=step_config.get("name", f"خطوة {i+1}"),
                description=step_config.get("description", ""),
                tool_name=step_config["tool"],
                parameters=step_config.get("parameters", {}),
                dependencies=step_config.get("dependencies", []),
                timeout=step_config.get("timeout"),
                retry_count=0,
                status=ProcessStatus.PENDING,
                result=None,
                error=None,
                start_time=None,
                end_time=None
            )
            
            steps.append(step)
        
        # إنشاء العملية
        process = DynamicProcess(
            id=process_id,
            name=name,
            description=description,
            steps=steps,
            priority=ProcessPriority.MEDIUM,
            status=ProcessStatus.PENDING,
            context={},
            metadata={"type": "custom"},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="user"
        )
        
        self.processes[process_id] = process
        logger.info(f"تم إنشاء عملية مخصصة: {process_id}")
        
        return process
    
    async def execute_process(self, process_id: str) -> Dict[str, Any]:
        """تنفيذ عملية"""
        if process_id not in self.processes:
            return {"success": False, "error": f"العملية غير موجودة: {process_id}"}
        
        process = self.processes[process_id]
        result = await self.process_executor.execute_process(process)
        
        return result
    
    def _substitute_parameters(self, parameters: Dict[str, Any], 
                             values: Dict[str, Any]) -> Dict[str, Any]:
        """استبدال المتغيرات في المعاملات"""
        substituted = {}
        
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                # استبدال متغير
                var_name = value[1:-1]
                if var_name in values:
                    substituted[key] = values[var_name]
                else:
                    substituted[key] = value  # الاحتفاظ بالقيمة الأصلية
            else:
                substituted[key] = value
        
        return substituted
    
    # واجهات برمجية للتفاعل مع المكونات الأخرى
    
    def get_process(self, process_id: str) -> Optional[DynamicProcess]:
        """الحصول على عملية"""
        return self.processes.get(process_id)
    
    def list_processes(self, status: Optional[ProcessStatus] = None) -> List[DynamicProcess]:
        """قائمة العمليات"""
        processes = list(self.processes.values())
        
        if status:
            processes = [p for p in processes if p.status == status]
        
        return sorted(processes, key=lambda x: x.created_at, reverse=True)
    
    def get_available_templates(self) -> List[str]:
        """الحصول على قوالب العمليات المتاحة"""
        return list(self.process_templates.keys())
    
    def register_tool(self, tool_definition: ToolDefinition):
        """تسجيل أداة جديدة"""
        self.tool_registry.register_tool(tool_definition)
    
    def get_available_tools(self) -> List[ToolDefinition]:
        """الحصول على الأدوات المتاحة"""
        return self.tool_registry.get_available_tools()
    
    def get_statistics(self) -> Dict[str, Any]:
        """إحصائيات النظام"""
        processes = list(self.processes.values())
        
        status_counts = {}
        for status in ProcessStatus:
            status_counts[status.value] = sum(1 for p in processes if p.status == status)
        
        return {
            "total_processes": len(processes),
            "status_distribution": status_counts,
            "available_tools": len(self.tool_registry.get_available_tools()),
            "available_templates": len(self.process_templates),
            "running_processes": len(self.process_executor.running_processes)
        }

def main():
    """دالة الاختبار الرئيسية"""
    async def test_system():
        # إنشاء نظام تحويل العمليات
        transformer = IntelligentProcessTransformer()
        
        try:
            # اختبار إنشاء عملية من قالب
            print("اختبار إنشاء عملية من قالب...")
            
            process = transformer.create_process_from_template(
                "text_analysis_pipeline",
                {"input_text": "هذا نص تجريبي للتحليل. النص يحتوي على معلومات مفيدة."}
            )
            
            if process:
                print(f"تم إنشاء العملية: {process.id}")
                
                # تنفيذ العملية
                print("تنفيذ العملية...")
                result = await transformer.execute_process(process.id)
                
                print(f"نتيجة التنفيذ: {result['status']}")
                
                # عرض الإحصائيات
                print("\nإحصائيات النظام:")
                stats = transformer.get_statistics()
                for key, value in stats.items():
                    print(f"{key}: {value}")
            
            # اختبار إنشاء عملية مخصصة
            print("\nاختبار إنشاء عملية مخصصة...")
            
            custom_process = transformer.create_custom_process(
                "عملية تحليل مخصصة",
                "تحليل وتحويل البيانات",
                [
                    {
                        "name": "تحليل النص",
                        "tool": "text_analyzer",
                        "parameters": {"text": "نص آخر للتحليل"}
                    },
                    {
                        "name": "تحويل البيانات",
                        "tool": "data_transformer",
                        "parameters": {"data": [1, 2, 3, 4, 5], "type": "normalize"}
                    }
                ]
            )
            
            print(f"تم إنشاء العملية المخصصة: {custom_process.id}")
            
            # تنفيذ العملية المخصصة
            custom_result = await transformer.execute_process(custom_process.id)
            print(f"نتيجة العملية المخصصة: {custom_result['status']}")
        
        except Exception as e:
            print(f"خطأ في الاختبار: {e}")
    
    # تشغيل الاختبار
    asyncio.run(test_system())

if __name__ == "__main__":
    main()

