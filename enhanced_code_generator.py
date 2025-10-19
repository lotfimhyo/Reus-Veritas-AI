#!/usr/bin/env python3
"""
Enhanced Code Generator and Optimizer for Reus Veritas
مولد ومحسن الكود المطور لمشروع Reus Veritas

هذا المكون مسؤول عن توليد وتحسين الأكواد البرمجية بشكل تلقائي باستخدام
النماذج اللغوية الضخمة وتقنيات التحليل المتقدمة.

Author: Lotfi mahiddine
Date: 2025
"""

import ast
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import openai
import hashlib

# إعداد نظام السجلات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('code_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CodeLanguage(Enum):
    """لغات البرمجة المدعومة"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    SQL = "sql"
    HTML = "html"
    CSS = "css"

class CodeQuality(Enum):
    """مستويات جودة الكود"""
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    POOR = 2
    CRITICAL = 1

class OptimizationType(Enum):
    """أنواع التحسين"""
    PERFORMANCE = "performance"
    READABILITY = "readability"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    MEMORY = "memory"
    ALL = "all"

@dataclass
class CodeRequest:
    """طلب توليد كود"""
    id: str
    description: str
    language: CodeLanguage
    requirements: List[str]
    constraints: List[str]
    context: Optional[str]
    priority: int
    created_at: datetime

@dataclass
class GeneratedCode:
    """كود مولد"""
    id: str
    request_id: str
    code: str
    language: CodeLanguage
    quality_score: float
    explanation: str
    dependencies: List[str]
    test_cases: List[str]
    documentation: str
    created_at: datetime

@dataclass
class CodeAnalysis:
    """تحليل الكود"""
    complexity: int
    maintainability: float
    security_score: float
    performance_score: float
    readability_score: float
    issues: List[str]
    suggestions: List[str]

class LLMCodeInterface:
    """واجهة النماذج اللغوية لتوليد الكود"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.model = "gpt-3.5-turbo"
    
    def generate_code(self, request: CodeRequest) -> str:
        """توليد كود بناءً على الطلب"""
        prompt = self._build_code_prompt(request)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(request.language)},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"خطأ في توليد الكود: {e}")
            return f"// خطأ في توليد الكود: {e}"
    
    def optimize_code(self, code: str, language: CodeLanguage, 
                     optimization_type: OptimizationType) -> str:
        """تحسين كود موجود"""
        prompt = f"""
        حسن الكود التالي بلغة {language.value} مع التركيز على {optimization_type.value}:
        
        الكود الأصلي:
        ```{language.value}
        {code}
        ```
        
        متطلبات التحسين:
        - تحسين {optimization_type.value}
        - الحفاظ على الوظيفة الأصلية
        - إضافة تعليقات توضيحية
        - اتباع أفضل الممارسات
        
        أرجع الكود المحسن فقط مع شرح مختصر للتحسينات.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"أنت خبير في تحسين كود {language.value}"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.2
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"خطأ في تحسين الكود: {e}")
            return code  # إرجاع الكود الأصلي في حالة الخطأ
    
    def analyze_code(self, code: str, language: CodeLanguage) -> Dict[str, Any]:
        """تحليل جودة الكود"""
        prompt = f"""
        حلل الكود التالي بلغة {language.value} وقدم تقييماً شاملاً:
        
        ```{language.value}
        {code}
        ```
        
        أرجع التحليل في شكل JSON يحتوي على:
        - complexity: مستوى التعقيد (1-10)
        - maintainability: قابلية الصيانة (0-1)
        - security_score: نقاط الأمان (0-1)
        - performance_score: نقاط الأداء (0-1)
        - readability_score: نقاط القابلية للقراءة (0-1)
        - issues: قائمة بالمشاكل المكتشفة
        - suggestions: قائمة بالاقتراحات للتحسين
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "أنت محلل كود خبير"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            analysis_text = response.choices[0].message.content
            
            # محاولة تحليل الاستجابة كـ JSON
            try:
                return json.loads(analysis_text)
            except:
                # في حالة فشل التحليل، إرجاع تحليل أساسي
                return {
                    "complexity": 5,
                    "maintainability": 0.7,
                    "security_score": 0.8,
                    "performance_score": 0.7,
                    "readability_score": 0.8,
                    "issues": ["تعذر تحليل الكود تلقائياً"],
                    "suggestions": ["مراجعة يدوية مطلوبة"]
                }
        
        except Exception as e:
            logger.error(f"خطأ في تحليل الكود: {e}")
            return {
                "complexity": 0,
                "maintainability": 0.0,
                "security_score": 0.0,
                "performance_score": 0.0,
                "readability_score": 0.0,
                "issues": [f"خطأ في التحليل: {e}"],
                "suggestions": []
            }
    
    def generate_tests(self, code: str, language: CodeLanguage) -> List[str]:
        """توليد اختبارات للكود"""
        prompt = f"""
        أنشئ اختبارات شاملة للكود التالي بلغة {language.value}:
        
        ```{language.value}
        {code}
        ```
        
        أرجع قائمة بحالات الاختبار تغطي:
        - الحالات العادية
        - الحالات الحدية
        - حالات الخطأ
        - اختبارات الأداء
        
        كل اختبار يجب أن يكون كود قابل للتنفيذ.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"أنت خبير في كتابة اختبارات {language.value}"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            tests_text = response.choices[0].message.content
            
            # استخراج الاختبارات من النص
            test_blocks = re.findall(r'```(?:' + language.value + r')?\n(.*?)\n```', 
                                   tests_text, re.DOTALL)
            
            return test_blocks if test_blocks else [tests_text]
        
        except Exception as e:
            logger.error(f"خطأ في توليد الاختبارات: {e}")
            return []
    
    def _build_code_prompt(self, request: CodeRequest) -> str:
        """بناء prompt لتوليد الكود"""
        prompt = f"""
        أنشئ كود بلغة {request.language.value} للمتطلبات التالية:
        
        الوصف: {request.description}
        
        المتطلبات:
        """
        
        for req in request.requirements:
            prompt += f"- {req}\n"
        
        if request.constraints:
            prompt += "\nالقيود:\n"
            for constraint in request.constraints:
                prompt += f"- {constraint}\n"
        
        if request.context:
            prompt += f"\nالسياق: {request.context}\n"
        
        prompt += """
        
        يجب أن يكون الكود:
        - واضح ومقروء
        - محسن للأداء
        - آمن
        - مع تعليقات توضيحية
        - يتبع أفضل الممارسات
        
        أرجع الكود فقط مع شرح مختصر.
        """
        
        return prompt
    
    def _get_system_prompt(self, language: CodeLanguage) -> str:
        """الحصول على system prompt حسب اللغة"""
        prompts = {
            CodeLanguage.PYTHON: "أنت خبير في برمجة Python. اكتب كود Python نظيف وفعال.",
            CodeLanguage.JAVASCRIPT: "أنت خبير في JavaScript. اكتب كود JavaScript حديث وفعال.",
            CodeLanguage.JAVA: "أنت خبير في Java. اكتب كود Java محسن ويتبع أفضل الممارسات.",
            CodeLanguage.CPP: "أنت خبير في C++. اكتب كود C++ محسن وآمن.",
            CodeLanguage.SQL: "أنت خبير في SQL. اكتب استعلامات SQL محسنة وآمنة."
        }
        
        return prompts.get(language, "أنت مطور برمجيات خبير. اكتب كود عالي الجودة.")

class CodeValidator:
    """مدقق صحة الكود"""
    
    def __init__(self):
        self.validators = {
            CodeLanguage.PYTHON: self._validate_python,
            CodeLanguage.JAVASCRIPT: self._validate_javascript,
            CodeLanguage.SQL: self._validate_sql
        }
    
    def validate(self, code: str, language: CodeLanguage) -> Tuple[bool, List[str]]:
        """التحقق من صحة الكود"""
        validator = self.validators.get(language)
        
        if validator:
            return validator(code)
        else:
            # للغات غير المدعومة، إرجاع صحيح افتراضياً
            return True, []
    
    def _validate_python(self, code: str) -> Tuple[bool, List[str]]:
        """التحقق من صحة كود Python"""
        errors = []
        
        try:
            # تحليل الكود للتحقق من صحة البناء
            ast.parse(code)
            
            # فحوصات إضافية
            if "eval(" in code or "exec(" in code:
                errors.append("استخدام eval() أو exec() قد يكون خطراً أمنياً")
            
            if "import os" in code and "system(" in code:
                errors.append("استخدام os.system() قد يكون خطراً أمنياً")
            
            return len(errors) == 0, errors
        
        except SyntaxError as e:
            errors.append(f"خطأ في البناء: {e}")
            return False, errors
        except Exception as e:
            errors.append(f"خطأ في التحليل: {e}")
            return False, errors
    
    def _validate_javascript(self, code: str) -> Tuple[bool, List[str]]:
        """التحقق من صحة كود JavaScript"""
        errors = []
        
        # فحوصات أمنية أساسية
        dangerous_patterns = [
            r"eval\s*\(",
            r"document\.write\s*\(",
            r"innerHTML\s*=",
            r"outerHTML\s*="
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                errors.append(f"نمط خطر مكتشف: {pattern}")
        
        return len(errors) == 0, errors
    
    def _validate_sql(self, code: str) -> Tuple[bool, List[str]]:
        """التحقق من صحة كود SQL"""
        errors = []
        
        # فحص SQL injection
        dangerous_patterns = [
            r";\s*DROP\s+TABLE",
            r";\s*DELETE\s+FROM",
            r"UNION\s+SELECT",
            r"--\s*$"
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                errors.append(f"نمط SQL خطر مكتشف: {pattern}")
        
        return len(errors) == 0, errors

class CodeExecutor:
    """منفذ الكود للاختبار"""
    
    def __init__(self):
        self.timeout = 30  # مهلة زمنية للتنفيذ
    
    def execute_python(self, code: str) -> Dict[str, Any]:
        """تنفيذ كود Python"""
        try:
            # إنشاء ملف مؤقت
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # تنفيذ الكود
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # حذف الملف المؤقت
            os.unlink(temp_file)
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "return_code": result.returncode
            }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "انتهت المهلة الزمنية للتنفيذ",
                "return_code": -1
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"خطأ في التنفيذ: {e}",
                "return_code": -1
            }

class EnhancedCodeGenerator:
    """مولد ومحسن الكود المطور"""
    
    def __init__(self, config_path: str = "code_generator_config.json"):
        self.config = self._load_config(config_path)
        self.llm_interface = LLMCodeInterface()
        self.validator = CodeValidator()
        self.executor = CodeExecutor()
        
        self.generated_codes = {}
        self.code_history = []
        
        logger.info("تم تهيئة مولد ومحسن الكود المطور")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """تحميل التكوين"""
        default_config = {
            "max_code_length": 5000,
            "default_language": "python",
            "enable_validation": True,
            "enable_testing": True,
            "optimization_level": "balanced",
            "security_checks": True
        }
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            logger.warning(f"ملف التكوين غير موجود: {config_path}. استخدام التكوين الافتراضي.")
            return default_config
    
    def generate_code(self, description: str, language: str = "python",
                     requirements: Optional[List[str]] = None,
                     constraints: Optional[List[str]] = None,
                     context: Optional[str] = None) -> GeneratedCode:
        """توليد كود جديد"""
        
        # إنشاء طلب توليد الكود
        request_id = f"req_{int(time.time())}_{hashlib.md5(description.encode()).hexdigest()[:8]}"
        
        request = CodeRequest(
            id=request_id,
            description=description,
            language=CodeLanguage(language.lower()),
            requirements=requirements or [],
            constraints=constraints or [],
            context=context,
            priority=1,
            created_at=datetime.now()
        )
        
        logger.info(f"بدء توليد كود للطلب: {request_id}")
        
        # توليد الكود باستخدام LLM
        raw_code = self.llm_interface.generate_code(request)
        
        # استخراج الكود من الاستجابة
        code = self._extract_code_from_response(raw_code, request.language)
        
        # التحقق من صحة الكود
        is_valid, validation_errors = self.validator.validate(code, request.language)
        
        if not is_valid and self.config["enable_validation"]:
            logger.warning(f"الكود المولد غير صحيح: {validation_errors}")
            # محاولة إصلاح الكود
            code = self._fix_code_issues(code, validation_errors, request.language)
        
        # تحليل جودة الكود
        analysis = self.llm_interface.analyze_code(code, request.language)
        quality_score = self._calculate_quality_score(analysis)
        
        # توليد الاختبارات
        test_cases = []
        if self.config["enable_testing"]:
            test_cases = self.llm_interface.generate_tests(code, request.language)
        
        # إنشاء الوثائق
        documentation = self._generate_documentation(code, request)
        
        # إنشاء كائن الكود المولد
        generated_code = GeneratedCode(
            id=f"code_{int(time.time())}",
            request_id=request_id,
            code=code,
            language=request.language,
            quality_score=quality_score,
            explanation=self._extract_explanation_from_response(raw_code),
            dependencies=self._extract_dependencies(code, request.language),
            test_cases=test_cases,
            documentation=documentation,
            created_at=datetime.now()
        )
        
        # حفظ الكود المولد
        self.generated_codes[generated_code.id] = generated_code
        self.code_history.append(generated_code.id)
        
        logger.info(f"تم توليد الكود بنجاح: {generated_code.id}")
        
        return generated_code
    
    def optimize_code(self, code_id: str, optimization_type: str = "all") -> GeneratedCode:
        """تحسين كود موجود"""
        
        if code_id not in self.generated_codes:
            raise ValueError(f"الكود غير موجود: {code_id}")
        
        original_code = self.generated_codes[code_id]
        
        logger.info(f"بدء تحسين الكود: {code_id}")
        
        # تحسين الكود باستخدام LLM
        optimized_code = self.llm_interface.optimize_code(
            original_code.code,
            original_code.language,
            OptimizationType(optimization_type)
        )
        
        # استخراج الكود المحسن
        code = self._extract_code_from_response(optimized_code, original_code.language)
        
        # تحليل الكود المحسن
        analysis = self.llm_interface.analyze_code(code, original_code.language)
        quality_score = self._calculate_quality_score(analysis)
        
        # إنشاء كود محسن جديد
        optimized_generated_code = GeneratedCode(
            id=f"opt_{int(time.time())}",
            request_id=original_code.request_id,
            code=code,
            language=original_code.language,
            quality_score=quality_score,
            explanation=f"نسخة محسنة من {code_id} - تحسين {optimization_type}",
            dependencies=self._extract_dependencies(code, original_code.language),
            test_cases=original_code.test_cases,
            documentation=original_code.documentation,
            created_at=datetime.now()
        )
        
        # حفظ الكود المحسن
        self.generated_codes[optimized_generated_code.id] = optimized_generated_code
        self.code_history.append(optimized_generated_code.id)
        
        logger.info(f"تم تحسين الكود بنجاح: {optimized_generated_code.id}")
        
        return optimized_generated_code
    
    def test_code(self, code_id: str) -> Dict[str, Any]:
        """اختبار كود مولد"""
        
        if code_id not in self.generated_codes:
            raise ValueError(f"الكود غير موجود: {code_id}")
        
        generated_code = self.generated_codes[code_id]
        
        logger.info(f"بدء اختبار الكود: {code_id}")
        
        results = {
            "code_id": code_id,
            "language": generated_code.language.value,
            "validation": {},
            "execution": {},
            "test_cases": []
        }
        
        # التحقق من صحة الكود
        is_valid, validation_errors = self.validator.validate(
            generated_code.code, 
            generated_code.language
        )
        
        results["validation"] = {
            "is_valid": is_valid,
            "errors": validation_errors
        }
        
        # تنفيذ الكود (للغة Python فقط حالياً)
        if generated_code.language == CodeLanguage.PYTHON:
            execution_result = self.executor.execute_python(generated_code.code)
            results["execution"] = execution_result
        
        # تنفيذ اختبارات الوحدة
        for i, test_case in enumerate(generated_code.test_cases):
            if generated_code.language == CodeLanguage.PYTHON:
                test_result = self.executor.execute_python(test_case)
                results["test_cases"].append({
                    "test_id": i,
                    "result": test_result
                })
        
        logger.info(f"انتهاء اختبار الكود: {code_id}")
        
        return results
    
    def _extract_code_from_response(self, response: str, language: CodeLanguage) -> str:
        """استخراج الكود من استجابة LLM"""
        # البحث عن كتل الكود
        code_blocks = re.findall(r'```(?:' + language.value + r')?\n(.*?)\n```', 
                                response, re.DOTALL)
        
        if code_blocks:
            return code_blocks[0].strip()
        
        # إذا لم توجد كتل كود، البحث عن أسطر تبدأ بمسافات
        lines = response.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            
            if in_code_block or line.startswith('    ') or line.startswith('\t'):
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        # كحل أخير، إرجاع الاستجابة كاملة
        return response.strip()
    
    def _extract_explanation_from_response(self, response: str) -> str:
        """استخراج الشرح من استجابة LLM"""
        # إزالة كتل الكود من الاستجابة للحصول على الشرح
        explanation = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
        return explanation.strip()
    
    def _extract_dependencies(self, code: str, language: CodeLanguage) -> List[str]:
        """استخراج التبعيات من الكود"""
        dependencies = []
        
        if language == CodeLanguage.PYTHON:
            # البحث عن import statements
            import_patterns = [
                r'import\s+(\w+)',
                r'from\s+(\w+)\s+import',
                r'import\s+(\w+\.\w+)'
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, code)
                dependencies.extend(matches)
        
        elif language == CodeLanguage.JAVASCRIPT:
            # البحث عن require أو import
            js_patterns = [
                r'require\([\'"]([^\'"]+)[\'"]\)',
                r'import.*from\s+[\'"]([^\'"]+)[\'"]'
            ]
            
            for pattern in js_patterns:
                matches = re.findall(pattern, code)
                dependencies.extend(matches)
        
        return list(set(dependencies))  # إزالة التكرارات
    
    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """حساب نقاط جودة الكود"""
        try:
            # حساب متوسط النقاط
            scores = [
                analysis.get("maintainability", 0.5),
                analysis.get("security_score", 0.5),
                analysis.get("performance_score", 0.5),
                analysis.get("readability_score", 0.5)
            ]
            
            # تقليل النقاط بناءً على التعقيد
            complexity = analysis.get("complexity", 5)
            complexity_penalty = max(0, (complexity - 5) * 0.1)
            
            # تقليل النقاط بناءً على عدد المشاكل
            issues_count = len(analysis.get("issues", []))
            issues_penalty = min(0.3, issues_count * 0.05)
            
            base_score = sum(scores) / len(scores)
            final_score = max(0, base_score - complexity_penalty - issues_penalty)
            
            return round(final_score, 2)
        
        except Exception as e:
            logger.error(f"خطأ في حساب نقاط الجودة: {e}")
            return 0.5
    
    def _fix_code_issues(self, code: str, errors: List[str], 
                        language: CodeLanguage) -> str:
        """محاولة إصلاح مشاكل الكود"""
        
        prompt = f"""
        الكود التالي بلغة {language.value} يحتوي على مشاكل. أصلحها:
        
        الكود:
        ```{language.value}
        {code}
        ```
        
        المشاكل المكتشفة:
        {chr(10).join(f"- {error}" for error in errors)}
        
        أرجع الكود المصحح فقط.
        """
        
        try:
            fixed_code = self.llm_interface.generate_code(
                CodeRequest(
                    id="fix_request",
                    description=prompt,
                    language=language,
                    requirements=[],
                    constraints=[],
                    context=None,
                    priority=1,
                    created_at=datetime.now()
                )
            )
            
            return self._extract_code_from_response(fixed_code, language)
        
        except Exception as e:
            logger.error(f"فشل في إصلاح الكود: {e}")
            return code  # إرجاع الكود الأصلي
    
    def _generate_documentation(self, code: str, request: CodeRequest) -> str:
        """توليد وثائق للكود"""
        
        prompt = f"""
        أنشئ وثائق شاملة للكود التالي:
        
        ```{request.language.value}
        {code}
        ```
        
        الوثائق يجب أن تتضمن:
        - وصف عام للكود
        - شرح الوظائف الرئيسية
        - معاملات الدخل والخرج
        - أمثلة على الاستخدام
        - ملاحظات مهمة
        """
        
        try:
            documentation = self.llm_interface.generate_response(prompt)
            return documentation
        
        except Exception as e:
            logger.error(f"فشل في توليد الوثائق: {e}")
            return f"وثائق للكود المولد في {request.created_at}"
    
    # واجهات برمجية للتفاعل مع المكونات الأخرى
    
    def get_code(self, code_id: str) -> Optional[GeneratedCode]:
        """الحصول على كود مولد"""
        return self.generated_codes.get(code_id)
    
    def list_codes(self, language: Optional[str] = None) -> List[GeneratedCode]:
        """قائمة بجميع الأكواد المولدة"""
        codes = list(self.generated_codes.values())
        
        if language:
            codes = [code for code in codes if code.language.value == language.lower()]
        
        return sorted(codes, key=lambda x: x.created_at, reverse=True)
    
    def delete_code(self, code_id: str) -> bool:
        """حذف كود مولد"""
        if code_id in self.generated_codes:
            del self.generated_codes[code_id]
            if code_id in self.code_history:
                self.code_history.remove(code_id)
            logger.info(f"تم حذف الكود: {code_id}")
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """إحصائيات مولد الكود"""
        codes = list(self.generated_codes.values())
        
        if not codes:
            return {
                "total_codes": 0,
                "languages": {},
                "average_quality": 0.0,
                "total_lines": 0
            }
        
        # إحصائيات اللغات
        languages = {}
        for code in codes:
            lang = code.language.value
            languages[lang] = languages.get(lang, 0) + 1
        
        # متوسط الجودة
        avg_quality = sum(code.quality_score for code in codes) / len(codes)
        
        # إجمالي الأسطر
        total_lines = sum(len(code.code.split('\n')) for code in codes)
        
        return {
            "total_codes": len(codes),
            "languages": languages,
            "average_quality": round(avg_quality, 2),
            "total_lines": total_lines,
            "latest_code": codes[-1].id if codes else None
        }

def main():
    """دالة الاختبار الرئيسية"""
    # إنشاء مولد الكود
    generator = EnhancedCodeGenerator()
    
    try:
        # اختبار توليد كود Python
        print("اختبار توليد كود Python...")
        
        python_code = generator.generate_code(
            description="دالة لحساب العدد الأولي",
            language="python",
            requirements=[
                "دالة تتحقق من كون العدد أولي",
                "دالة لإيجاد جميع الأعداد الأولية حتى رقم معين",
                "معالجة الأخطاء للمدخلات غير الصحيحة"
            ],
            constraints=[
                "استخدام خوارزمية فعالة",
                "تجنب الحلقات غير الضرورية"
            ]
        )
        
        print(f"تم توليد الكود: {python_code.id}")
        print(f"جودة الكود: {python_code.quality_score}")
        print(f"الكود:\n{python_code.code}")
        
        # اختبار تحسين الكود
        print("\nاختبار تحسين الكود...")
        
        optimized_code = generator.optimize_code(python_code.id, "performance")
        print(f"تم تحسين الكود: {optimized_code.id}")
        print(f"جودة الكود المحسن: {optimized_code.quality_score}")
        
        # اختبار الكود
        print("\nاختبار الكود...")
        
        test_results = generator.test_code(python_code.id)
        print(f"نتائج الاختبار: {test_results['validation']['is_valid']}")
        
        # إحصائيات
        print("\nإحصائيات مولد الكود:")
        stats = generator.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"خطأ في الاختبار: {e}")

if __name__ == "__main__":
    main()

