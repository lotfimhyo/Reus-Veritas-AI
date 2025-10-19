#!/usr/bin/env python3
"""
Research & Evolution Engine for Reus Veritas
البنية البحثية والتطورية R&E لمشروع Reus Veritas

هذا المكون مسؤول عن البحث والتطوير المستمر للنظام، الحوسبة الموزعة،
التعلم التعاوني، والتغذية المستمرة من مصادر معرفية مفتوحة.

Author: Lotfi mahiddine
Date: 2025
"""

import json
import logging
import asyncio
import threading
import time
import requests
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import queue
import sqlite3
import numpy as np
from collections import defaultdict
import xml.etree.ElementTree as ET

# إعداد نظام السجلات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_evolution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ResearchType(Enum):
    """أنواع البحث"""
    LITERATURE_REVIEW = "literature_review"
    DATA_MINING = "data_mining"
    PATTERN_ANALYSIS = "pattern_analysis"
    TREND_DETECTION = "trend_detection"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    INNOVATION_DISCOVERY = "innovation_discovery"

class EvolutionStrategy(Enum):
    """استراتيجيات التطور"""
    GENETIC_ALGORITHM = "genetic_algorithm"
    NEURAL_EVOLUTION = "neural_evolution"
    SWARM_OPTIMIZATION = "swarm_optimization"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ADAPTIVE_LEARNING = "adaptive_learning"

class KnowledgeSource(Enum):
    """مصادر المعرفة"""
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    GITHUB = "github"
    STACKOVERFLOW = "stackoverflow"
    WIKIPEDIA = "wikipedia"
    NEWS_APIS = "news_apis"
    RESEARCH_PAPERS = "research_papers"
    OPEN_DATASETS = "open_datasets"

@dataclass
class ResearchTask:
    """مهمة بحثية"""
    id: str
    title: str
    description: str
    research_type: ResearchType
    keywords: List[str]
    sources: List[KnowledgeSource]
    priority: int
    status: str
    created_at: datetime
    deadline: Optional[datetime]
    assigned_nodes: List[str]
    results: Optional[Dict[str, Any]]

@dataclass
class EvolutionExperiment:
    """تجربة تطورية"""
    id: str
    name: str
    strategy: EvolutionStrategy
    parameters: Dict[str, Any]
    population_size: int
    generations: int
    fitness_function: str
    current_generation: int
    best_solution: Optional[Dict[str, Any]]
    fitness_history: List[float]
    status: str
    created_at: datetime

@dataclass
class KnowledgeItem:
    """عنصر معرفة"""
    id: str
    title: str
    content: str
    source: KnowledgeSource
    url: Optional[str]
    authors: List[str]
    publication_date: Optional[datetime]
    keywords: List[str]
    relevance_score: float
    quality_score: float
    extracted_at: datetime
    metadata: Dict[str, Any]

@dataclass
class ComputeNode:
    """عقدة حاسوبية"""
    id: str
    name: str
    address: str
    port: int
    capabilities: List[str]
    cpu_cores: int
    memory_gb: float
    gpu_available: bool
    status: str
    last_heartbeat: datetime
    current_tasks: List[str]
    performance_metrics: Dict[str, float]

class KnowledgeExtractor:
    """مستخرج المعرفة"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Reus-Veritas-Research-Engine/1.0'
        })
        
        # معدلات الطلبات لتجنب الحظر
        self.rate_limits = {
            KnowledgeSource.ARXIV: 3,  # طلبات في الثانية
            KnowledgeSource.GITHUB: 10,
            KnowledgeSource.WIKIPEDIA: 10
        }
        
        self.last_request_time = defaultdict(float)
    
    async def extract_from_arxiv(self, keywords: List[str], max_results: int = 50) -> List[KnowledgeItem]:
        """استخراج من arXiv"""
        try:
            # بناء استعلام البحث
            query = " AND ".join(keywords)
            url = f"http://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            # احترام معدل الطلبات
            await self._respect_rate_limit(KnowledgeSource.ARXIV)
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # تحليل XML
            root = ET.fromstring(response.content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            
            knowledge_items = []
            
            for entry in root.findall('atom:entry', namespace):
                try:
                    title = entry.find('atom:title', namespace).text.strip()
                    summary = entry.find('atom:summary', namespace).text.strip()
                    
                    # استخراج المؤلفين
                    authors = []
                    for author in entry.findall('atom:author', namespace):
                        name = author.find('atom:name', namespace)
                        if name is not None:
                            authors.append(name.text)
                    
                    # تاريخ النشر
                    published = entry.find('atom:published', namespace)
                    pub_date = None
                    if published is not None:
                        try:
                            pub_date = datetime.fromisoformat(published.text.replace('Z', '+00:00'))
                        except:
                            pass
                    
                    # الرابط
                    link = entry.find('atom:id', namespace)
                    url = link.text if link is not None else None
                    
                    # حساب نقاط الصلة
                    relevance_score = self._calculate_relevance(title + " " + summary, keywords)
                    
                    knowledge_item = KnowledgeItem(
                        id=hashlib.md5(f"arxiv_{title}".encode()).hexdigest(),
                        title=title,
                        content=summary,
                        source=KnowledgeSource.ARXIV,
                        url=url,
                        authors=authors,
                        publication_date=pub_date,
                        keywords=keywords,
                        relevance_score=relevance_score,
                        quality_score=0.8,  # arXiv عادة عالي الجودة
                        extracted_at=datetime.now(),
                        metadata={"source_type": "academic_paper"}
                    )
                    
                    knowledge_items.append(knowledge_item)
                
                except Exception as e:
                    logger.warning(f"خطأ في معالجة مدخل arXiv: {e}")
                    continue
            
            logger.info(f"تم استخراج {len(knowledge_items)} عنصر من arXiv")
            return knowledge_items
        
        except Exception as e:
            logger.error(f"فشل في استخراج من arXiv: {e}")
            return []
    
    async def extract_from_github(self, keywords: List[str], max_results: int = 30) -> List[KnowledgeItem]:
        """استخراج من GitHub"""
        try:
            # بناء استعلام البحث
            query = " ".join(keywords)
            url = "https://api.github.com/search/repositories"
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': max_results
            }
            
            await self._respect_rate_limit(KnowledgeSource.GITHUB)
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            knowledge_items = []
            
            for repo in data.get('items', []):
                try:
                    title = repo['full_name']
                    description = repo.get('description', '')
                    
                    # معلومات إضافية
                    stars = repo.get('stargazers_count', 0)
                    language = repo.get('language', 'Unknown')
                    
                    # حساب نقاط الجودة بناءً على النجوم والنشاط
                    quality_score = min(1.0, stars / 1000.0)  # تطبيع بناءً على النجوم
                    
                    # حساب نقاط الصلة
                    relevance_score = self._calculate_relevance(title + " " + description, keywords)
                    
                    knowledge_item = KnowledgeItem(
                        id=hashlib.md5(f"github_{title}".encode()).hexdigest(),
                        title=title,
                        content=description,
                        source=KnowledgeSource.GITHUB,
                        url=repo['html_url'],
                        authors=[repo['owner']['login']],
                        publication_date=datetime.fromisoformat(repo['created_at'].replace('Z', '+00:00')),
                        keywords=keywords,
                        relevance_score=relevance_score,
                        quality_score=quality_score,
                        extracted_at=datetime.now(),
                        metadata={
                            "stars": stars,
                            "language": language,
                            "forks": repo.get('forks_count', 0)
                        }
                    )
                    
                    knowledge_items.append(knowledge_item)
                
                except Exception as e:
                    logger.warning(f"خطأ في معالجة مستودع GitHub: {e}")
                    continue
            
            logger.info(f"تم استخراج {len(knowledge_items)} عنصر من GitHub")
            return knowledge_items
        
        except Exception as e:
            logger.error(f"فشل في استخراج من GitHub: {e}")
            return []
    
    async def extract_from_wikipedia(self, keywords: List[str], max_results: int = 20) -> List[KnowledgeItem]:
        """استخراج من Wikipedia"""
        try:
            knowledge_items = []
            
            for keyword in keywords[:max_results]:
                try:
                    await self._respect_rate_limit(KnowledgeSource.WIKIPEDIA)
                    
                    # البحث عن المقال
                    search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + keyword.replace(" ", "_")
                    response = self.session.get(search_url, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        title = data.get('title', keyword)
                        extract = data.get('extract', '')
                        
                        if extract:
                            relevance_score = self._calculate_relevance(title + " " + extract, keywords)
                            
                            knowledge_item = KnowledgeItem(
                                id=hashlib.md5(f"wikipedia_{title}".encode()).hexdigest(),
                                title=title,
                                content=extract,
                                source=KnowledgeSource.WIKIPEDIA,
                                url=data.get('content_urls', {}).get('desktop', {}).get('page'),
                                authors=["Wikipedia Contributors"],
                                publication_date=None,
                                keywords=keywords,
                                relevance_score=relevance_score,
                                quality_score=0.7,  # Wikipedia عادة موثوق
                                extracted_at=datetime.now(),
                                metadata={"page_id": data.get('pageid')}
                            )
                            
                            knowledge_items.append(knowledge_item)
                
                except Exception as e:
                    logger.warning(f"خطأ في استخراج من Wikipedia للكلمة {keyword}: {e}")
                    continue
            
            logger.info(f"تم استخراج {len(knowledge_items)} عنصر من Wikipedia")
            return knowledge_items
        
        except Exception as e:
            logger.error(f"فشل في استخراج من Wikipedia: {e}")
            return []
    
    def _calculate_relevance(self, text: str, keywords: List[str]) -> float:
        """حساب نقاط الصلة"""
        text_lower = text.lower()
        keyword_matches = 0
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                keyword_matches += 1
        
        return keyword_matches / len(keywords) if keywords else 0.0
    
    async def _respect_rate_limit(self, source: KnowledgeSource):
        """احترام معدل الطلبات"""
        if source in self.rate_limits:
            min_interval = 1.0 / self.rate_limits[source]
            elapsed = time.time() - self.last_request_time[source]
            
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            
            self.last_request_time[source] = time.time()

class DistributedComputing:
    """نظام الحوسبة الموزعة"""
    
    def __init__(self):
        self.nodes: Dict[str, ComputeNode] = {}
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # إضافة العقدة المحلية
        self._add_local_node()
    
    def _add_local_node(self):
        """إضافة العقدة المحلية"""
        local_node = ComputeNode(
            id="local_node",
            name="Local Compute Node",
            address="127.0.0.1",
            port=8080,
            capabilities=["cpu_compute", "data_processing", "ml_training"],
            cpu_cores=mp.cpu_count(),
            memory_gb=8.0,  # افتراضي
            gpu_available=False,
            status="active",
            last_heartbeat=datetime.now(),
            current_tasks=[],
            performance_metrics={"cpu_usage": 0.0, "memory_usage": 0.0}
        )
        
        self.nodes[local_node.id] = local_node
        logger.info(f"تم إضافة العقدة المحلية: {local_node.cpu_cores} cores")
    
    def register_node(self, node: ComputeNode):
        """تسجيل عقدة جديدة"""
        self.nodes[node.id] = node
        logger.info(f"تم تسجيل عقدة جديدة: {node.name}")
    
    def distribute_task(self, task_func: Callable, task_data: Any, node_id: Optional[str] = None) -> str:
        """توزيع مهمة على العقد"""
        task_id = hashlib.md5(f"{task_func.__name__}_{time.time()}".encode()).hexdigest()
        
        # اختيار العقدة
        if node_id and node_id in self.nodes:
            target_node = self.nodes[node_id]
        else:
            # اختيار أفضل عقدة متاحة
            target_node = self._select_best_node()
        
        if not target_node:
            logger.error("لا توجد عقد متاحة")
            return None
        
        # إضافة المهمة إلى العقدة
        target_node.current_tasks.append(task_id)
        
        # تنفيذ المهمة
        future = self.executor.submit(self._execute_distributed_task, task_func, task_data, task_id, target_node.id)
        
        logger.info(f"تم توزيع المهمة {task_id} على العقدة {target_node.name}")
        
        return task_id
    
    def _select_best_node(self) -> Optional[ComputeNode]:
        """اختيار أفضل عقدة متاحة"""
        available_nodes = [node for node in self.nodes.values() if node.status == "active"]
        
        if not available_nodes:
            return None
        
        # ترتيب العقد حسب الحمولة (أقل مهام حالية)
        available_nodes.sort(key=lambda n: len(n.current_tasks))
        
        return available_nodes[0]
    
    def _execute_distributed_task(self, task_func: Callable, task_data: Any, task_id: str, node_id: str):
        """تنفيذ مهمة موزعة"""
        try:
            start_time = time.time()
            result = task_func(task_data)
            execution_time = time.time() - start_time
            
            # تحديث مقاييس الأداء
            node = self.nodes[node_id]
            node.performance_metrics["last_execution_time"] = execution_time
            node.current_tasks.remove(task_id)
            
            # حفظ النتيجة
            self.result_queue.put({
                "task_id": task_id,
                "result": result,
                "execution_time": execution_time,
                "node_id": node_id,
                "status": "completed"
            })
            
            logger.info(f"تم تنفيذ المهمة {task_id} في {execution_time:.2f} ثانية")
            
        except Exception as e:
            logger.error(f"فشل في تنفيذ المهمة {task_id}: {e}")
            
            # إزالة المهمة من العقدة
            if task_id in self.nodes[node_id].current_tasks:
                self.nodes[node_id].current_tasks.remove(task_id)
            
            self.result_queue.put({
                "task_id": task_id,
                "result": None,
                "error": str(e),
                "node_id": node_id,
                "status": "failed"
            })
    
    def get_task_result(self, task_id: str, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """الحصول على نتيجة مهمة"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = self.result_queue.get(timeout=1.0)
                if result["task_id"] == task_id:
                    return result
                else:
                    # إعادة النتيجة إلى الطابور إذا لم تكن المطلوبة
                    self.result_queue.put(result)
            except queue.Empty:
                continue
        
        logger.warning(f"انتهت المهلة الزمنية للمهمة: {task_id}")
        return None
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """الحصول على حالة المجموعة"""
        total_cores = sum(node.cpu_cores for node in self.nodes.values())
        active_nodes = sum(1 for node in self.nodes.values() if node.status == "active")
        total_tasks = sum(len(node.current_tasks) for node in self.nodes.values())
        
        return {
            "total_nodes": len(self.nodes),
            "active_nodes": active_nodes,
            "total_cores": total_cores,
            "current_tasks": total_tasks,
            "nodes": [asdict(node) for node in self.nodes.values()]
        }

class EvolutionEngine:
    """محرك التطور"""
    
    def __init__(self):
        self.experiments: Dict[str, EvolutionExperiment] = {}
        self.population_storage = {}
    
    def create_genetic_algorithm_experiment(self, name: str, fitness_function: Callable,
                                          population_size: int = 50, generations: int = 100,
                                          mutation_rate: float = 0.1, crossover_rate: float = 0.8) -> str:
        """إنشاء تجربة خوارزمية جينية"""
        
        experiment_id = hashlib.md5(f"{name}_{time.time()}".encode()).hexdigest()
        
        experiment = EvolutionExperiment(
            id=experiment_id,
            name=name,
            strategy=EvolutionStrategy.GENETIC_ALGORITHM,
            parameters={
                "mutation_rate": mutation_rate,
                "crossover_rate": crossover_rate,
                "selection_method": "tournament"
            },
            population_size=population_size,
            generations=generations,
            fitness_function=fitness_function.__name__,
            current_generation=0,
            best_solution=None,
            fitness_history=[],
            status="created",
            created_at=datetime.now()
        )
        
        self.experiments[experiment_id] = experiment
        logger.info(f"تم إنشاء تجربة خوارزمية جينية: {name}")
        
        return experiment_id
    
    def run_genetic_algorithm(self, experiment_id: str, initial_population: Optional[List] = None) -> Dict[str, Any]:
        """تشغيل خوارزمية جينية"""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"التجربة غير موجودة: {experiment_id}")
        
        experiment = self.experiments[experiment_id]
        experiment.status = "running"
        
        # إنشاء المجتمع الأولي
        if initial_population:
            population = initial_population
        else:
            population = self._generate_random_population(experiment.population_size)
        
        best_fitness = float('-inf')
        best_individual = None
        
        for generation in range(experiment.generations):
            experiment.current_generation = generation
            
            # تقييم اللياقة
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_fitness(individual, experiment.fitness_function)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy() if hasattr(individual, 'copy') else individual
            
            experiment.fitness_history.append(best_fitness)
            
            # الاختيار والتكاثر
            new_population = []
            
            # الحفاظ على أفضل الأفراد (Elitism)
            elite_count = max(1, experiment.population_size // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            
            for idx in elite_indices:
                new_population.append(population[idx])
            
            # إنتاج باقي المجتمع
            while len(new_population) < experiment.population_size:
                # اختيار الوالدين
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # التزاوج
                if np.random.random() < experiment.parameters["crossover_rate"]:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                # الطفرة
                if np.random.random() < experiment.parameters["mutation_rate"]:
                    child1 = self._mutate(child1)
                if np.random.random() < experiment.parameters["mutation_rate"]:
                    child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            # تحديد حجم المجتمع
            population = new_population[:experiment.population_size]
            
            logger.info(f"الجيل {generation}: أفضل لياقة = {best_fitness:.4f}")
        
        # حفظ أفضل حل
        experiment.best_solution = {
            "individual": best_individual,
            "fitness": best_fitness,
            "generation": experiment.current_generation
        }
        
        experiment.status = "completed"
        
        return {
            "experiment_id": experiment_id,
            "best_solution": experiment.best_solution,
            "fitness_history": experiment.fitness_history,
            "final_generation": experiment.current_generation
        }
    
    def _generate_random_population(self, size: int) -> List:
        """إنتاج مجتمع عشوائي"""
        # مثال بسيط - قائمة من الأرقام العشوائية
        population = []
        for _ in range(size):
            individual = np.random.random(10).tolist()  # فرد بـ 10 جينات
            population.append(individual)
        return population
    
    def _evaluate_fitness(self, individual: Any, fitness_function_name: str) -> float:
        """تقييم لياقة الفرد"""
        # مثال بسيط - مجموع المربعات
        if hasattr(individual, '__iter__'):
            return sum(x**2 for x in individual)
        else:
            return float(individual)
    
    def _tournament_selection(self, population: List, fitness_scores: List[float], tournament_size: int = 3) -> Any:
        """اختيار البطولة"""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover(self, parent1: List, parent2: List) -> Tuple[List, List]:
        """التزاوج (نقطة واحدة)"""
        if len(parent1) != len(parent2):
            return parent1, parent2
        
        crossover_point = np.random.randint(1, len(parent1))
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def _mutate(self, individual: List, mutation_strength: float = 0.1) -> List:
        """الطفرة"""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < 0.1:  # احتمال طفرة كل جين
                mutated[i] += np.random.normal(0, mutation_strength)
        
        return mutated

class ResearchEvolutionEngine:
    """محرك البحث والتطور الرئيسي"""
    
    def __init__(self, config_path: str = "research_evolution_config.json"):
        self.config = self._load_config(config_path)
        
        # تهيئة المكونات
        self.knowledge_extractor = KnowledgeExtractor()
        self.distributed_computing = DistributedComputing()
        self.evolution_engine = EvolutionEngine()
        
        # قاعدة بيانات المعرفة
        self.knowledge_db = self._init_knowledge_db()
        
        # مهام البحث
        self.research_tasks: Dict[str, ResearchTask] = {}
        
        # إحصائيات
        self.stats = {
            "knowledge_items_extracted": 0,
            "research_tasks_completed": 0,
            "evolution_experiments_run": 0,
            "distributed_tasks_executed": 0
        }
        
        logger.info("تم تهيئة محرك البحث والتطور")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """تحميل التكوين"""
        default_config = {
            "knowledge_extraction_interval": 3600,  # ساعة
            "max_knowledge_items_per_source": 100,
            "research_task_timeout": 1800,  # 30 دقيقة
            "evolution_population_size": 50,
            "evolution_generations": 100,
            "enable_distributed_computing": True
        }
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            logger.warning(f"ملف التكوين غير موجود: {config_path}. استخدام التكوين الافتراضي.")
            return default_config
    
    def _init_knowledge_db(self) -> sqlite3.Connection:
        """تهيئة قاعدة بيانات المعرفة"""
        db_path = "knowledge_database.db"
        conn = sqlite3.connect(db_path, check_same_thread=False)
        
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                url TEXT,
                authors TEXT,
                publication_date TEXT,
                keywords TEXT,
                relevance_score REAL,
                quality_score REAL,
                extracted_at TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                research_type TEXT,
                keywords TEXT,
                sources TEXT,
                priority INTEGER,
                status TEXT,
                created_at TEXT,
                deadline TEXT,
                results TEXT
            )
        ''')
        
        conn.commit()
        return conn
    
    async def create_research_task(self, title: str, description: str, 
                                 research_type: str, keywords: List[str],
                                 sources: List[str], priority: int = 5) -> str:
        """إنشاء مهمة بحثية"""
        
        task_id = hashlib.md5(f"{title}_{time.time()}".encode()).hexdigest()
        
        task = ResearchTask(
            id=task_id,
            title=title,
            description=description,
            research_type=ResearchType(research_type),
            keywords=keywords,
            sources=[KnowledgeSource(s) for s in sources],
            priority=priority,
            status="created",
            created_at=datetime.now(),
            deadline=None,
            assigned_nodes=[],
            results=None
        )
        
        self.research_tasks[task_id] = task
        
        # حفظ في قاعدة البيانات
        cursor = self.knowledge_db.cursor()
        cursor.execute('''
            INSERT INTO research_tasks 
            (id, title, description, research_type, keywords, sources, priority, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task_id, title, description, research_type,
            json.dumps(keywords), json.dumps(sources),
            priority, "created", datetime.now().isoformat()
        ))
        self.knowledge_db.commit()
        
        logger.info(f"تم إنشاء مهمة بحثية: {title}")
        
        return task_id
    
    async def execute_research_task(self, task_id: str) -> Dict[str, Any]:
        """تنفيذ مهمة بحثية"""
        
        if task_id not in self.research_tasks:
            raise ValueError(f"المهمة البحثية غير موجودة: {task_id}")
        
        task = self.research_tasks[task_id]
        task.status = "running"
        
        logger.info(f"بدء تنفيذ المهمة البحثية: {task.title}")
        
        all_knowledge_items = []
        
        # استخراج المعرفة من المصادر المختلفة
        for source in task.sources:
            try:
                if source == KnowledgeSource.ARXIV:
                    items = await self.knowledge_extractor.extract_from_arxiv(
                        task.keywords, 
                        self.config["max_knowledge_items_per_source"]
                    )
                elif source == KnowledgeSource.GITHUB:
                    items = await self.knowledge_extractor.extract_from_github(
                        task.keywords,
                        self.config["max_knowledge_items_per_source"]
                    )
                elif source == KnowledgeSource.WIKIPEDIA:
                    items = await self.knowledge_extractor.extract_from_wikipedia(
                        task.keywords,
                        self.config["max_knowledge_items_per_source"]
                    )
                else:
                    logger.warning(f"مصدر غير مدعوم: {source}")
                    continue
                
                all_knowledge_items.extend(items)
                
                # حفظ عناصر المعرفة في قاعدة البيانات
                for item in items:
                    self._save_knowledge_item(item)
                
            except Exception as e:
                logger.error(f"فشل في استخراج من {source}: {e}")
        
        # تحليل وتجميع النتائج
        results = self._analyze_research_results(all_knowledge_items, task)
        
        task.results = results
        task.status = "completed"
        
        # تحديث الإحصائيات
        self.stats["knowledge_items_extracted"] += len(all_knowledge_items)
        self.stats["research_tasks_completed"] += 1
        
        logger.info(f"تم إنجاز المهمة البحثية: {task.title}")
        
        return {
            "task_id": task_id,
            "status": "completed",
            "knowledge_items_found": len(all_knowledge_items),
            "results": results
        }
    
    def _save_knowledge_item(self, item: KnowledgeItem):
        """حفظ عنصر معرفة في قاعدة البيانات"""
        cursor = self.knowledge_db.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO knowledge_items
            (id, title, content, source, url, authors, publication_date, keywords,
             relevance_score, quality_score, extracted_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item.id, item.title, item.content, item.source.value, item.url,
            json.dumps(item.authors), 
            item.publication_date.isoformat() if item.publication_date else None,
            json.dumps(item.keywords), item.relevance_score, item.quality_score,
            item.extracted_at.isoformat(), json.dumps(item.metadata)
        ))
        
        self.knowledge_db.commit()
    
    def _analyze_research_results(self, knowledge_items: List[KnowledgeItem], 
                                task: ResearchTask) -> Dict[str, Any]:
        """تحليل نتائج البحث"""
        
        if not knowledge_items:
            return {"summary": "لم يتم العثور على نتائج", "insights": []}
        
        # تجميع الإحصائيات
        sources_count = defaultdict(int)
        avg_relevance = 0
        avg_quality = 0
        
        for item in knowledge_items:
            sources_count[item.source.value] += 1
            avg_relevance += item.relevance_score
            avg_quality += item.quality_score
        
        avg_relevance /= len(knowledge_items)
        avg_quality /= len(knowledge_items)
        
        # استخراج الكلمات المفتاحية الأكثر شيوعاً
        all_keywords = []
        for item in knowledge_items:
            all_keywords.extend(item.keywords)
        
        keyword_freq = defaultdict(int)
        for keyword in all_keywords:
            keyword_freq[keyword] += 1
        
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # أفضل النتائج
        best_items = sorted(knowledge_items, 
                          key=lambda x: x.relevance_score * x.quality_score, 
                          reverse=True)[:5]
        
        return {
            "summary": f"تم العثور على {len(knowledge_items)} عنصر معرفة",
            "statistics": {
                "total_items": len(knowledge_items),
                "sources_distribution": dict(sources_count),
                "average_relevance": avg_relevance,
                "average_quality": avg_quality
            },
            "top_keywords": top_keywords,
            "best_results": [
                {
                    "title": item.title,
                    "source": item.source.value,
                    "relevance_score": item.relevance_score,
                    "quality_score": item.quality_score,
                    "url": item.url
                }
                for item in best_items
            ],
            "insights": self._generate_insights(knowledge_items, task)
        }
    
    def _generate_insights(self, knowledge_items: List[KnowledgeItem], 
                         task: ResearchTask) -> List[str]:
        """توليد رؤى من نتائج البحث"""
        insights = []
        
        # تحليل الاتجاهات الزمنية
        dated_items = [item for item in knowledge_items if item.publication_date]
        if dated_items:
            recent_items = [item for item in dated_items 
                          if item.publication_date > datetime.now() - timedelta(days=365)]
            
            if len(recent_items) > len(dated_items) * 0.5:
                insights.append("هناك نشاط بحثي متزايد في هذا المجال خلال العام الماضي")
        
        # تحليل المصادر
        github_items = [item for item in knowledge_items if item.source == KnowledgeSource.GITHUB]
        if github_items:
            high_quality_repos = [item for item in github_items if item.quality_score > 0.7]
            if high_quality_repos:
                insights.append(f"تم العثور على {len(high_quality_repos)} مستودع عالي الجودة في GitHub")
        
        # تحليل الجودة العامة
        high_quality_items = [item for item in knowledge_items if item.quality_score > 0.8]
        if len(high_quality_items) > len(knowledge_items) * 0.3:
            insights.append("معظم النتائج المكتشفة عالية الجودة")
        
        return insights
    
    # واجهات برمجية للتفاعل مع المكونات الأخرى
    
    def search_knowledge(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """البحث في قاعدة المعرفة"""
        cursor = self.knowledge_db.cursor()
        
        cursor.execute('''
            SELECT * FROM knowledge_items 
            WHERE title LIKE ? OR content LIKE ?
            ORDER BY relevance_score DESC, quality_score DESC
            LIMIT ?
        ''', (f'%{query}%', f'%{query}%', limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "title": row[1],
                "content": row[2][:200] + "..." if len(row[2]) > 200 else row[2],
                "source": row[3],
                "url": row[4],
                "relevance_score": row[8],
                "quality_score": row[9]
            })
        
        return results
    
    def get_research_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """الحصول على حالة مهمة بحثية"""
        if task_id in self.research_tasks:
            task = self.research_tasks[task_id]
            return {
                "id": task.id,
                "title": task.title,
                "status": task.status,
                "progress": "100%" if task.status == "completed" else "في التقدم",
                "created_at": task.created_at.isoformat(),
                "results_summary": task.results.get("summary") if task.results else None
            }
        return None
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات النظام"""
        # إحصائيات قاعدة البيانات
        cursor = self.knowledge_db.cursor()
        cursor.execute("SELECT COUNT(*) FROM knowledge_items")
        total_knowledge_items = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM research_tasks")
        total_research_tasks = cursor.fetchone()[0]
        
        return {
            "knowledge_database": {
                "total_items": total_knowledge_items
            },
            "research_tasks": {
                "total": total_research_tasks,
                "active": len([t for t in self.research_tasks.values() if t.status == "running"])
            },
            "distributed_computing": self.distributed_computing.get_cluster_status(),
            "runtime_statistics": self.stats
        }

def main():
    """دالة الاختبار الرئيسية"""
    async def test_system():
        # إنشاء محرك البحث والتطور
        engine = ResearchEvolutionEngine()
        
        try:
            # اختبار إنشاء مهمة بحثية
            print("اختبار إنشاء مهمة بحثية...")
            
            task_id = await engine.create_research_task(
                title="بحث في الذكاء الاصطناعي",
                description="البحث عن أحدث التطورات في مجال الذكاء الاصطناعي",
                research_type="literature_review",
                keywords=["artificial intelligence", "machine learning", "deep learning"],
                sources=["arxiv", "github", "wikipedia"],
                priority=8
            )
            
            print(f"تم إنشاء المهمة البحثية: {task_id}")
            
            # تنفيذ المهمة البحثية
            print("تنفيذ المهمة البحثية...")
            result = await engine.execute_research_task(task_id)
            
            print(f"نتائج البحث: {result['knowledge_items_found']} عنصر معرفة")
            print(f"ملخص: {result['results']['summary']}")
            
            # اختبار البحث في قاعدة المعرفة
            print("\nاختبار البحث في قاعدة المعرفة...")
            search_results = engine.search_knowledge("machine learning", 5)
            
            for i, result in enumerate(search_results, 1):
                print(f"{i}. {result['title']} (جودة: {result['quality_score']:.2f})")
            
            # عرض الإحصائيات
            print("\nإحصائيات النظام:")
            stats = engine.get_system_statistics()
            for key, value in stats.items():
                print(f"{key}: {value}")
        
        except Exception as e:
            print(f"خطأ في الاختبار: {e}")
    
    # تشغيل الاختبار
    asyncio.run(test_system())

if __name__ == "__main__":
    main()

