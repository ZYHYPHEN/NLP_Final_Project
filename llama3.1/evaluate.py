import torch
import numpy as np
import json
import requests
import time
import os
import gc
import re
import math
from collections import Counter
from typing import List, Dict, Tuple, Any
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# 确保nltk数据可用
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

warnings.filterwarnings('ignore')

# ==================== 优化模型管理器 ====================
class OptimizedModelManager:
    def __init__(self):
        self.current_model = None
        self.tokenizer = None
        self.device = None
        
    def check_memory(self):
        """检查GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated = torch.cuda.memory_allocated() / 1e9
            available = total_memory - allocated
            
            print(f"GPU内存 - 总量: {total_memory:.1f}GB, 已用: {allocated:.1f}GB, 可用: {available:.1f}GB")
            
            if available > 8:
                return "cuda"
        return "cpu"
    
    def load_model(self, model_path, lora_path=None):
        """加载模型"""
        self.unload_model()
        
        print(f"加载模型: {model_path}")
        
        device_strategy = self.check_memory()
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.pad_token = self.tokenizer.eos_token = "<|end_of_text|>"
            
            if device_strategy == "cuda":
                self.device = "cuda"
                print("使用GPU加载")
                
                # 使用bfloat16加速
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    trust_remote_code=True
                ).eval()
                
            else:  # CPU模式
                self.device = "cpu"
                print("使用CPU加载")
                
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=True
                ).eval().to("cpu")
            
            # 加载LoRA适配器
            if lora_path and os.path.exists(lora_path):
                print(f"加载LoRA权重: {lora_path}")
                self.current_model = PeftModel.from_pretrained(
                    self.current_model, 
                    model_id=lora_path
                )
            
            print(f"模型加载完成，设备: {self.device}")
            return self.current_model, self.tokenizer, self.device
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def generate_response(self, prompt, max_new_tokens=100, temperature=0.8, top_p=0.9):
        """生成回答"""
        if not self.current_model or not self.tokenizer:
            raise ValueError("模型未加载")
        
        try:
            # 构建消息
            messages = [
                {"role": "system", "content": "假设你是《西游记》原著中的神话角色--孙悟空, 请模仿《西游记》成书年代的语言风格和孙悟空本人的语言风格。"},
                {"role": "user", "content": prompt}
            ]
            
            # 应用聊天模板
            input_ids = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 准备输入
            model_inputs = self.tokenizer([input_ids], return_tensors="pt")
            if self.device == "cuda":
                model_inputs = model_inputs.to("cuda")
            
            # 确保有attention_mask
            if "attention_mask" not in model_inputs:
                model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"])
            
            # 生成
            generated_ids = self.current_model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
            
            # 解码
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # 清理响应文本
            response = self.clean_response(response)
            
            return response
            
        except Exception as e:
            print(f"生成失败: {e}")
            return ""
    
    def clean_response(self, text):
        """清理响应文本"""
        if text:
            text = text.encode('utf-8', 'ignore').decode('utf-8')
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        return text
    
    def cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def unload_model(self):
        """卸载模型"""
        if self.current_model:
            del self.current_model
            self.current_model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        self.cleanup_memory()
        print("模型已卸载")

# ==================== 语言分析器（仅保留相似度计算） ====================
class LanguageAnalyzer:
    """语言特征分析器"""
    
    def __init__(self):
        try:
            jieba.initialize()
        except:
            pass
    
    def calculate_similarity_batch(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """批量计算文本相似度（使用TF-IDF余弦相似度）"""
        if not texts1 or not texts2:
            return np.array([])
        
        # 组合所有文本
        all_texts = texts1 + texts2
        
        # 创建TF-IDF向量器
        vectorizer = TfidfVectorizer(
            tokenizer=lambda x: list(jieba.cut(x)),
            max_features=1000,
            lowercase=False
        )
        
        try:
            # 拟合和转换
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # 分割矩阵
            matrix1 = tfidf_matrix[:len(texts1)]
            matrix2 = tfidf_matrix[len(texts1):]
            
            # 计算余弦相似度
            similarity_matrix = cosine_similarity(matrix1, matrix2)
            
            return similarity_matrix
            
        except Exception as e:
            print(f"计算相似度失败: {e}")
            return np.zeros((len(texts1), len(texts2)))

# ==================== 多样性评估器 ====================
class DiversityEvaluator:
    """文本多样性评估器"""
    
    @staticmethod
    def calculate_distinct_n(texts: List[str], n: int = 2) -> Tuple[float, float]:
        """计算Distinct-n指标"""
        if not texts:
            return 0.0, 0
        
        all_ngrams = []
        total_ngrams = 0
        
        for text in texts:
            if not text.strip():
                continue
                
            words = list(jieba.cut(text))
            if len(words) < n:
                continue
                
            # 提取n-grams
            ngrams = []
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i+n])
                ngrams.append(ngram)
                all_ngrams.append(ngram)
            
            total_ngrams += len(ngrams)
        
        if total_ngrams == 0:
            return 0.0, 0
        
        # 计算distinct-n
        distinct_ngrams = set(all_ngrams)
        distinct_score = len(distinct_ngrams) / total_ngrams
        
        return distinct_score, total_ngrams
    
    @staticmethod
    def calculate_self_bleu(texts: List[str], n: int = 4) -> float:
        """计算Self-BLEU分数（评估文本间多样性）"""
        if len(texts) < 2:
            return 0.0
        
        try:
            total_bleu = 0.0
            count = 0
            
            smoothing = SmoothingFunction().method1
            
            for i in range(len(texts)):
                # 当前文本作为候选
                candidate = texts[i]
                # 其他文本作为参考
                references = [texts[j] for j in range(len(texts)) if j != i]
                
                if not candidate.strip() or not any(ref.strip() for ref in references):
                    continue
                
                # 分词
                candidate_tokens = list(jieba.cut(candidate))
                reference_tokens = [list(jieba.cut(ref)) for ref in references]
                
                # 计算BLEU
                bleu_score = sentence_bleu(
                    reference_tokens,
                    candidate_tokens,
                    weights=[1/n] * n,
                    smoothing_function=smoothing
                )
                
                total_bleu += bleu_score
                count += 1
            
            return total_bleu / max(count, 1) if count > 0 else 0.0
            
        except Exception as e:
            print(f"计算Self-BLEU失败: {e}")
            return 0.0
    
    @staticmethod
    def calculate_repetition_rate(texts: List[str]) -> float:
        """计算重复率"""
        if not texts:
            return 0.0
        
        all_words = []
        for text in texts:
            words = list(jieba.cut(text))
            all_words.extend(words)
        
        if len(all_words) < 2:
            return 0.0
        
        # 计算相邻重复
        repeated_count = 0
        for i in range(1, len(all_words)):
            if all_words[i] == all_words[i-1]:
                repeated_count += 1
        
        return repeated_count / max(len(all_words) - 1, 1)

# ==================== DeepSeek评估器 ====================
class DeepSeekEvaluator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com"
        
        if self.api_key:
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        else:
            self.headers = {}
            print("未提供DeepSeek API密钥")
    
    def clean_text_for_api(self, text):
        """清理文本，避免API调用失败"""
        if not text:
            return ""
        
        # 移除特殊字符和控制字符[citation:1]
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        text = re.sub(r'[\x00-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202f]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 限制长度，避免API限制
        return text[:500]
    
    def evaluate_response(self, prompt, response, reference=None):
        """评估单个回答"""
        if not self.api_key:
            return None
        
        # 检查响应是否为空
        if not response or not response.strip():
            print(f"警告：响应为空，跳过DeepSeek评估")
            return 0
        
        # 清理文本
        clean_prompt = self.clean_text_for_api(prompt)
        clean_response = self.clean_text_for_api(response)
        clean_reference = self.clean_text_for_api(reference) if reference else None
        
        # 再次检查清理后的响应
        if not clean_response or len(clean_response) < 3:
            print(f"警告：清理后响应过短，跳过DeepSeek评估")
            return 0
        
        evaluation_prompt = self.build_evaluation_prompt(clean_prompt, clean_response, clean_reference)
        
        try:
            result = self.call_api(evaluation_prompt)
            return self.parse_score(result)
        except Exception as e:
            print(f"DeepSeek API调用失败: {e}")
            return None
    
    def build_evaluation_prompt(self, prompt, response, reference):
        """构建评估提示[citation:3]"""
        if reference:
            template = """请评估以下AI回答的质量，给出1-10分的分数。

用户问题: {prompt}
参考回答: {reference}
待评估回答: {response}

评估标准:
1.符合角色: 是否符合《西游记》原著中孙悟空的语言风格和角色设定
2.符合时代：是否符合《西游记》原著成书年代的语言风格

请只返回一个1-10分的整数分数，不要有任何其他文字。"""
        else:
            template = """请评估以下AI回答的质量，给出1-10分的分数。

用户问题: {prompt}
待评估回答: {response}

评估标准:
1.符合角色: 是否符合《西游记》原著中孙悟空的语言风格和角色设定
2.符合时代：是否符合《西游记》原著成书年代的语言风格

请只返回一个1-10分的整数分数，不要有任何其他文字。"""
        
        return template.format(
            prompt=prompt[:200],
            response=response[:300],
            reference=reference[:200] if reference else "无"
        )
    
    def call_api(self, prompt):
        """调用DeepSeek API[citation:7]"""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个专业的AI评估专家，对《西游记》原著和孙悟空的语言风格十分了解，请按照要求输出分数。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 10,
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            api_response = result["choices"][0]["message"]["content"].strip()
            
            # 调试：记录API响应
            print(f"DeepSeek API响应: {api_response}")
            
            return api_response
        except Exception as e:
            print(f"DeepSeek API调用失败: {e}")
            raise
    
    def parse_score(self, text):
        """解析分数"""
        if not text:
            return None

        try:
            # 移除所有空格
            text = text.strip()
            
            # 检查是否为纯数字（最常见情况）
            if text.isdigit():
                score = int(text)
                if 1 <= score <= 10:
                    return score
                else:
                    print(f"警告：分数 {score} 不在1-10范围内")
                    return None
            
            # 检查常见模式
            patterns = [
                r'^(\d+)$',  # 纯数字
                r'评分[:：]\s*(\d+)',
                r'分数[:：]\s*(\d+)',
                r'(\d+)\s*分',
                r'得分[:：]\s*(\d+)',
                r'score[:：]\s*(\d+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    score = int(match.group(1))
                    if 1 <= score <= 10:
                        return score
                    else:
                        print(f"警告：匹配到分数 {score} 但不在1-10范围内")
            
            # 如果找不到有效分数，记录日志
            print(f"警告：无法从文本中解析有效分数: {text[:50]}...")
            return None
            
        except Exception as e:
            print(f"解析分数时出错: {e}, 文本: {text[:50]}")
            return None

# ==================== 批量评估函数 ====================
def run_batch_evaluation():
    """批量评估主函数"""
    print("=" * 60)
    print("LoRA微调效果批量评估系统")
    print("=" * 60)
    
    # 配置模型路径（请根据实际情况修改）
    BASE_MODEL_PATH = "./LLM-Research/Meta-Llama-3___1-8B-Instruct"
    LORA_PATH = "./output/llama3_1_instruct_lora/best_model"
    
    # 检查路径
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"错误：基础模型路径不存在: {BASE_MODEL_PATH}")
        print("请检查BASE_MODEL_PATH配置")
        return
    
    if not os.path.exists(LORA_PATH):
        print(f"警告：LoRA路径不存在: {LORA_PATH}")
        print("将仅评估基线模型")
        LORA_PATH = None
    
    # 1. 从文件读取DeepSeek API密钥[citation:1]
    print("\n1. 读取配置文件...")
    api_key = None
    try:
        with open("deepseek_api_key.txt", "r", encoding="utf-8") as f:
            api_key = f.read().strip()
        if api_key:
            print(f"  已读取DeepSeek API密钥（长度: {len(api_key)}）")
        else:
            print("  警告：deepseek_api_key.txt文件为空")
    except FileNotFoundError:
        print("  错误：找不到deepseek_api_key.txt文件")
        print("  请在同目录下创建deepseek_api_key.txt文件，并填入API密钥")
        return
    
    # 2. 从文件读取评估问题
    prompts = []
    try:
        with open("evaluate.txt", "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        if not prompts:
            print("  错误：evaluate.txt文件中没有有效问题")
            return
        
        print(f"  已从evaluate.txt读取 {len(prompts)} 个评估问题")
        
        # 显示前几个问题
        print("  前3个问题示例:")
        for i, prompt in enumerate(prompts[:3]):
            print(f"    {i+1}. {prompt[:50]}...")
        if len(prompts) > 3:
            print(f"    ... 以及另外 {len(prompts)-3} 个问题")
            
    except FileNotFoundError:
        print("  错误：找不到evaluate.txt文件")
        print("  请在同目录下创建evaluate.txt文件，每行一个评估问题")
        return
    
    # 3. 初始化评估器
    deepseek_eval = DeepSeekEvaluator(api_key=api_key)
    language_analyzer = LanguageAnalyzer()
    diversity_evaluator = DiversityEvaluator()
    model_manager = OptimizedModelManager()
    
    all_results = []
    base_responses = []
    finetuned_responses = []
    
    print(f"\n2. 开始批量评估 {len(prompts)} 个问题...")
    
    # 4. 处理基线模型回答
    print("\n2.1 生成基线模型回答...")
    try:
        print("加载基线模型...")
        model_manager.load_model(BASE_MODEL_PATH, None)
        
        for i, prompt in enumerate(prompts):
            print(f"  问题 {i+1}/{len(prompts)}: {prompt[:40]}...")
            response = model_manager.generate_response(prompt, max_new_tokens=150)
            base_responses.append(response)
        
        model_manager.unload_model()
    except Exception as e:
        print(f"基线模型处理失败: {e}")
        base_responses = [f"生成失败: {e}"] * len(prompts)
    
    # 清理内存
    time.sleep(2)
    model_manager.cleanup_memory()
    
    # 5. 处理微调模型回答
    print("\n2.2 生成微调模型回答...")
    if LORA_PATH:
        try:
            print("加载微调模型...")
            model_manager.load_model(BASE_MODEL_PATH, LORA_PATH)
            
            for i, prompt in enumerate(prompts):
                print(f"  问题 {i+1}/{len(prompts)}: {prompt[:40]}...")
                response = model_manager.generate_response(prompt, max_new_tokens=150)
                finetuned_responses.append(response)
            
            model_manager.unload_model()
        except Exception as e:
            print(f"微调模型处理失败: {e}")
            finetuned_responses = [f"生成失败: {e}"] * len(prompts)
    else:
        finetuned_responses = ["未加载微调模型"] * len(prompts)
    
    # 6. 计算多样性指标[citation:2]
    print("\n2.3 计算多样性指标...")
    
    # 计算微调模型多样性
    finetuned_diversity = {}
    valid_finetuned = [r for r in finetuned_responses if r.strip() and r != "未加载微调模型" and not r.startswith("生成失败")]
    
    if len(valid_finetuned) >= 2:
        distinct1, total1 = diversity_evaluator.calculate_distinct_n(valid_finetuned, 1)
        distinct2, total2 = diversity_evaluator.calculate_distinct_n(valid_finetuned, 2)
        self_bleu = diversity_evaluator.calculate_self_bleu(valid_finetuned, 2)
        repetition_rate = diversity_evaluator.calculate_repetition_rate(valid_finetuned)
        
        finetuned_diversity = {
            'distinct_1': distinct1,
            'distinct_2': distinct2,
            'self_bleu': self_bleu,
            'repetition_rate': repetition_rate,
            'unique_ngrams_1': total1 * distinct1,
            'unique_ngrams_2': total2 * distinct2,
        }
    
    # 计算基线模型多样性
    baseline_diversity = {}
    valid_baseline = [r for r in base_responses if r.strip() and not r.startswith("生成失败")]
    
    if len(valid_baseline) >= 2:
        distinct1, total1 = diversity_evaluator.calculate_distinct_n(valid_baseline, 1)
        distinct2, total2 = diversity_evaluator.calculate_distinct_n(valid_baseline, 2)
        self_bleu = diversity_evaluator.calculate_self_bleu(valid_baseline, 2)
        repetition_rate = diversity_evaluator.calculate_repetition_rate(valid_baseline)
        
        baseline_diversity = {
            'distinct_1': distinct1,
            'distinct_2': distinct2,
            'self_bleu': self_bleu,
            'repetition_rate': repetition_rate,
            'unique_ngrams_1': total1 * distinct1,
            'unique_ngrams_2': total2 * distinct2,
        }
    
    # 7. 计算与基线相似度
    print("\n2.4 计算与基线相似度...")
    baseline_similarities = []
    if base_responses and finetuned_responses and LORA_PATH:
        # 只计算有效响应的相似度
        valid_pairs = [(b, f) for b, f in zip(base_responses, finetuned_responses) 
                      if b.strip() and f.strip() and f != "未加载微调模型" 
                      and not b.startswith("生成失败") and not f.startswith("生成失败")]
        
        if valid_pairs:
            base_valid = [b for b, f in valid_pairs]
            finetuned_valid = [f for b, f in valid_pairs]
            
            similarity_matrix = language_analyzer.calculate_similarity_batch(finetuned_valid, base_valid)
            if similarity_matrix.size > 0:
                # 获取对角线上的相似度（一一对应的比较）
                for i in range(min(len(finetuned_valid), len(base_valid))):
                    if i < similarity_matrix.shape[0] and i < similarity_matrix.shape[1]:
                        baseline_similarities.append(float(similarity_matrix[i, i]))
    
    # 8. DeepSeek评估
    print("\n2.5 进行DeepSeek评估...")
    deepseek_scores = {'base': [], 'finetuned': []}
    
    for i in range(len(prompts)):
        if i >= len(base_responses) or i >= len(finetuned_responses):
            continue
        
        # 检查响应是否有效
        base_response_valid = base_responses[i] and base_responses[i].strip()
        finetuned_response_valid = finetuned_responses[i] and finetuned_responses[i].strip() and finetuned_responses[i] != "未加载微调模型"
        
        try:
            # 基线模型评估
            if base_response_valid:
                base_score = deepseek_eval.evaluate_response(prompts[i], base_responses[i])
            else:
                base_score = 0
                print(f"  问题 {i+1} 基线响应无效，给0分")
            
            # 微调模型评估
            if finetuned_response_valid:
                finetuned_score = deepseek_eval.evaluate_response(prompts[i], finetuned_responses[i])
            else:
                finetuned_score = 0
                print(f"  问题 {i+1} 微调响应无效，给0分")
            
            # 记录分数
            if base_score is not None:
                deepseek_scores['base'].append(base_score)
            if finetuned_score is not None:
                deepseek_scores['finetuned'].append(finetuned_score)
            
            print(f"  问题 {i+1}: 基线={base_score if base_score is not None else 'N/A'}, 微调={finetuned_score if finetuned_score is not None else 'N/A'}")
            
            # API速率限制
            if i < len(prompts) - 1:
                time.sleep(1)
                
        except Exception as e:
            print(f"  问题 {i+1} DeepSeek评估失败: {e}")
    
    # 9. 计算比较统计
    print("\n2.6 计算模型比较统计...")
    comparison_stats = {'微调更好': 0, '基线更好': 0, '两者相当': 0}
    
    if deepseek_scores['base'] and deepseek_scores['finetuned']:
        min_len = min(len(deepseek_scores['base']), len(deepseek_scores['finetuned']))
        
        for i in range(min_len):
            base_score = deepseek_scores['base'][i]
            finetuned_score = deepseek_scores['finetuned'][i]
            
            if base_score is not None and finetuned_score is not None:
                if finetuned_score > base_score:
                    comparison_stats['微调更好'] += 1
                elif finetuned_score < base_score:
                    comparison_stats['基线更好'] += 1
                else:
                    comparison_stats['两者相当'] += 1
    
    # 10. 显示统计结果
    print("\n3. 评估统计:")
    print("=" * 50)
    
    # 多样性结果
    print("\n【多样性评估】")
    
    if finetuned_diversity:
        print(f"\n微调模型多样性:")
        print(f"  Distinct-1: {finetuned_diversity.get('distinct_1', 0):.3f}")
        print(f"  Distinct-2: {finetuned_diversity.get('distinct_2', 0):.3f}")
        print(f"  Self-BLEU: {finetuned_diversity.get('self_bleu', 0):.3f}")
        print(f"  重复率: {finetuned_diversity.get('repetition_rate', 0):.3f}")
    
    if baseline_diversity:
        print(f"\n基线模型多样性:")
        print(f"  Distinct-1: {baseline_diversity.get('distinct_1', 0):.3f}")
        print(f"  Distinct-2: {baseline_diversity.get('distinct_2', 0):.3f}")
        print(f"  Self-BLEU: {baseline_diversity.get('self_bleu', 0):.3f}")
        print(f"  重复率: {baseline_diversity.get('repetition_rate', 0):.3f}")
    
    # 与基线相似度
    if baseline_similarities:
        print(f"\n【与基线相似度】")
        print(f"  平均相似度: {np.mean(baseline_similarities):.3f} (±{np.std(baseline_similarities):.3f})")
        print(f"  最小相似度: {np.min(baseline_similarities):.3f}")
        print(f"  最大相似度: {np.max(baseline_similarities):.3f}")
        print(f"  有效比较数: {len(baseline_similarities)}")
    
    # DeepSeek评估结果
    if deepseek_scores['base'] and deepseek_scores['finetuned']:
        print("\n【DeepSeek人工评分】")
        avg_base = np.mean(deepseek_scores['base'])
        avg_finetuned = np.mean(deepseek_scores['finetuned'])
        improvement = avg_finetuned - avg_base
        
        print(f"  基线模型平均分: {avg_base:.2f}/10")
        print(f"  微调模型平均分: {avg_finetuned:.2f}/10")
        print(f"  平均提升: {improvement:+.2f}")
    
    # 模型比较统计
    if comparison_stats['微调更好'] > 0 or comparison_stats['基线更好'] > 0:
        print(f"\n【模型比较统计】")
        print(f"  微调模型更好: {comparison_stats['微调更好']} 个问题")
        print(f"  基线模型更好: {comparison_stats['基线更好']} 个问题")
        print(f"  两者相当: {comparison_stats['两者相当']} 个问题")
    
    # 11. 保存所有结果
    print("\n4. 保存结果...")
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(prompts),
        "model_path": BASE_MODEL_PATH,
        "lora_path": LORA_PATH,
        "prompts": prompts,
        "base_responses": base_responses,
        "finetuned_responses": finetuned_responses,
        "evaluations": {
            "diversity": {
                "finetuned": finetuned_diversity,
                "baseline": baseline_diversity
            },
            "baseline_similarity": {
                "values": baseline_similarities,
                "mean": float(np.mean(baseline_similarities)) if baseline_similarities else 0.0,
                "std": float(np.std(baseline_similarities)) if baseline_similarities else 0.0,
                "count": len(baseline_similarities)
            }
        }
    }
    
    if deepseek_scores['base'] and deepseek_scores['finetuned']:
        summary["deepseek_scores"] = {
            "base_scores": deepseek_scores['base'],
            "finetuned_scores": deepseek_scores['finetuned'],
            "avg_base": float(np.mean(deepseek_scores['base'])),
            "avg_finetuned": float(np.mean(deepseek_scores['finetuned'])),
            "comparison_stats": comparison_stats
        }

    # 10.5 确保results目录存在
    print("\n4. 准备保存结果...")
    results_dir = "results"
    try:
        os.makedirs(results_dir, exist_ok=True)
        print(f"  结果将保存到 {results_dir} 目录")
    except Exception as e:
        print(f"  创建结果目录失败: {e}")
        print("  将尝试保存到当前目录")
        results_dir = "."

    # 11. 保存所有结果
    output_file = os.path.join(results_dir, f"batch_evaluation_result_{int(time.time())}.json")
    try:
        # 安全的JSON编码函数
        def safe_encode(obj):
            if isinstance(obj, str):
                return obj.encode('utf-8', 'ignore').decode('utf-8')
            elif isinstance(obj, dict):
                return {safe_encode(k): safe_encode(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [safe_encode(item) for item in obj]
            elif isinstance(obj, (int, float, bool)):
                return obj
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif obj is None:
                return None
            else:
                return str(obj)
        
        # 清理数据
        cleaned_summary = safe_encode(summary)
        
        with open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
            json.dump(cleaned_summary, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存到: {output_file}")
        
        # 同时保存简化的结果便于阅读
        simplified_file = os.path.join(results_dir, f"simple_summary_{int(time.time())}.txt")
        with open(simplified_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("微调模型评估结果摘要\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"评估时间: {summary['timestamp']}\n")
            f.write(f"问题数量: {summary['total_questions']}\n")
            f.write(f"基础模型: {BASE_MODEL_PATH}\n")
            f.write(f"LoRA路径: {LORA_PATH or '无'}\n\n")
            
            f.write("【多样性指标】\n")
            f.write("-" * 40 + "\n")
            
            if finetuned_diversity:
                f.write(f"微调模型:\n")
                f.write(f"  Distinct-1: {finetuned_diversity.get('distinct_1', 0):.3f}\n")
                f.write(f"  Distinct-2: {finetuned_diversity.get('distinct_2', 0):.3f}\n")
                f.write(f"  Self-BLEU: {finetuned_diversity.get('self_bleu', 0):.3f}\n")
                f.write(f"  重复率: {finetuned_diversity.get('repetition_rate', 0):.3f}\n")
            
            if baseline_diversity:
                f.write(f"\n基线模型:\n")
                f.write(f"  Distinct-1: {baseline_diversity.get('distinct_1', 0):.3f}\n")
                f.write(f"  Distinct-2: {baseline_diversity.get('distinct_2', 0):.3f}\n")
                f.write(f"  Self-BLEU: {baseline_diversity.get('self_bleu', 0):.3f}\n")
                f.write(f"  重复率: {baseline_diversity.get('repetition_rate', 0):.3f}\n")
            
            if baseline_similarities:
                f.write(f"\n【与基线相似度】\n")
                f.write(f"  平均相似度: {np.mean(baseline_similarities):.3f}\n")
                f.write(f"  比较数量: {len(baseline_similarities)}\n")
            
            if deepseek_scores['base'] and deepseek_scores['finetuned']:
                f.write(f"\n【DeepSeek人工评分】\n")
                f.write(f"  基线平均: {np.mean(deepseek_scores['base']):.2f}/10\n")
                f.write(f"  微调平均: {np.mean(deepseek_scores['finetuned']):.2f}/10\n")
                f.write(f"  提升: {np.mean(deepseek_scores['finetuned']) - np.mean(deepseek_scores['base']):+.2f}\n")
                
                f.write(f"\n【模型对比】\n")
                f.write(f"  微调更好: {comparison_stats['微调更好']} 个问题\n")
                f.write(f"  基线更好: {comparison_stats['基线更好']} 个问题\n")
                f.write(f"  两者相当: {comparison_stats['两者相当']} 个问题\n")
            
            f.write("\n【样本示例】\n")
            f.write("-" * 40 + "\n")
            
            # 显示前3个样本
            for i in range(min(3, len(prompts))):
                f.write(f"\n问题 {i+1}: {prompts[i]}\n")
                f.write(f"基线回答: {base_responses[i][:70]}...\n")
                if i < len(finetuned_responses):
                    f.write(f"微调回答: {finetuned_responses[i][:70]}...\n")
                
                if deepseek_scores['base'] and deepseek_scores['finetuned'] and i < min(len(deepseek_scores['base']), len(deepseek_scores['finetuned'])):
                    base_score = deepseek_scores['base'][i]
                    finetuned_score = deepseek_scores['finetuned'][i]
                    if base_score is not None and finetuned_score is not None:
                        f.write(f"DeepSeek评分: 基线={base_score}, 微调={finetuned_score}\n")
        
        print(f"评估摘要已保存到: {simplified_file}")
        
    except Exception as e:
        print(f"保存结果失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("批量评估完成!")
    print("=" * 60)
    
    # output_file = f"batch_evaluation_result_{int(time.time())}.json"
    # try:
    #     # 安全的JSON编码函数
    #     def safe_encode(obj):
    #         if isinstance(obj, str):
    #             return obj.encode('utf-8', 'ignore').decode('utf-8')
    #         elif isinstance(obj, dict):
    #             return {safe_encode(k): safe_encode(v) for k, v in obj.items()}
    #         elif isinstance(obj, list):
    #             return [safe_encode(item) for item in obj]
    #         elif isinstance(obj, (int, float, bool)):
    #             return obj
    #         elif isinstance(obj, np.integer):
    #             return int(obj)
    #         elif isinstance(obj, np.floating):
    #             return float(obj)
    #         elif isinstance(obj, np.ndarray):
    #             return obj.tolist()
    #         elif obj is None:
    #             return None
    #         else:
    #             return str(obj)
        
    #     # 清理数据
    #     cleaned_summary = safe_encode(summary)
        
    #     with open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
    #         json.dump(cleaned_summary, f, ensure_ascii=False, indent=2)
    #     print(f"结果已保存到: {output_file}")
        
    #     # 同时保存简化的结果便于阅读
    #     simplified_file = f"simple_summary_{int(time.time())}.txt"
    #     with open(simplified_file, 'w', encoding='utf-8') as f:
    #         f.write("=" * 60 + "\n")
    #         f.write("微调模型评估结果摘要\n")
    #         f.write("=" * 60 + "\n\n")
            
    #         f.write(f"评估时间: {summary['timestamp']}\n")
    #         f.write(f"问题数量: {summary['total_questions']}\n")
    #         f.write(f"基础模型: {BASE_MODEL_PATH}\n")
    #         f.write(f"LoRA路径: {LORA_PATH or '无'}\n\n")
            
    #         f.write("【多样性指标】\n")
    #         f.write("-" * 40 + "\n")
            
    #         if finetuned_diversity:
    #             f.write(f"微调模型:\n")
    #             f.write(f"  Distinct-1: {finetuned_diversity.get('distinct_1', 0):.3f}\n")
    #             f.write(f"  Distinct-2: {finetuned_diversity.get('distinct_2', 0):.3f}\n")
    #             f.write(f"  Self-BLEU: {finetuned_diversity.get('self_bleu', 0):.3f}\n")
    #             f.write(f"  重复率: {finetuned_diversity.get('repetition_rate', 0):.3f}\n")
            
    #         if baseline_diversity:
    #             f.write(f"\n基线模型:\n")
    #             f.write(f"  Distinct-1: {baseline_diversity.get('distinct_1', 0):.3f}\n")
    #             f.write(f"  Distinct-2: {baseline_diversity.get('distinct_2', 0):.3f}\n")
    #             f.write(f"  Self-BLEU: {baseline_diversity.get('self_bleu', 0):.3f}\n")
    #             f.write(f"  重复率: {baseline_diversity.get('repetition_rate', 0):.3f}\n")
            
    #         if baseline_similarities:
    #             f.write(f"\n【与基线相似度】\n")
    #             f.write(f"  平均相似度: {np.mean(baseline_similarities):.3f}\n")
    #             f.write(f"  比较数量: {len(baseline_similarities)}\n")
            
    #         if deepseek_scores['base'] and deepseek_scores['finetuned']:
    #             f.write(f"\n【DeepSeek人工评分】\n")
    #             f.write(f"  基线平均: {np.mean(deepseek_scores['base']):.2f}/10\n")
    #             f.write(f"  微调平均: {np.mean(deepseek_scores['finetuned']):.2f}/10\n")
    #             f.write(f"  提升: {np.mean(deepseek_scores['finetuned']) - np.mean(deepseek_scores['base']):+.2f}\n")
                
    #             f.write(f"\n【模型对比】\n")
    #             f.write(f"  微调更好: {comparison_stats['微调更好']} 个问题\n")
    #             f.write(f"  基线更好: {comparison_stats['基线更好']} 个问题\n")
    #             f.write(f"  两者相当: {comparison_stats['两者相当']} 个问题\n")
            
    #         f.write("\n【样本示例】\n")
    #         f.write("-" * 40 + "\n")
            
    #         # 显示前3个样本
    #         for i in range(min(3, len(prompts))):
    #             f.write(f"\n问题 {i+1}: {prompts[i]}\n")
    #             f.write(f"基线回答: {base_responses[i][:70]}...\n")
    #             if i < len(finetuned_responses):
    #                 f.write(f"微调回答: {finetuned_responses[i][:70]}...\n")
                
    #             if deepseek_scores['base'] and deepseek_scores['finetuned'] and i < min(len(deepseek_scores['base']), len(deepseek_scores['finetuned'])):
    #                 base_score = deepseek_scores['base'][i]
    #                 finetuned_score = deepseek_scores['finetuned'][i]
    #                 if base_score is not None and finetuned_score is not None:
    #                     f.write(f"DeepSeek评分: 基线={base_score}, 微调={finetuned_score}\n")
        
    #     print(f"摘要已保存到: {simplified_file}")
        
    # except Exception as e:
    #     print(f"保存结果失败: {e}")
    
    # print("\n" + "=" * 60)
    # print("批量评估完成!")
    # print("=" * 60)

# ==================== 主程序 ====================
if __name__ == "__main__":
    run_batch_evaluation()