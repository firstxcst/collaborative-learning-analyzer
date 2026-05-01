"""语义智能体 - LLM 对话质量分析"""

import warnings
warnings.filterwarnings("ignore")

import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from .config import get_config, get_semantic_config, SemanticAgentConfig
from .data_models import SemanticAnalysisResult, SpeakingSegment


# 默认 Prompt 模板
DEFAULT_PROMPT_TEMPLATE = """你是一个教育评价专家。请分析以下小组讨论的转录文本，输出结构化的协作质量指标。

## 输入格式
对话转录（包含说话人标识和发言内容）：
{transcript}

## 分析任务
请根据上述对话，分析以下维度：
1. **主题相关度** (0-1)：对话是否围绕任务主题展开
2. **观点碰撞次数**：反驳、补充、质疑等深度交互的次数
3. **轮流模式**：balanced（均衡）/ monopolizing（垄断）/ chaotic（无序）
4. **论证深度** (0-1)：观点是否有充分理由支撑
5. **共识质量** (0-1)：结论是否经过充分讨论达成

## 输出格式（严格 JSON）
```json
{{
  "topic_relevance": 0.85,
  "opinion_collisions": 3,
  "turn_taking_pattern": "balanced",
  "argument_depth_score": 0.7,
  "consensus_quality": 0.6,
  "evidence": [
    "证据1：学生A提出了观点X",
    "证据2：学生B对此进行了反驳"
  ]
}}
```

请直接输出 JSON，不要包含其他解释文字。
"""


class SemanticAgent:
    """
    语义智能体：判断对话是否围绕任务、是否有深度观点碰撞
    
    核心能力：
    1. 对话主题相关度分析
    2. 观点碰撞检测（反驳/补充/质疑）
    3. 轮流模式识别
    4. 论证深度评估
    """
    
    def __init__(self, config: Optional[SemanticAgentConfig] = None):
        self.config = config or get_semantic_config()
        self._client = None
        self._prompt_template = self._load_prompt_template()
        
    def _load_prompt_template(self) -> str:
        """加载 Prompt 模板"""
        if self.config.prompt_template_path:
            return Path(self.config.prompt_template_path).read_text()
        return DEFAULT_PROMPT_TEMPLATE
    
    def _get_client(self):
        """获取 LLM 客户端"""
        if self._client is not None:
            return self._client
        
        api_config = get_config()
        
        if self.config.provider == "openai":
            import openai
            if not api_config.openai_api_key:
                raise ValueError("请设置 OPENAI_API_KEY 环境变量")
            self._client = openai.OpenAI(api_key=api_config.openai_api_key)
            
        elif self.config.provider == "dashscope":
            import dashscope
            if not api_config.qwen_api_key:
                raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
            dashscope.api_key = api_config.qwen_api_key
            self._client = dashscope
            
        elif self.config.provider == "vllm":
            import openai
            if not api_config.vllm_endpoint:
                raise ValueError("请设置 VLLM_ENDPOINT 环境变量")
            self._client = openai.OpenAI(
                base_url=api_config.vllm_endpoint,
                api_key="dummy"
            )
        else:
            raise ValueError(f"不支持的 LLM 提供者: {self.config.provider}")
        
        return self._client
    
    def format_transcript(self, segments: List[SpeakingSegment]) -> str:
        """将发言片段格式化为对话文本"""
        lines = []
        for seg in segments:
            speaker = seg.speaker_id.replace("SPEAKER_", "学生")
            lines.append(f"{speaker}: {seg.text}")
        return "\n".join(lines)
    
    def analyze(
        self, 
        segments: List[SpeakingSegment],
        context: Optional[str] = None
    ) -> SemanticAnalysisResult:
        """
        分析对话质量
        
        Args:
            segments: 发言片段列表
            context: 可选的上下文信息（如讨论主题）
            
        Returns:
            SemanticAnalysisResult: 分析结果
        """
        # 格式化对话
        transcript = self.format_transcript(segments)
        
        # 如果对话太长，进行压缩
        if self.config.compress_before_llm and len(transcript) > 4000:
            transcript = self._compress_transcript(transcript)
        
        # 构建 Prompt
        prompt = self._prompt_template.format(transcript=transcript)
        
        if context:
            prompt = f"讨论主题/背景：{context}\n\n" + prompt
        
        # 调用 LLM
        response = self._call_llm(prompt)
        
        # 解析结果
        result = self._parse_response(response)
        
        return result
    
    def _call_llm(self, prompt: str) -> str:
        """调用 LLM API"""
        client = self._get_client()
        api_config = get_config()
        
        if self.config.provider == "openai":
            response = client.chat.completions.create(
                model=api_config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=api_config.openai_temperature,
            )
            return response.choices[0].message.content
            
        elif self.config.provider == "dashscope":
            from dashscope import Generation
            response = Generation.call(
                model=self.config.qwen_model,
                prompt=prompt,
            )
            return response.output.text
            
        elif self.config.provider == "vllm":
            response = client.chat.completions.create(
                model=api_config.vllm_model or "default",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        
        return ""
    
    def _parse_response(self, response: str) -> SemanticAnalysisResult:
        """解析 LLM 响应"""
        result = SemanticAnalysisResult()
        
        # 尝试提取 JSON
        try:
            # 去除可能的 markdown 代码块标记
            text = response.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            data = json.loads(text)
            
            result.topic_relevance = float(data.get("topic_relevance", 0.5))
            result.opinion_collisions = int(data.get("opinion_collisions", 0))
            result.turn_taking_pattern = data.get("turn_taking_pattern", "balanced")
            result.argument_depth_score = float(data.get("argument_depth_score", 0.5))
            result.consensus_quality = float(data.get("consensus_quality", 0.5))
            result.evidence = data.get("evidence", [])
            
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}")
            print(f"原始响应: {response}")
            # 使用默认值
            result.evidence = [f"解析失败，原始响应: {response[:200]}..."]
        
        return result
    
    def _compress_transcript(self, transcript: str) -> str:
        """压缩过长的对话转录"""
        # 简化版：取前 2000 字符 + 后 2000 字符
        if len(transcript) <= 4000:
            return transcript
        
        return (
            transcript[:2000] 
            + "\n...[中间部分省略]...\n" 
            + transcript[-2000:]
        )
    
    def analyze_batch(
        self,
        segments: List[SpeakingSegment],
        window_seconds: float = 60.0
    ) -> List[SemanticAnalysisResult]:
        """
        分批分析长对话
        
        Args:
            segments: 发言片段列表
            window_seconds: 时间窗口大小
            
        Returns:
            List[SemanticAnalysisResult]: 每个窗口的分析结果
        """
        if not segments:
            return []
        
        # 按时间窗口分割
        total_duration = segments[-1].end_time
        num_windows = int(total_duration / window_seconds) + 1
        
        results = []
        for i in range(num_windows):
            start = i * window_seconds
            end = (i + 1) * window_seconds
            
            # 提取窗口内的片段
            window_segments = [
                s for s in segments
                if s.start_time >= start and s.end_time <= end
            ]
            
            if window_segments:
                result = self.analyze(window_segments)
                results.append(result)
        
        return results
