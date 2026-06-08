from rich_logger import logger
from pathlib import Path
from multiprocessing import Manager
import json
import copy
import hashlib

class CacheManager:
    def __init__(self, cache_path, prompt_cache=None, cache_hit=None):
        self.cache_path = Path(cache_path)
        # 支持外部传入共享 dict（多进程场景）
        if prompt_cache is not None:
            self.prompt_cache = prompt_cache
            self.cache_hit = cache_hit
            self._own_dict = False
        else:
            self.prompt_cache = Manager().dict()
            self.cache_hit = Manager().dict()
            self.cache_hit.update({"try": 0, "hit": 0})
            self._own_dict = True
        self._load_cache()

    def _get_prompt_hash(self, prompt):
        """计算prompt的稳定hash值，支持字符串、列表、字典类型的prompt"""
        if isinstance(prompt, (list, dict)):
            prompt_str = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
        else:
            prompt_str = str(prompt)
        return hashlib.sha256(prompt_str.encode('utf-8')).hexdigest()

    def _load_cache(self):
        """从文件加载缓存到 dict（不清空已有数据，不重置计数器）"""
        if self.cache_path.exists():
            cache_data = json.loads(self.cache_path.read_text())
            for key, value in cache_data.items():
                if isinstance(value, list):
                    self.prompt_cache[key] = value[0]
                else:
                    self.prompt_cache[key] = value

    def _save_cache(self, print=False):
        self.cache_path.write_text(
            json.dumps(copy.deepcopy(dict(self.prompt_cache)), indent=4, ensure_ascii=False)
        )
        if print:
            logger.info(f"cache size: {len(self.prompt_cache)}, hit: {dict(self.cache_hit)}")

    def hit(self, prompt):
        prompt_hash = self._get_prompt_hash(prompt)
        _hit = prompt_hash in self.prompt_cache
        self.cache_hit["try"] += 1
        self.cache_hit["hit"] += _hit
        return _hit

    def get(self, prompt):
        prompt_hash = self._get_prompt_hash(prompt)
        return self.prompt_cache[prompt_hash]

    def set(self, prompt, resp):
        prompt_hash = self._get_prompt_hash(prompt)
        self.prompt_cache[prompt_hash] = resp
