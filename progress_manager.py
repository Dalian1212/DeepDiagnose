# ============================================================
# 进度保存与加载管理器
# 把评估进度存到本地 JSON 文件，支持中途退出后继续
# ============================================================

import json
import os
from datetime import datetime

SAVE_FILE = os.path.join(os.path.dirname(__file__), "saved_progress.json")


def save_progress(engine, messages: list):
    """把当前评估进度保存到文件"""
    data = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "answers": engine.answers,
        "module_results": engine.module_results,
        "current_module_id": engine.current_module["id"] if engine.current_module else None,
        "current_q_index": engine.current_q_index,
        "remaining_module_ids": [m["id"] for m in engine.modules],
        "is_done": engine.is_done,
        # 只保存最近30条对话（避免文件太大）
        "messages": messages[-30:],
    }
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_progress():
    """从文件加载进度，文件不存在则返回 None"""
    if not os.path.exists(SAVE_FILE):
        return None
    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def delete_progress():
    """删除保存的进度文件（重新开始时调用）"""
    if os.path.exists(SAVE_FILE):
        os.remove(SAVE_FILE)


def has_saved_progress() -> bool:
    return os.path.exists(SAVE_FILE)


def restore_engine(engine, data: dict):
    """
    把保存的数据恢复到 engine 对象中。
    需要在 engine 初始化后调用。
    """
    from dsm5_questions import ALL_MODULES

    engine.answers = data.get("answers", {})
    engine.module_results = data.get("module_results", {})
    engine.current_q_index = data.get("current_q_index", 0)
    engine.is_done = data.get("is_done", False)

    # 恢复当前模块
    current_id = data.get("current_module_id")
    remaining_ids = data.get("remaining_module_ids", [])

    all_modules = {m["id"]: m for m in ALL_MODULES}
    engine.current_module = all_modules.get(current_id)
    engine.modules = [all_modules[mid] for mid in remaining_ids if mid in all_modules]
