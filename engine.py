# ============================================================
# 评估引擎：严格按照 SCID-5-CV 访谈手册逻辑
# ============================================================

from openai import OpenAI
import json
from dsm5_questions import ALL_MODULES

# 阶段中文名
MODULE_NAMES = {
    "depression":       "当前抑郁发作评估（模块A）",
    "past_depression":  "既往抑郁发作筛查（模块A）",
    "mania":            "当前躁狂发作评估（模块A）",
    "past_mania":       "既往躁狂发作筛查（模块A）",
    "hypomania":        "当前轻躁狂发作评估（模块A）",
    "past_hypomania":   "既往轻躁狂发作筛查（模块A）",
    "pdd":              "持续性抑郁障碍评估（模块A）",
    "panic":            "惊恐障碍评估（模块F）",
    "agoraphobia":      "广场恐惧症评估（模块F）",
    "social_anxiety":   "社交焦虑障碍评估（模块F）",
    "anxiety":          "广泛性焦虑评估（模块F）",
    "ocd":              "强迫症评估（模块G）",
    "trauma":           "创伤后应激评估（模块G）",
    "summary":          "生成评估报告",
    "done":             "评估完成",
}

# ── 提示词 ──

ASK_PROMPT = """你是 DeepDiagnose，一个专业、温和的 AI 心理评估助手。

现在正在进行【{module_name}】阶段。

你需要用自然、温和的语气问以下这个问题（不要照念原文，用口语化方式表达）：
问题内容：{question}
追问提示：{followup}

规则：
- 每次只问这一个问题，不要延伸
- 语气温和，像关心朋友的人
- 不超过80字
- 用中文
"""

JUDGE_PROMPT = """你是一个心理评估的判断助手。

用户对以下问题的回答是：
问题：{question}
用户回答：{user_answer}

请判断用户的回答对应的 DSM-5 标准「{criterion}」是否符合（即"是"还是"否"）。

只返回 JSON，不要其他文字：
{{"result": "yes" 或 "no", "reason": "一句话说明"}}

注意：
- "是"表示该症状存在且达到临床意义
- "否"表示症状不存在或程度不足
- 如果用户回答不确定，倾向于"no"
"""

SUMMARY_PROMPT = """你是 DeepDiagnose，根据以下详细的评估数据生成一份专业的评估摘要报告。

{module_details}

对话片段：
{conversation}

请生成如下格式的报告，要求具体、有依据（直接引用用户确认的症状）：

---
📋 **DeepDiagnose 评估摘要**

**您提到的主要困扰：**
（根据对话内容，用1-2句话描述用户的主要诉求）

**评估发现：**
（对每个模块，具体列出用户确认了哪些症状，是否达到关注阈值。用"您提到了…"的方式表述确认的症状，语气温和）

**建议：**
（根据评估结果，具体说明是否建议寻求专业帮助，推荐什么类型的资源）

**重要声明：**
本报告仅供参考，不构成任何医学诊断。如有需要，请咨询专业的心理健康人士或精神科医生。
---
"""

CRISIS_MESSAGE = """
⚠️ **我注意到你提到了一些让我非常担心的话。**

你并不孤单。如果你现在有伤害自己的想法，请立即联系专业帮助：

- **北京心理危机研究与干预中心**：010-82951332
- **全国心理援助热线**：400-161-9995
- **生命热线**：400-821-1215
- **紧急情况请拨打**：120

我在这里陪着你，但专业的帮助更重要。💙
"""


class AssessmentEngine:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=60.0
        )
        self.modules = ALL_MODULES[:]          # 待评估的模块列表
        self.current_module = None             # 当前模块
        self.current_q_index = 0              # 当前问题索引
        self.answers = {}                      # {question_id: "yes"/"no"}
        self.module_results = {}               # {module_id: "positive"/"negative"/"skipped"}
        self.gate_failed = False               # 当前模块是否被门卫跳过
        self.is_done = False
        self._load_next_module()

    # ── 模块管理 ──

    # 当前发作阳性时，自动跳过对应的既往发作模块
    _SKIP_IF_CURRENT_POSITIVE = {
        "past_depression": "depression",
        "past_mania":      "mania",
        "past_hypomania":  "hypomania",
    }

    def _load_next_module(self):
        """加载下一个待评估的模块，自动跳过已被当前阳性结果覆盖的既往发作模块"""
        while self.modules:
            self.current_module = self.modules.pop(0)
            self.current_q_index = 0
            self.gate_failed = False

            # 若当前发作已阳性，直接跳过对应的既往发作模块
            current_id = self.current_module["id"]
            blocks = self._SKIP_IF_CURRENT_POSITIVE.get(current_id)
            if blocks and self.module_results.get(blocks) == "positive":
                self.module_results[current_id] = "skipped"
                continue  # 继续加载下一个模块
            return  # 找到合适的模块，停止

        # 没有剩余模块
        self.current_module = None
        self.is_done = True

    def get_stage_name(self) -> str:
        if self.is_done:
            return MODULE_NAMES["done"]
        if self.current_module is None:
            return MODULE_NAMES["summary"]
        return MODULE_NAMES.get(self.current_module["id"], "评估中")

    def get_progress(self) -> dict:
        """返回进度信息供界面展示"""
        done = list(self.module_results.keys())
        current = self.current_module["id"] if self.current_module else "summary"
        remaining = [m["id"] for m in self.modules]
        return {
            "current": current,
            "done": done,
            "remaining": remaining,
            "results": self.module_results,
        }

    # ── 核心流程 ──

    def _get_current_question(self):
        """获取当前需要问的问题"""
        if not self.current_module:
            return None
        qs = self.current_module["questions"]
        if self.current_q_index < len(qs):
            return qs[self.current_q_index]
        return None

    def _check_gate(self) -> bool:
        """
        检查门卫条件，决定是否跳过当前模块。
        返回 True 表示应该跳过（gate failed）。
        """
        gate = self.current_module.get("gate", {})
        rule = gate.get("rule", "")
        ids = gate.get("ids", [])

        if rule == "both_no_skip":
            # 所有门卫题都是"否"才跳过（A1和A2都否→跳过抑郁模块）
            return all(self.answers.get(i, "no") == "no" for i in ids)
        elif rule == "any_no_skip":
            # 任一门卫题是"否"就跳过（GAD_A或GAD_B为否→跳过）
            return any(self.answers.get(i, "no") == "no" for i in ids)
        return False

    def _check_threshold(self) -> bool:
        """检查是否达到诊断阈值"""
        threshold = self.current_module.get("threshold", {})

        # 简单阈值（抑郁/焦虑/躁狂）
        if "min_yes" in threshold:
            items = threshold.get("all_items", [])
            yes_count = sum(1 for i in items if self.answers.get(i) == "yes")
            must_include = threshold.get("must_include_one_of")
            if must_include:
                has_required = any(self.answers.get(i) == "yes" for i in must_include)
                if not has_required:
                    return False
            # 条件阈值：纯易激惹型需要更多症状（DSM-5 规定）
            min_yes = threshold["min_yes"]
            cond = threshold.get("min_yes_if_no")
            if cond and self.answers.get(cond["question"]) == "no":
                min_yes = cond["then"]
            if yes_count < min_yes:
                return False
            # 必须全部为"是"的项目（如功能损害、病程标准）
            required_yes = threshold.get("required_yes", [])
            if required_yes:
                if not all(self.answers.get(i) == "yes" for i in required_yes):
                    return False
            return True

        # 分组阈值（PTSD）
        if "groups" in threshold:
            for group_name, group in threshold["groups"].items():
                yes_count = sum(1 for i in group["items"] if self.answers.get(i) == "yes")
                if yes_count < group["min_yes"]:
                    return False
            # 检查必须为"是"的项目（如持续时间、功能损害）
            required_yes = threshold.get("required_yes", [])
            if required_yes:
                if not all(self.answers.get(i) == "yes" for i in required_yes):
                    return False
            return True

        return False

    def _fast_judge(self, user_answer: str):
        """规则判断：对明显的回答直接返回，跳过 LLM 调用。返回 None 表示需要 LLM。"""
        text = user_answer.strip().rstrip("。！？，,.!? ")
        no_words  = ["没有", "没", "不会", "不太", "从没", "从未", "不曾", "否", "没有过", "没经历", "没经历过"]
        yes_words = ["有过", "有", "是的", "对", "确实", "经常", "总是", "嗯嗯", "嗯", "是", "会", "有时", "偶尔"]
        for w in no_words:
            if text == w or text.startswith(w + "，") or text.startswith(w + " ") or text.startswith(w + "。"):
                return "no"
        for w in yes_words:
            if text == w or text.startswith(w + "，") or text.startswith(w + " ") or text.startswith(w + "。"):
                return "yes"
        return None

    def _ask_question_stream(self, question: dict, prefix: str = ""):
        """流式生成问题文本，返回 generator"""
        if prefix:
            yield prefix + "\n\n"
        prompt = ASK_PROMPT.format(
            module_name=self.get_stage_name(),
            question=question["ask"],
            followup=question.get("followup", "")
        )
        stream = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _ask_question(self, question: dict) -> str:
        """单独生成问题文本（非流式，用于开场）"""
        return "".join(self._ask_question_stream(question))

    def _judge_answer(self, question: dict, user_answer: str) -> str:
        """LLM 判断 yes/no（规则无法判断时使用）"""
        prompt = JUDGE_PROMPT.format(
            question=question["ask"],
            user_answer=user_answer,
            criterion=question["criterion"]
        )
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            content = response.choices[0].message.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                return data.get("result", "no")
        except Exception:
            pass
        return "no"

    # 各模块症状严重程度分级标准（基于确认症状数）
    _SEVERITY_RULES = {
        "depression":     [(5, "轻度"), (7, "中度"), (9, "重度")],   # A1-A9，满5项起
        "mania":          [(3, "轻/中度"), (6, "重度")],             # M_B1-B7，满3项起
        "hypomania":      [(3, "轻度")],                             # 轻躁狂按定义为轻度
        "anxiety":        [(3, "轻度"), (5, "中度"), (6, "重度")],   # GAD_C1-C6
        "panic":          [(4, "轻度"), (6, "中度"), (7, "重度")],   # PD_S1-S7
        "social_anxiety": [(1, "轻度"), (2, "中度"), (3, "重度")],   # SA_A类
        "agoraphobia":    [(2, "轻度"), (4, "中度"), (5, "重度")],   # AG_A1-A5
        "ocd":            [(1, "轻/中度"), (2, "重度")],             # OCD_G1+G2
        "trauma": {
            # PTSD按各组症状数综合评级
            "B": [(1, ""), (3, ""), (5, "")],
            "D": [(2, "轻度"), (4, "中度"), (7, "重度")],
        },
    }

    def _get_severity(self, mod_id: str, module_data: dict) -> str:
        """根据确认的症状数量返回严重程度标签"""
        rules = self._SEVERITY_RULES.get(mod_id)
        if not rules:
            return ""

        if mod_id == "trauma":
            # PTSD 用D组症状数估算
            d_items = ["G_D1","G_D2","G_D3","G_D4","G_D5","G_D6","G_D7"]
            count = sum(1 for i in d_items if self.answers.get(i) == "yes")
            thresholds = [(2, "轻度"), (4, "中度"), (7, "重度")]
        else:
            symptom_items = module_data["threshold"].get("all_items", [])
            # 对于分组模块（PTSD已在上面处理），取所有组的items
            if "groups" in module_data["threshold"]:
                symptom_items = [i for g in module_data["threshold"]["groups"].values()
                                 for i in g["items"]]
            count = sum(1 for i in symptom_items if self.answers.get(i) == "yes")
            thresholds = rules

        label = thresholds[0][1]
        for threshold, severity in thresholds:
            if count >= threshold:
                label = severity
        return f"（{label}，确认 {count} 项症状）"

    def generate_summary(self, conversation_history: list) -> str:
        """生成最终评估报告（含具体症状列表和严重程度）"""
        # 构建问题索引（id → 问题dict）
        q_lookup = {}
        for module in ALL_MODULES:
            for q in module["questions"]:
                q_lookup[q["id"]] = q

        # 逐模块整理确认/否认的症状
        module_details = ""
        for mod_id, result in self.module_results.items():
            module_data = next((m for m in ALL_MODULES if m["id"] == mod_id), None)
            if not module_data:
                continue
            name = MODULE_NAMES.get(mod_id, mod_id)

            if result == "positive":
                severity = self._get_severity(mod_id, module_data)
                result_label = f"达到关注阈值 {severity}"
            elif result == "negative":
                result_label = "未达到关注阈值"
            else:
                result_label = "初步筛查已跳过"

            module_details += f"\n【{name}】总体结果：{result_label}\n"

            confirmed, denied = [], []
            for q in module_data["questions"]:
                qid = q["id"]
                if qid not in self.answers:
                    continue
                if self.answers[qid] == "yes":
                    confirmed.append(f"  ✓ {q['criterion']}")
                else:
                    denied.append(f"  ✗ {q['criterion']}")

            if confirmed:
                module_details += "用户确认存在：\n" + "\n".join(confirmed) + "\n"
            if denied:
                module_details += "用户否认：\n" + "\n".join(denied) + "\n"

        # 截取对话片段
        conv_text = "\n".join([
            f"{'用户' if m['role'] == 'user' else 'AI'}: {m['content']}"
            for m in conversation_history if m["role"] != "system"
        ])[-2000:]

        prompt = SUMMARY_PROMPT.format(
            module_details=module_details,
            conversation=conv_text
        )
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.5
        )
        return response.choices[0].message.content

    # ── 主入口 ──

    def get_next_question_text(self) -> str:
        """获取下一个问题的自然语言表述（用于开场和阶段切换时主动提问）"""
        q = self._get_current_question()
        if q is None:
            return ""
        return self._ask_question(q)

    def process_answer(self, user_answer: str) -> tuple:
        """
        处理用户对当前问题的回答。
        返回：(next_action, content, is_crisis)
        next_action:
          "ask_next"   → 继续问下一题，content 是下一题的问法
          "module_done" → 当前模块结束，content 是过渡提示
          "summary"    → 所有模块结束，content 是报告
          "crisis"     → 危机响应
        """
        q = self._get_current_question()
        if q is None:
            return "summary", iter([""]), False

        is_crisis = q.get("is_crisis", False)

        # 1. 优先规则判断，再 LLM 兜底
        judgment = self._fast_judge(user_answer)
        if judgment is None:
            judgment = self._judge_answer(q, user_answer)
        self.answers[q["id"]] = judgment

        # 危机标志（先记录，等状态推进后再决定是否触发）
        crisis_triggered = is_crisis and judgment == "yes"

        # 2. 移动到下一题
        self.current_q_index += 1

        # 3. 检查门卫
        gate_ids = self.current_module.get("gate", {}).get("ids", [])
        all_gate_answered = all(qid in self.answers for qid in gate_ids)
        if all_gate_answered and self._check_gate():
            self.module_results[self.current_module["id"]] = "skipped"
            self._load_next_module()
            if crisis_triggered:
                return "crisis", iter([CRISIS_MESSAGE]), True
            if self.is_done or self.current_module is None:
                return "summary", iter([""]), False
            nq = self._get_current_question()
            return "ask_next", self._ask_question_stream(nq, "好的，我们来聊聊另一个方面。"), False

        # 4. 检查是否问完当前模块
        if self.current_q_index >= len(self.current_module["questions"]):
            completed_q_count = len(self.current_module["questions"])
            passed = self._check_threshold()
            self.module_results[self.current_module["id"]] = "positive" if passed else "negative"
            self._load_next_module()
            if crisis_triggered:
                return "crisis", iter([CRISIS_MESSAGE]), True
            if self.is_done or self.current_module is None:
                return "summary", iter([""]), False
            nq = self._get_current_question()
            if completed_q_count == 1:
                prefix = ""
            else:
                prefix = "谢谢你告诉我这些。我们再了解一些其他方面。"
            return "module_done", self._ask_question_stream(nq, prefix), False

        # 5. 继续当前模块下一题（流式）
        if crisis_triggered:
            return "crisis", iter([CRISIS_MESSAGE]), True
        nq = self._get_current_question()
        return "ask_next", self._ask_question_stream(nq), False
