# ============================================================
# DeepDiagnose 准确性测试
# 模拟8种患者，绕过LLM直接注入预设答案，验证DSM-5判断逻辑
# ============================================================

import sys
from unittest.mock import MagicMock, patch
from dsm5_questions import ALL_MODULES

# ── 患者档案：定义每个问题ID的预设答案 ──
PATIENT_PROFILES = {

    "患者A：当前重度抑郁": {
        "expected": {"depression": "positive"},
        "answers": {
            "A1": "yes", "A2": "yes", "A3": "yes", "A4": "yes",
            "A5": "no",  "A6": "yes", "A7": "yes", "A8": "yes",
            "A9": "no",  "A10": "yes",
            # 其他模块全部否定
            "A_P1": "no", "M_gate": "no", "HM_gate": "no",
            "PDD_gate": "no",
            "PD_gate": "no", "AG_gate": "no", "SA_gate": "no",
            "GAD_A": "no", "OCD_G1": "no", "OCD_G2": "no",
            "G_screen": "no",
        },
        "note": "A1+A2+A3+A4+A6+A7+A8=7项，A10=yes → 应判阳性"
    },

    "患者B：当前躁狂发作（情绪高涨型）": {
        "expected": {"mania": "positive"},
        "answers": {
            "A1": "no", "A2": "no", "A_P1": "no",
            "M_gate": "yes",
            "M_gate_elev": "yes",  # 情绪高涨型 → 需3项B标准
            "M_B1": "yes", "M_B2": "yes", "M_B3": "yes",
            "M_B4": "no",  "M_B5": "no",  "M_B6": "no",  "M_B7": "no",
            "M_C": "yes", "M_D": "yes",
            "M_P1": "no", "HM_gate": "no", "PDD_gate": "no",
            "PD_gate": "no", "AG_gate": "no", "SA_gate": "no", "GAD_A": "no",
            "OCD_G1": "no", "OCD_G2": "no", "G_screen": "no",
        },
        "note": "M_B1+B2+B3=3项，M_C=yes，M_D=yes → 应判阳性"
    },

    "患者C：当前轻躁狂（非全躁狂）": {
        "expected": {"mania": "skipped", "hypomania": "positive"},
        "answers": {
            "A1": "no", "A2": "no", "A_P1": "no",
            "M_gate": "no", "M_P1": "no",
            "HM_gate": "yes",
            "HM_gate_elev": "yes",  # 情绪高涨型 → 需3项B标准
            "HM_B1": "yes", "HM_B2": "yes", "HM_B3": "yes",
            "HM_B4": "no",  "HM_B5": "no",  "HM_B6": "no",  "HM_B7": "no",
            "HM_C": "yes",
            "HM_P1": "no", "PDD_gate": "no", "PD_gate": "no", "AG_gate": "no",
            "SA_gate": "no", "GAD_A": "no",
            "OCD_G1": "no", "OCD_G2": "no", "G_screen": "no",
        },
        "note": "M_gate=no→躁狂跳过；HM_B1+B2+B3=3项，HM_C=yes → 轻躁狂阳性"
    },

    "患者D：惊恐障碍": {
        "expected": {"panic": "positive"},
        "answers": {
            "A1": "no", "A2": "no", "A_P1": "no",
            "M_gate": "no", "M_P1": "no", "HM_gate": "no", "HM_P1": "no",
            "PDD_gate": "no",
            # 惊恐障碍
            "PD_gate": "yes",
            "PD_S1": "yes", "PD_S2": "yes", "PD_S3": "yes", "PD_S4": "yes",
            "PD_S5": "no",  "PD_S6": "no",  "PD_S7": "no",
            "PD_unexp": "yes", "PD_B": "yes",
            # 其他全否
            "AG_gate": "no", "SA_gate": "no", "GAD_A": "no",
            "OCD_G1": "no", "OCD_G2": "no", "G_screen": "no",
        },
        "note": "S1+S2+S3+S4=4组症状，unexpected=yes，B=yes → 应判阳性"
    },

    "患者E：社交焦虑障碍": {
        "expected": {"social_anxiety": "positive"},
        "answers": {
            "A1": "no", "A2": "no", "A_P1": "no",
            "M_gate": "no", "M_P1": "no", "HM_gate": "no", "HM_P1": "no",
            "PDD_gate": "no",
            "PD_gate": "no", "AG_gate": "no",
            # 社交焦虑
            "SA_gate": "yes",
            "SA_A1": "no", "SA_A2": "no", "SA_A3": "yes",
            "SA_B": "yes", "SA_C": "yes", "SA_D": "yes", "SA_FG": "yes",
            # 其他全否
            "GAD_A": "no",
            "OCD_G1": "no", "OCD_G2": "no", "G_screen": "no",
        },
        "note": "SA_A3=yes（表演场合），SA_B/C/D/FG=yes → 应判阳性"
    },

    "患者F：强迫症": {
        "expected": {"ocd": "positive"},
        "answers": {
            "A1": "no", "A2": "no", "A_P1": "no",
            "M_gate": "no", "M_P1": "no", "HM_gate": "no", "HM_P1": "no",
            "PDD_gate": "no",
            "PD_gate": "no", "AG_gate": "no", "SA_gate": "no", "GAD_A": "no",
            # 强迫症
            "OCD_G1": "yes", "OCD_G2": "yes", "OCD_C": "yes",
            "G_screen": "no",
        },
        "note": "强迫思维+强迫行为均存在，功能损害yes → 应判阳性"
    },

    "患者G：创伤后应激障碍": {
        "expected": {"trauma": "positive"},
        "answers": {
            "A1": "no", "A2": "no", "A_P1": "no",
            "M_gate": "no", "M_P1": "no", "HM_gate": "no", "HM_P1": "no",
            "PDD_gate": "no",
            "PD_gate": "no", "AG_gate": "no", "SA_gate": "no", "GAD_A": "no",
            "OCD_G1": "no", "OCD_G2": "no",
            # PTSD
            "G_screen": "yes",
            "G_B1": "yes", "G_B2": "no",  "G_B3": "no",  "G_B4": "no",  "G_B5": "no",
            "G_C1": "yes", "G_C2": "no",
            "G_D1": "no",  "G_D2": "yes", "G_D3": "no",  "G_D4": "yes",
            "G_D5": "no",  "G_D6": "no",  "G_D7": "no",
            "G_E1": "no",  "G_E2": "no",  "G_E3": "yes", "G_E4": "yes",
            "G_E5": "no",  "G_E6": "no",
            "G_F": "yes",  "G_G": "yes",
        },
        "note": "B1≥1, C1≥1, D2+D4≥2, E3+E4≥2, F=yes, G=yes → 应判阳性"
    },

    "患者H：无任何障碍（健康人）": {
        "expected": {
            "depression": "skipped", "mania": "skipped",
            "hypomania": "skipped", "pdd": "skipped",
            "panic": "skipped", "social_anxiety": "skipped",
            "ocd": "skipped", "trauma": "skipped",
        },
        "answers": {
            "A1": "no", "A2": "no", "A_P1": "no",
            "M_gate": "no", "M_P1": "no", "HM_gate": "no", "HM_P1": "no",
            "PDD_gate": "no",
            "PD_gate": "no", "AG_gate": "no", "SA_gate": "no", "GAD_A": "no",
            "OCD_G1": "no", "OCD_G2": "no", "G_screen": "no",
        },
        "note": "所有门卫问题均否 → 所有模块应被跳过"
    },

    "患者I：持续性抑郁障碍（心境恶劣）": {
        "expected": {"pdd": "positive"},
        "answers": {
            "A1": "no", "A2": "no", "A_P1": "no",
            "M_gate": "no", "M_P1": "no", "HM_gate": "no", "HM_P1": "no",
            # 持续性抑郁障碍
            "PDD_gate": "yes",
            "PDD_B1": "no",  "PDD_B2": "yes", "PDD_B3": "yes",
            "PDD_B4": "yes", "PDD_B5": "no",  "PDD_B6": "yes",
            "PDD_C": "yes", "PDD_H": "yes",
            # 其他全否
            "PD_gate": "no", "AG_gate": "no", "SA_gate": "no", "GAD_A": "no",
            "OCD_G1": "no", "OCD_G2": "no", "G_screen": "no",
        },
        "note": "PDD_B2+B3+B4+B6=4项≥2，PDD_C=yes，PDD_H=yes → 应判阳性"
    },
}

# ── 引擎驱动器 ──

def run_patient(name, profile):
    """用预设答案驱动引擎完成完整评估，返回实际结果"""
    from engine import AssessmentEngine

    # 创建引擎实例（不初始化真实API）
    engine = AssessmentEngine.__new__(AssessmentEngine)
    engine.modules = [m.copy() for m in ALL_MODULES]
    engine.current_module = engine.modules.pop(0) if engine.modules else None
    engine.current_q_index = 0
    engine.answers = {}
    engine.module_results = {}
    engine.is_done = False
    engine.client = MagicMock()  # 不调用真实API

    preset = profile["answers"]

    # 替换 _judge_answer：直接查预设表
    def mock_judge(q, user_answer):
        return preset.get(q["id"], "no")

    # 替换 _ask_question_stream：返回空占位
    def mock_stream(q, prefix=""):
        yield ""

    engine._judge_answer = mock_judge
    engine._ask_question_stream = mock_stream

    # 覆盖 _fast_judge：统一返回 None（强制走 mock_judge）
    engine._fast_judge = lambda x: None

    # 驱动引擎直到结束
    MAX_STEPS = 200
    step = 0
    while not engine.is_done and step < MAX_STEPS:
        step += 1
        q = engine._get_current_question()
        if q is None:
            break

        action, content_gen, is_crisis = engine.process_answer("模拟回答")
        # 消耗 generator 防止挂起
        list(content_gen)

        if action == "summary":
            break

    return engine.module_results


# ── 测试执行 ──

def run_all_tests():
    PASS = "[PASS]"
    FAIL = "[FAIL]"

    total, passed = 0, 0
    all_results = []

    for name, profile in PATIENT_PROFILES.items():
        actual = run_patient(name, profile)
        expected = profile["expected"]

        case_results = []
        case_pass = True

        for mod_id, exp_result in expected.items():
            act_result = actual.get(mod_id, "未运行")
            ok = act_result == exp_result
            if not ok:
                case_pass = False
            case_results.append((mod_id, exp_result, act_result, ok))
            total += 1
            if ok:
                passed += 1

        all_results.append((name, profile["note"], case_pass, case_results, actual))

    # ── 打印报告 ──
    print("\n" + "="*65)
    print("  DeepDiagnose 准确性测试报告")
    print("="*65)

    for name, note, case_pass, case_results, actual in all_results:
        status = PASS if case_pass else FAIL
        print(f"\n{status}  {name}")
        print(f"      说明：{note}")
        for mod_id, exp, act, ok in case_results:
            mark = "+" if ok else "X"
            print(f"      {mark} [{mod_id}]  期望:{exp:<10} 实际:{act}")
        if not case_pass:
            # 显示所有实际结果供调试
            other = {k: v for k, v in actual.items() if k not in {r[0] for r in case_results}}
            if other:
                print(f"      其他模块结果: {other}")

    print("\n" + "-"*65)
    rate = passed / total * 100 if total else 0
    print(f"  总计：{passed}/{total} 通过  准确率：{rate:.1f}%")
    print("="*65 + "\n")

    return passed, total


if __name__ == "__main__":
    passed, total = run_all_tests()
    sys.exit(0 if passed == total else 1)
