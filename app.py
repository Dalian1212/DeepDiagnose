import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
from engine import AssessmentEngine, CRISIS_MESSAGE, MODULE_NAMES
from progress_manager import (
    save_progress, load_progress, delete_progress,
    has_saved_progress, restore_engine
)

load_dotenv()
# 本地用 .env，云端用 Streamlit Secrets
try:
    API_KEY = st.secrets["DEEPSEEK_API_KEY"]
    IS_CLOUD = True   # 云端部署，禁用文件进度保存
except Exception:
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    IS_CLOUD = False  # 本地运行，可以用文件保存进度

CRISIS_KEYWORDS = ["自杀", "自伤", "割腕", "不想活", "去死", "结束生命", "轻生", "想死", "活不下去"]

def check_crisis(text: str) -> bool:
    return any(kw in text for kw in CRISIS_KEYWORDS)

def get_partial_report(engine, messages: list) -> str:
    """生成已完成模块的中途报告"""
    if not engine.module_results:
        return "目前还没有完成任何模块的评估，暂时无法生成报告。"
    return engine.generate_summary(messages)

# ── 页面设置 ──
st.set_page_config(
    page_title="DeepDiagnose - AI 心理评估",
    page_icon="🧠",
    layout="wide"
)


# ── 侧边栏 ──
with st.sidebar:
    st.title("📊 评估进度")
    st.divider()

    if "engine" in st.session_state:
        progress = st.session_state.engine.get_progress()

        # ── 进度指示器 ──
        total_modules = 12
        done_count = len(progress["done"])
        answered_q = len(st.session_state.engine.answers)
        st.progress(done_count / total_modules, text=f"模块进度 {done_count}/{total_modules}")
        st.caption(f"已回答 {answered_q} 题")
        st.divider()

        # ── 模块A：心境发作（可展开/折叠）──
        MODULE_A_IDS = ["depression", "past_depression", "mania", "past_mania",
                        "hypomania", "past_hypomania", "pdd"]
        MODULE_A_LABELS = {
            "depression":      "当前抑郁发作",
            "past_depression": "既往抑郁发作",
            "mania":           "当前躁狂发作",
            "past_mania":      "既往躁狂发作",
            "hypomania":       "当前轻躁狂发作",
            "past_hypomania":  "既往轻躁狂发作",
            "pdd":             "持续性抑郁障碍",
        }

        # 判断模块A整体状态，用于大类标题显示
        a_done = [mid for mid in MODULE_A_IDS if mid in progress["done"]]
        a_current_in_module = progress["current"] in MODULE_A_IDS
        a_all_done = len(a_done) == len(MODULE_A_IDS)

        if a_current_in_module:
            a_header = "**▶ A. 心境发作** ← 进行中"
        elif a_all_done:
            a_header = "✅ ~~A. 心境发作~~"
        elif a_done:
            a_header = "**A. 心境发作**（部分完成）"
        else:
            a_header = "○ A. 心境发作"

        # 默认展开：模块A正在进行时自动展开
        with st.expander(a_header, expanded=a_current_in_module or a_all_done):
            for mid in MODULE_A_IDS:
                label = MODULE_A_LABELS[mid]
                if mid in progress["done"]:
                    result = progress["results"].get(mid, "")
                    tag = "（发现症状）" if result == "positive" else "（未发现）" if result == "negative" else "（已跳过）"
                    st.markdown(f"　✅ ~~{label}~~ {tag}")
                elif mid == progress["current"]:
                    st.markdown(f"　**▶ {label}** ← 当前")
                else:
                    st.markdown(f"　○ {label}")

        # ── 模块F：焦虑障碍（可展开/折叠）──
        MODULE_F_IDS = ["panic", "agoraphobia", "social_anxiety", "anxiety"]
        MODULE_F_LABELS = {
            "panic":          "惊恐障碍",
            "agoraphobia":    "广场恐惧症",
            "social_anxiety": "社交焦虑障碍",
            "anxiety":        "广泛性焦虑障碍",
        }
        f_done = [mid for mid in MODULE_F_IDS if mid in progress["done"]]
        f_current_in_module = progress["current"] in MODULE_F_IDS
        f_all_done = len(f_done) == len(MODULE_F_IDS)

        if f_current_in_module:
            f_header = "**▶ F. 焦虑障碍** ← 进行中"
        elif f_all_done:
            f_header = "✅ ~~F. 焦虑障碍~~"
        elif f_done:
            f_header = "**F. 焦虑障碍**（部分完成）"
        else:
            f_header = "○ F. 焦虑障碍"

        with st.expander(f_header, expanded=f_current_in_module or f_all_done):
            for mid in MODULE_F_IDS:
                label = MODULE_F_LABELS[mid]
                if mid in progress["done"]:
                    result = progress["results"].get(mid, "")
                    tag = "（发现症状）" if result == "positive" else "（未发现）" if result == "negative" else "（已跳过）"
                    st.markdown(f"　✅ ~~{label}~~ {tag}")
                elif mid == progress["current"]:
                    st.markdown(f"　**▶ {label}** ← 当前")
                else:
                    st.markdown(f"　○ {label}")

        # ── 模块G：强迫症与创伤后应激（可展开/折叠）──
        MODULE_G_IDS = ["ocd", "trauma"]
        MODULE_G_LABELS = {
            "ocd":    "强迫症",
            "trauma": "创伤后应激障碍",
        }
        g_done = [mid for mid in MODULE_G_IDS if mid in progress["done"]]
        g_current_in_module = progress["current"] in MODULE_G_IDS
        g_all_done = len(g_done) == len(MODULE_G_IDS)

        if g_current_in_module:
            g_header = "**▶ G. 强迫症与创伤后应激** ← 进行中"
        elif g_all_done:
            g_header = "✅ ~~G. 强迫症与创伤后应激~~"
        elif g_done:
            g_header = "**G. 强迫症与创伤后应激**（部分完成）"
        else:
            g_header = "○ G. 强迫症与创伤后应激"

        with st.expander(g_header, expanded=g_current_in_module or g_all_done):
            for mid in MODULE_G_IDS:
                label = MODULE_G_LABELS[mid]
                if mid in progress["done"]:
                    result = progress["results"].get(mid, "")
                    tag = "（发现症状）" if result == "positive" else "（未发现）" if result == "negative" else "（已跳过）"
                    st.markdown(f"　✅ ~~{label}~~ {tag}")
                elif mid == progress["current"]:
                    st.markdown(f"　**▶ {label}** ← 当前")
                else:
                    st.markdown(f"　○ {label}")

        st.divider()

        # 中途查看结果 / 下载按钮
        if progress["done"] and not st.session_state.engine.is_done:
            if st.button("📋 查看已完成模块的结果"):
                st.session_state.show_partial_report = True

            # 下载中途报告
            if st.session_state.get("partial_download_data"):
                st.download_button(
                    label="⬇️ 下载中途报告",
                    data=st.session_state.partial_download_data,
                    file_name="deepdiagnose_partial_report.md",
                    mime="text/markdown",
                )
            else:
                if st.button("📥 生成中途报告以下载"):
                    with st.spinner("生成中..."):
                        try:
                            st.session_state.partial_download_data = get_partial_report(
                                st.session_state.engine, st.session_state.messages
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"生成失败：{e}")

        # 评估完成后查看完整报告
        if st.session_state.engine.is_done and st.session_state.get("final_report"):
            if st.button("📄 查看完整报告", type="primary"):
                st.session_state.show_report_page = True
                st.rerun()

        # 保存进度按钮（仅本地可用）
        if not IS_CLOUD and not st.session_state.engine.is_done:
            if st.button("💾 保存进度并退出"):
                save_progress(st.session_state.engine, st.session_state.messages)
                st.success("进度已保存！下次打开可以继续。")

        st.caption("💡 基于 DSM-5 SCID-5-CV 标准")

    st.divider()
    if st.button("🔄 重新开始"):
        delete_progress()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

st.title("🧠 DeepDiagnose")
st.caption("基于 DSM-5 SCID-5-CV 的 AI 心理健康评估系统 | 仅供参考，不替代专业诊断")

# 移动端进度提示（侧边栏在手机上需手动展开，这里补一行）
if "engine" in st.session_state and not st.session_state.engine.is_done:
    progress = st.session_state.engine.get_progress()
    done_count = len(progress["done"])
    st.caption(f"📊 评估进度：{done_count}/12 模块 | 已回答 {len(st.session_state.engine.answers)} 题")

st.divider()

# ── 免责声明/知情同意 ──
if not st.session_state.get("disclaimer_accepted"):
    st.markdown("""
## 📋 使用须知

在开始评估之前，请仔细阅读以下说明：

**本工具是什么：**
DeepDiagnose 基于 DSM-5《精神疾病诊断与统计手册》的 SCID-5-CV 结构化临床访谈框架，通过 AI 对话形式对常见心理健康状况进行**初步筛查**。

**本工具不是什么：**
- ❌ 不是临床诊断，不能替代精神科医生或心理咨询师的专业判断
- ❌ 评估结果仅供个人参考，不具有医学诊断效力
- ❌ 不适用于精神科急症或危机干预场景

**您的数据：**
- 本次对话数据仅保存在当前会话中，关闭页面后自动清除，不会被用于任何其他用途

**如果您正处于危机：**
如果您现在有伤害自己或他人的想法，请立即拨打心理援助热线：
- 🆘 北京心理危机研究与干预中心：**010-82951332**
- 🆘 全国心理援助热线：**400-161-9995**
""")

    st.warning("⚠️ 继续使用即表示您已了解本工具的使用限制，并同意以上说明。")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("✅ 我已了解，开始评估", type="primary", use_container_width=True):
            st.session_state.disclaimer_accepted = True
            st.rerun()
    st.stop()

# ── 报告全屏页面 ──
if st.session_state.get("show_report_page") and st.session_state.get("final_report"):
    st.title("📋 DeepDiagnose 评估报告")
    st.divider()
    st.markdown(st.session_state.final_report)
    st.divider()
    col1, col2 = st.columns([1, 4])
    with col1:
        st.download_button(
            label="⬇️ 下载报告",
            data=st.session_state.final_report,
            file_name="deepdiagnose_report.md",
            mime="text/markdown",
        )
    with col2:
        if st.button("← 返回对话"):
            st.session_state.show_report_page = False
            st.rerun()
    st.stop()

# ── 初始化：检查是否有保存的进度（仅本地）──
if "engine" not in st.session_state:
    if not IS_CLOUD and has_saved_progress() and "resume_choice" not in st.session_state:
        saved = load_progress()
        st.info(f"🔔 发现上次保存的进度（保存于 {saved.get('saved_at', '未知时间')}）")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ 继续上次的评估"):
                st.session_state.resume_choice = "continue"
                st.rerun()
        with col2:
            if st.button("🆕 重新开始"):
                st.session_state.resume_choice = "new"
                st.rerun()
        st.stop()

    engine = AssessmentEngine(API_KEY)

    if not IS_CLOUD and st.session_state.get("resume_choice") == "continue":
        saved = load_progress()
        if saved:
            restore_engine(engine, saved)
            st.session_state.messages = saved.get("messages", [])
            st.session_state.engine = engine
            st.session_state.resume_choice = None
            st.rerun()

    st.session_state.engine = engine

if "messages" not in st.session_state:
    st.session_state.messages = []
    with st.spinner("正在连接..."):
        try:
            opening = "你好，我是 DeepDiagnose，一个 AI 心理健康评估助手。我会通过一些问题来了解你最近的状态。请放心，你的回答只用于评估参考。\n\n"
            first_q = st.session_state.engine.get_next_question_text()
            st.session_state.messages.append({"role": "assistant", "content": opening + first_q})
        except Exception as e:
            st.error(f"连接失败：{e}")

# ── 显示对话历史 ──
for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "🙂"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ── 中途报告（在聊天底部生成，用户可见）──
if st.session_state.get("show_partial_report"):
    with st.chat_message("assistant", avatar="🤖"):
        # 只在第一次生成，后续刷新直接读缓存，避免重复调用 API
        if "partial_report_cache" not in st.session_state:
            with st.spinner("正在生成已完成模块的报告，请稍候..."):
                try:
                    st.session_state.partial_report_cache = get_partial_report(
                        st.session_state.engine,
                        st.session_state.messages
                    )
                except Exception as e:
                    st.error(f"生成报告失败：{e}")
                    st.session_state.partial_report_cache = None
        if st.session_state.partial_report_cache:
            st.markdown(st.session_state.partial_report_cache)
    if st.button("关闭报告，继续评估"):
        st.session_state.show_partial_report = False
        st.session_state.pop("partial_report_cache", None)
        st.rerun()
    st.stop()

# ── 用户输入 ──
placeholder = "请输入你的回答..." if not st.session_state.engine.is_done else "评估已完成，如有问题可继续提问..."
user_input = st.chat_input(placeholder)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(user_input)

    # 危机关键词预检（不中断评估，继续正常流程）
    if check_crisis(user_input):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(CRISIS_MESSAGE)
        st.session_state.messages.append({"role": "assistant", "content": CRISIS_MESSAGE})

    # 评估完成后的普通问答
    if st.session_state.engine.is_done:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("思考中..."):
                try:
                    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com", timeout=60.0)
                    history = [{"role": m["role"], "content": m["content"]}
                               for m in st.session_state.messages[-10:]]
                    resp = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "system", "content": "你是 DeepDiagnose，评估已完成，现在温和专业地回答用户问题。"}] + history,
                        max_tokens=300
                    )
                    reply = resp.choices[0].message.content
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.warning(f"⏱️ 网络超时，请重试。")
        st.stop()

    # 正常评估流程
    with st.chat_message("assistant", avatar="🤖"):
        try:
            action, content_stream, is_crisis = st.session_state.engine.process_answer(user_input)
        except TimeoutError:
            st.warning("⏱️ 网络超时，请再发一次消息重试。")
            st.session_state.messages.pop()
            st.stop()
        except Exception as e:
            st.error(f"发生错误：{e}")
            st.session_state.messages.pop()
            st.stop()

        if is_crisis:
            # 危机响应：展示帮助信息，然后继续问下一题
            st.markdown(CRISIS_MESSAGE)
            st.session_state.messages.append({"role": "assistant", "content": CRISIS_MESSAGE})
            if not st.session_state.engine.is_done:
                with st.spinner(""):
                    next_q_text = st.session_state.engine.get_next_question_text()
                if next_q_text:
                    st.markdown(next_q_text)
                    st.session_state.messages.append({"role": "assistant", "content": next_q_text})
            if not IS_CLOUD:
                save_progress(st.session_state.engine, st.session_state.messages)

        elif action == "summary":
            st.markdown("📋 **所有模块评估完成，正在生成报告，请稍候...**")
            with st.spinner("报告生成中，约需10-20秒..."):
                try:
                    report = st.session_state.engine.generate_summary(st.session_state.messages)
                    st.session_state.final_report = report
                    st.session_state.messages.append({"role": "assistant", "content": report})
                    st.session_state.engine.is_done = True
                    st.session_state.show_report_page = True
                    if not IS_CLOUD:
                        delete_progress()
                except Exception as e:
                    fallback = "评估已完成。建议将结果与专业心理咨询师分享，获取更准确的评估。"
                    st.markdown(fallback)
                    st.session_state.messages.append({"role": "assistant", "content": fallback})

        else:
            # 流式显示，边生成边输出
            full_text = st.write_stream(content_stream)
            st.session_state.messages.append({"role": "assistant", "content": full_text})
            if not IS_CLOUD:
                save_progress(st.session_state.engine, st.session_state.messages)

    st.rerun()
