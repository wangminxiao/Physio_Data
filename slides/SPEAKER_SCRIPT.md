# Speaker Script — physio-data skill talk
英文讲稿，按**实际页码（共 19 页，含 3 张分隔页）**排列。
每页给 **🔗 Bridge**（承接上一页的过渡句，解决"连贯性"）+ **🎤 Talk**（核心内容）+ 中文提示。
讲法：每页记住 Bridge 那一句 + 一个核心句即可，其余自然展开。

---

## 讲述技巧
- **慢 + 停顿**：每翻页停 1–2 秒。先说 🔗Bridge，再说内容。
- **Bridge 是连贯性的关键**：它把上一页和这一页缝起来。哪怕内容忘了，说出 Bridge，听众也不会迷路。
- **三色追踪条**（右上角 Design/Build/Use）会一直告诉听众"现在在第几部分"，你讲分隔页时可以指它。
- **时长**：~14–16 分钟；加问答 ~20 分钟。只有 10 分钟 → 跳过第 6 页和第 14 页。
- 卡壳： *"Let me put it another way…"* 然后用最简单的词重说。

---

## 1 — Title
🎤 "Hi everyone. Today I'll show you a Claude Code skill I built, called **physio-data**. But it's not really about data. It's about two habits you'll use as researchers: using Claude Code as a **research partner**, and turning a **repeated task** into a reusable *skill*. We'll go through three parts — how I **designed** it, **built** it, and how you **use** it."
> 中文：开场。点明"两个习惯 + 一个例子 + 三部分"。

## 2 — Overview（路线图）
🔗 "So here's the map for the whole talk."
🎤 "Three takeaways. One — use Claude Code for research. Two — capture recurring work as a skill. Three — the example, physio-data, already used on **seven** datasets. See these three colours at the bottom — teal, indigo, amber — that's **Design, Build, Use**. They'll track across the top of every slide, so you always know where we are."
> 中文：介绍三色追踪条 —— 这页要"教会"听众看右上角的进度条。

## 3 — The recurring problem
🔗 "Before any solution — why build this at all?"
🎤 "Every physiological dataset needs the **same** onboarding. The left column never changes. But the **details** on the right change every time — format, sampling rate, IDs, time zones. *Same shape, different details.* Do it by hand each time and you repeat the same mistakes. **That gap is the trigger for a skill.**"
> 中文：左=不变，右=每次都变。这就是该做 skill 的信号。

## 4 — Divider: DESIGN（分隔页）
🔗 "So, part one — **design**. We have a recurring problem; first we design the solution, before any code."
🎤 （指右侧 "IN THIS PART"）"Four steps: the key insight, five principles, the format, and the first meta-lesson."
> 中文：分隔页就读 Bridge + 念一下右边的小目录,给听众"地图"。

## 5 — Key insight: Signals ≠ events
🔗 "The whole design starts from one idea."
🎤 "**Signals and events are different, so we store them differently.** Signals — ECG, PPG — are dense, so they're memory-mapped arrays. Events — labs, vitals — are sparse, so they're a sparse table. The link is just an **index**: each event points into the signal array. We never copy or pad."
> 中文：核心洞见。一句话:不同的数据,不同的存法;对齐用索引。

## 6 — Five rules, written before any code
🔗 "From that one insight, five principles fell out — and I wrote all five **before** any code."
🎤 "Different data, different storage. Alignment is an index. Zero CPU on the hot path. Extensible without breaking old data. And store **raw** values. Writing principles is cheap; skipping them is expensive."
> 中文：五条原则,写在代码之前。（时间紧可跳过此页）

## 7 — One canonical format
🔗 "Those principles produce one concrete thing — the format itself."
🎤 "Each patient-encounter is one folder: two signal files, a timestamp file, a sparse events file. Every event is one tiny record — time, segment index, variable ID, value. The ID **ranges** encode the category, so you can grab 'just the labs' with one comparison. **One format for every dataset.**"
> 中文：这是 skill 要在每个数据集上保证的"契约"。

## 8 — META-LESSON #1: Design it WITH Claude Code
🔗 "And here's the most important design habit — the first thing to steal."
🎤 "This design did **not** come from typing 'write the code.' First **research** — explore the raw files together. Then a **design doc** — make Claude argue trade-offs before code. Then **one patient end-to-end**, and look at the alignment plot with your own eyes before scaling. **Explore and design first — the code that follows is far better.**"
> 中文：第一个方法论高潮。强调"先调研、先设计、肉眼验证"。

## 9 — Divider: BUILD（分隔页）
🔗 "That's design. Part two — **build**. We have the design; now we encode it as a skill Claude can reuse."
🎤 （指右侧目录）"Five steps: what a skill is, what goes inside, the workflow, the second meta-lesson, and how to grow your own."
> 中文：过渡 design→build。

## 10 — A skill is just a SKILL.md file
🔗 "Let's demystify the word 'skill'."
🎤 "It's just a markdown file you drop in a folder. Claude reads the **description** and, when it matches, loads the whole file — no plugin, no code. The **description is the trigger** — the one line that decides whether Claude picks it. The **body** is the expertise, loaded only when it fires."
> 中文：skill = 一个 md 文件;description 是触发器。

## 11 — Inside the body: hard-won knowledge
🔗 "So what do we actually put in that body?"
🎤 "Your hard-won knowledge, as guardrails. The format spec. The ordered workflow. Verification gates. And hard constraints — like 'split by **patient**, not by admission,' because admission splits **leak data**. **A skill is your mistakes, written down as rules.**"
> 中文：把踩过的坑写成规则。

## 12 — The workflow, encoded as stages
🔗 "Here's the workflow those rules enforce."
🎤 "Step zero, then five stages. **Order matters** — we check who has clinical data **before** the expensive waveform extraction. Stage three, in coral, is the **gate**: keep only patients with both. And we verify after every stage, and test on five patients before any full run."
> 中文：顺序很重要;红色 Stage 3 是"门";跑全量前先 --limit 5。

## 13 — META-LESSON #2: Same + different
🔗 "Now the second habit — how one skill covers many different datasets."
🎤 "Split **shared** from **variable**. The SKILL.md and the code package are the same for every dataset — about **90%**. The per-dataset spec — paths, rates, IDs, time zones — is the **10%**. A new dataset means one spec file plus a thin extractor. **That's why one skill covers MIMIC and VitalDB.**"
> 中文：第二个方法论。90% 共享 + 10% 每个数据集不同。

## 14 — How to build & grow a skill
🔗 "One more build slide — how *you* make your own."
🎤 "Don't plan it upfront. Do the task **two or three times**, then lift out the pattern. Scaffold with **skill-creator**. Keep the skill and code **in sync**. And it **compounds** — every dataset made mine sharper. **Every task you repeat is a candidate skill.**"
> 中文：做 2–3 次再抽象;同步更新;越用越好。（时间紧可跳过）

## 15 — Divider: USE（分隔页）
🔗 "That's how it's built. Part three — **use**. We have the skill; now we point it at a brand-new dataset."
🎤 （指右侧目录）"Three steps: how to invoke it, a step-by-step run, and the payoff."
> 中文：过渡 build→use。

## 16 — Two ways to start
🔗 "How do you actually run it?"
🎤 "Two ways. **Explicit** — type slash-physio-data and the dataset name. Or **automatic** — just say 'help me onboard this physiological dataset,' and the description triggers it. **Both lead to the same guided workflow** — which is exactly what the next slide walks through."
> 中文：两种触发方式,都走同一套流程(引到下一页)。

## 17 — A real run-through, with you in the loop
🔗 "Here's that workflow, end to end — and notice where **you** stay in the loop."
🎤 "Claude profiles the raw files. Then — **first human check** — you eyeball the demo alignment plot. **Second check** — you approve the spec. Then it tests on five patients, runs the full extraction in parallel, and writes the splits. **The two coral gates are where a human must look.**"
> 中文:两个红点 = 必须人工看的地方。

## 18 — Proof: one skill, seven datasets
🔗 "So does it actually work across datasets? Here's the proof."
🎤 "**Seven** datasets — PhysioNet, hospital, open surgical, two internal — all in the **same** format with the **same** skill. Yields differ because strictness differs, but the format never changed. **Seven datasets, one format.**"
> 中文：七个数据集、一套格式、十万+ encounter。

## 19 — Takeaways（收尾）
🔗 "Let me close with what to take home."
🎤 "**Find your recurring problem, write the skill once, reap it forever.** Four things: research and design before code; spot the same-shape task; split shared from variable; keep humans at the gates. physio-data is one example — the pattern works for almost anything you repeat. **Thank you — happy to take questions.**"
> 中文：四条要点 + 开放提问。

---

## 难词发音 / 重音
- **canonical** /kəˈNON-i-kəl/ · **memory-mapped** "mem-ory mapped" · **sparse** /spɑːrs/
- **alignment** /əˈLINE-mənt/ · **encounter** /in-COWN-ter/ · **manifest** /MAN-i-fest/
- **extraction** /ik-STRAK-shən/ · **hertz** /hɜːrts/ · **physiological** /ˌfiz-ee-ə-LOJ-i-kəl/（太长就说 "physio data"）

## 问答常用句
- 没听清： *"Sorry, could you repeat the question?"*
- 复述确认： *"So your question is whether… — is that right?"*
- 不知道： *"Good question — I haven't tested that, but my guess is…"*
- 结束： *"Does that answer your question?"*

## 暖场 / 收尾备用
- 暖场互动： *"Quick show of hands — who has written the same data-loading script more than twice?"*
- 一句话总结： *"If you remember one thing: the moment a task repeats, capture it."*
