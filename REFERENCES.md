# REFERENCES — литература по Machine Unlearning

Подборка под цели проекта: SOTA-анализ, CPU-бейзлайны (SISA, influence functions),
gradient-based и Fisher-методы, LLM unlearning (RMU, DPO, task arithmetic),
а также оценка (MIA / utility / efficiency) и анализ геометрии весов и представлений,
который уже ведётся в `experiments/convolutional`.

---

## Содержание
1. [Обзорные статьи (surveys)](#1-обзорные-статьи)
2. [Фундаментальные / точные методы](#2-фундаментальные--точные-методы)
3. [Приближённое разобучение для глубоких сетей](#3-приближённое-разобучение-для-глубоких-сетей)
4. [LLM Unlearning](#4-llm-unlearning)
5. [Оценка: MIA, бенчмарки, верификация](#5-оценка-mia-бенчмарки-верификация)
6. [Геометрия весов и представлений моделей](#6-геометрия-весов-и-представлений-моделей)
7. [Continual learning и ортогональные подпространства (фундамент OSU-LoRA)](#7-continual-learning-и-ортогональные-подпространства)
8. [Редактирование знаний](#8-редактирование-знаний)
9. [Как это ложится на фазы проекта](#9-как-это-ложится-на-фазы-проекта)

---

## 1. Обзорные статьи

**A Survey of Machine Unlearning** — Nguyen, Huynh, Nguyen, Liew, Yin, Nguyen, 2022 — https://arxiv.org/abs/2209.02299 · https://ar5iv.labs.arxiv.org/html/2209.02299
*О чём:* самая ранняя системная классификация области: точные vs приближённые методы, централизованное и federated unlearning, требования к «забыванию» (точность, время, стоимость). *Выводы:* формализует постановку «удалить влияние точки $x$ так, будто её не было в обучении» и показывает, что exact-unlearning стоит $\mathcal{O}(\text{retrain})$, а approximate — компромисс по utility. **Взять:** язык формализации и таксономию для литературного обзора (месяц 1).

**Machine Unlearning: A Survey** — Xu, Zhu, Zhang, Zhou, Yu, 2023 (ACM CSUR) — https://arxiv.org/abs/2306.03558
*О чём:* обзор методов по типу модели и типу запроса (point/class/feature), включая SISA-подобные, gradient-based и certified. *Выводы:* явно разводит *unlearning efficacy* и *model utility* как независимые оси, показывает отсутствие единой метрики. **Взять:** структура для сравнительной таблицы методов.

**Machine Unlearning: A Comprehensive Survey** — Wang, Tian, Zhang, Yu, et al., 2024 — https://arxiv.org/abs/2405.07406 · https://ar5iv.labs.arxiv.org/html/2405.07406
*О чём:* четыре блока: классические методы, verification, домены (CV/NLP/recsys/graph/federated), privacy & security. *Выводы:* unlearning проверяется не только точностью — нужен отдельный слой верификации (MIA, backdoor, robustness). **Взять:** источники по доменам и по безопасности.

**A Comprehensive Survey of Machine Unlearning Techniques for Large Language Models** — Geng, Li, Woisetschläger, Chen, Wang, Nakov, Jacobsen, Karray, 2025 — https://arxiv.org/abs/2503.01854
*О чём:* обзор LLM-unlearning: методы (GA, DPO/NPO, RMU, task arithmetic, in-context), данные, метрики, открытые проблемы. *Выводы:* большинство LLM-методов стирают *поведение*, а не *знание*; нужны атаки/зонды, чтобы отличить «не говорит» от «не знает». **Взять:** каркас месяца 3.

**Rethinking Machine Unlearning for Large Language Models** — Liu, Yao, Jia, Casper, Baracaldo, Hase, et al., 2024/2025 (Nature Machine Intelligence) — https://arxiv.org/abs/2402.08787
*О чём:* критический взгляд: unlearning ≠ приватность; таксономия *что* и *зачем* удаляем (вредоносный контент, копирайт, приватность, галлюцинации, bias). *Выводы:* вводит набор «что делать / чего не делать» и показывает, что MIA-метрики для LLM часто неприменимы. **Взять:** обоснование выбора задач месяца 3 (Toxigen / копирайт) и постановки оценки.

**Towards Reliable Forgetting: A Survey on Machine Unlearning Verification** — 2025 — https://arxiv.org/abs/2506.15115
*О чём:* первый обзор именно *верификации*: behavioral (MIA, зонды, backdoor) vs parametric (analysis весов/активаций). *Выводы:* параметрическая верификация (анализ весов и представлений) недопредставлена и «фрагментирована» — прямая ниша для экспериментов проекта. **Взять:** рамка для раздела «Доказательства» и для анализа `layer_alignment_and_transplant.ipynb`.

---

## 2. Фундаментальные / точные методы

**Machine Unlearning (SISA)** — Bourtoule, Chandrasekaran, Choquette-Choo, Jia, Travers, Zhang, Lie, Papernot, 2019/2021 (IEEE S&P) — https://arxiv.org/abs/1912.03817
*О чём:* Sharded-Isolated-Sliced-Aggregated: данные бьются на шарды, на каждом обучается своя модель; при запросе удаления ретрейнится только релевантный шард/слайс. *Выводы:* при запросах, затрагивающих одну точку, overhead падает пропорционально числу шардов; даёт «приближённо точное» забывание. **Взять:** CPU-бейзлайн месяца 1 (MLP/LogReg на Adult, CIFAR-10) — базовая реализация и постановка trade-off «число шардов ↔ точность ↔ время».

**Certified Data Removal from Machine Learning Models** — Guo, Goldstein, Hannun, van der Maaten, 2019/2020 (ICML) — https://arxiv.org/abs/1911.03030
*О чём:* Newton-обновление с шумом даёт формальную гарантию $\epsilon$-certified removal (в духе DP). *Выводы:* для выпуклых задач (LogReg, линейные) существуют методы с доказуемой границей «насколько похоже на retrain». **Взять:** теоретическая опора и эталон «точного» unlearning для линейных моделей.

**Certified Unlearning for Neural Networks** — Koloskova, Allouah, Jha, Guerraoui, Koyejo, 2025 (ICML) — https://arxiv.org/abs/2506.06985
*О чём:* certified-метод для сетей через шумный fine-tuning на retain-данных и связь с «privacy amplification by stochastic post-processing». *Выводы:* формальные гарантии достижимы и вне выпуклой постановки; метод обходит baselines. **Взять:** современный reference для перехода «линейная теория → нейросети».

**Understanding Black-box Predictions via Influence Functions** — Koh & Liang, 2017 (ICML) — https://arxiv.org/abs/1703.04730 · https://www.researchgate.net/publication/315111148
*О чём:* влияние точки на параметры/предсказание через $-\mathcal{H}^{-1}\nabla_\theta L$; HVP считаются без явной обратной матрицы (LiSSA/Conjugate Gradient). *Выводы:* один самый влиятельный пример можно найти и «удалить», сдвинув веса по influence-направлению; приближение работает и для невыпуклых сетей. **Взять:** CPU/GPU-бейзлайн месяца 1–2 («обновление весов CNN через Hessian-vector products»), а также идея направления в весах, созвучная OSU-LoRA.

---

## 3. Приближённое разобучение для глубоких сетей

**Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks (Fisher Forgetting)** — Golatkar, Achille, Soatto, 2019/2020 (CVPR) — https://arxiv.org/abs/1911.04933 · https://www.researchgate.net/publication/343461401
*О чём:* «забывание» через возмущение весов шумом, границы которого задаются Fisher information; optimal невыпуклое решение ищется в окрестности исходного минимума. *Выводы:* выборочное забывание возможно без ретрейнинга при малых forget-set; чувствительность к Fisher-приближению и stability SGD. **Взять:** реализация Fisher Forgetting (Hessian-free) месяца 2, неделя 3; идея «оптимум unlearning близко к исходным весам» важна для анализа окрестности.

**Towards Unbounded Machine Unlearning (SCRUB)** — Kurmanji, Triantafillou, Hayes, Triantafillou, 2023 (NeurIPS) — https://arxiv.org/abs/2302.09880
*О чём:* teacher-student: студент максимизирует забывание по forget-set и минимизирует дивергенцию с retain-моделью; формализует «unbounded unlearning» с разными целями (privacy vs bias). *Выводы:* на забытых данных задача часто не «увеличить ошибку до oracle», а добиться иного целевого поведения; предшествующие методы плохо масштабируются и завязаны на близость к исходным весам. **Взять:** метод-конкурент для месяцев 2–3 и правильная постановка цели забывания (см. критерий выбора класса в `base_models_comparison.ipynb`).

**Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher** — Chundawat, Tarun, Mandal, Kankanhalli, 2022/2023 (AAAI) — https://arxiv.org/html/2205.08096v2 · https://ojs.aaai.org/index.php/AAAI/article/view/25879
*О чём:* два учителя (компетентный и «некомпетентный»), студент избирательно перенимает забывание. *Выводы:* дистилляция от «плохого» учителя даёт забывание без полного ретрейнинга и лучше держит retain-точность. **Взять:** простой и дешёвый approximate-бейзлайн для месяца 2.

**Amnesiac Machine Learning** — Graves, Nagisetty, Ganesh, 2020/2021 (AAAI) — https://arxiv.org/abs/2010.10981 · https://github.com/lmgraves/AmnesiacML
*О чём:* «unlearning by muddling»: ретрейнинг на модифицированном датасете, который портит представление о забываемых данных. *Выводы:* получилось добиться забывания при сохранении качества на остальных данных; ранняя, но полезная базовая линия. **Взять:** ещё один CPU-бейзлайн и код-референс.

**Deep Unlearning: Fast and Efficient Gradient-free Approach to Class Forgetting** — Kodge, Saha, Roy, 2023 — https://arxiv.org/abs/2312.00761
*О чём:* training-free/gradient-free удаление целого класса через декомпозицию активаций и прицельное зануление «направлений класса». *Выводы:* до ~6× дешевле baselines при лучшей retain-точности; исследуется влияние на decision boundary и saliency. **Взять:** прямое попадание в постановку проекта «удалить класс на CIFAR-10» (это ровно `forget_class` в `learning.ipynb`), хороший non-gradient baseline.

**Boundary Unlearning** — Chen, Gao, Wang, et al., 2023 — https://arxiv.org/abs/2303.11570
*О чём:* сдвиг решающей границы вместо изменения параметров: Boundary Shrink/Expanding, имитация поведения retrained-модели. *Выводы:* быстрее, чем параметрические методы, и стабильнее по забыванию класса; смотрит в *decision space*, а не в пространство весов. **Взять:** альтернатива «изменению весов»; полезно сопоставить с CKA/analysis.

**Self-Destructing Models** — Henderson, Mitchell, Manning, Jurafsky, Finn, 2022/2023 — https://arxiv.org/abs/2211.14946
*О чём:* мета-обучение «самоуничтожающейся» модели: при попытке использовать её для вредного таска качество падает. *Выводы:* unlearning можно закладывать на этапе обучения, а не делать post-hoc; полезно для threat-model «unlearning как защита». **Взять:** контекст для production-раздела (когда unlearning нужен *до* инцидента).

**DAMP — Class Unlearning via Depth-Aware Removal of Forget-Specific Directions** — 2026 (CVPRW) — https://arxiv.org/abs/2604.15166
*О чём:* удаление «направлений, специфичных для забываемого класса», с учётом глубины слоя; ближе к retraining-gold-standard, чем прошлые методы (MNIST/CIFAR-10/CIFAR-100/Tiny ImageNet, CNN и ViT). *Выводы:* вклад забывания распределён по глубине неравномерно — **прямая перекличка с наблюдением проекта**, что специфика удаления концентрируется в поздних стадиях (`S1_logits`, `S2_embedding`), а сверточный ствол функционально универсален (CKA 0.94–0.99). **Взять:** свежий ориентир и внешнее подтверждение гипотезы «слой-зависимого» забывания.

**NegMerge: Sign-Consensual Weight Merging for Machine Unlearning** — Kim et al., 2024/2025 (ICML) — https://arxiv.org/abs/2410.05583
*О чём:* строит task-вектор забывания из моделей с разными гиперпараметрами, оставляет согласованные по знаку координаты и *вычитает* его из исходной модели. *Выводы:* проще и сильнее статических task-векторов на изображениях и zero-shot. **Взять:** связка с месяцами 2–3 (unlearning через арифметику весов) и сравнение с OSU-LoRA.

---

## 4. LLM Unlearning

**Knowledge Unlearning for Mitigating Privacy Risks in Language Models** — Jang, Yoon, Yang, et al., 2022/2023 (ACL) — https://arxiv.org/abs/2210.01504
*О чём:* один из первых LLM-unlearning: градиентный подъём по забываемому корпусу + сохраняющий лосс. *Выводы:* GA стирает дословную регенерацию, но бьёт по общей связности; задаёт базовую постановку «forget corpus vs retain». **Взять:** базовый GA-бейзлайн для LLM и источник мотивации стабилизации через KL.

**Who's Harry Potter? Approximate Unlearning in LLMs** — Eldan & Russinovich, 2023 — https://arxiv.org/abs/2310.02238
*О чём:* «unlearning by direct preference / дообучение на перефразированном контенте»: сравнение logits базовой и «отягощённой» модели, усиление забывания. *Выводы:* удаётся убрать конкретную сущность (Harry Potter), сохранив общую способность; но эффект неполный. **Взять:** канонический кейс-стади месяца 3 (удаление знаний/копирайта) и предупреждение про «неполное забывание».

**The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning** — Li, Pan, Gopal, et al., 2024 (ICML) — https://arxiv.org/abs/2403.03218
*О чём:* бенчмарк опасных знаний (bio/cyber/chem) + метод **RMU** (Representation Misdirection for Unlearning): направленное искажение скрытых представлений на forget-данных. *Выводы:* RMU снижает вредное знание, сохраняя MMLU; представления — эффективная цель для LLM-unlearning. **Взять:** реализация RMU (месяц 3, неделя 2) и готовый внешний бенчмарк-щит.

**TOFU: A Task of Fictitious Unlearning for LLMs** — Maini, Feng, Schwarzschild, Lipton, Kolter, 2024 (COLM) — https://arxiv.org/abs/2401.06121
*О чём:* синтетические «биографии авторов»: forget/retain сплиты, где know/не-know строго контролируются. *Выводы:* вводит калиброванный набор метрик (model utility, forget quality) и показывает, что методы «не забывают по-настоящему», а лишь хуже отвечают. **Взять:** контролируемая LLM-песочница для месяца 3 (дешевле, чем возня с реальными копирайт-датасетами).

**Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning (NPO)** — Zhang, Lin, Bai, Zhou, et al., 2024 — https://arxiv.org/abs/2404.05868
*О чём:* использование DPO-объектива только с «негативными» предпочтениями как замена голому GA. *Выводы:* NPO борется с катастрофическим коллапсом GA и даёт более контролируемое забывание. **Взять:** замена/усиление «GA + KL» для LLM (месяц 3); теоретически ближе к DPO-unlearning из плана.

**Editing Models with Task Arithmetic** — Ilharco, Ribeiro, Wortsman, Gururangan, Schmidt, Hajishirzi, Farhadi, 2022/2023 (ICLR) — https://arxiv.org/abs/2212.04089
*О чём:* task vectors (разность весов до/после fine-tuning) складываются/инвертируются; отрицание task-вектора убирает навык. *Выводы:* арифметика в весах управляет поведением модели без ретрейнинга. **Взять:** основа «task arithmetic / T-merging»-ветки месяца 3 и родственная идея для OSU-LoRA (направления в весах).

**Safe LoRA: the Silver Lining of Reducing Safety Risks when Fine-tuning LLMs** — Hsu, Chien, et al., 2024 (NeurIPS) — https://arxiv.org/abs/2405.16833
*О чём:* проекция LoRA-дельта-весов на подпространство safety-aligned модели перед применением. *Выводы:* ортогональная проекция LoRA-обновлений возвращает безопасность, почти не теряя полезность. **Взять:** **самая близкая по механике работа к OSU-LoRA** — проекция градиента/обновления в подпространство. Обязательна к разбору при разработке собственного метода.

**LoRA Learns Less and Forgets Less** — Biderman, Portes, Ortiz, et al., 2024 (TMLR) — https://arxiv.org/abs/2405.09673
*О чём:* эмпирика: LoRA vs full fine-tuning по забыванию/сохранению. *Выводы:* LoRA забывает меньше (в т.ч. менее подвержена катастрофическому забыванию), но и учит меньше. **Взять:** аргумент в пользу LoRA-based unlearning (OSU-LoRA) и baseline для retain-метрик.

**Do Unlearning Methods Remove Information from Language Model Weights?** — Thaker, Hu, Kale, et al., 2024/2025 — https://arxiv.org/abs/2410.08827
*О чём:* атака-зонд, восстанавливающий «забытое» знание из весов после разных методов. *Выводы:* существующие методы часто **маскируют**, а не удаляют информацию — знание частично извлекаемо. **Взять:** обязательный sanity-check для месяца 3 и риторика «поведение ≠ знание»; согласуется с параметрической верификацией из раздела 5.

---

## 5. Оценка: MIA, бенчмарки, верификация

**Membership Inference Attacks Against Machine Learning Models** — Shokri, Stronati, Song, Shmatikov, 2017 (IEEE S&P) — https://arxiv.org/abs/1610.05820 · https://www.researchgate.net/publication/317002535
*О чём:* shadow-модели обучают атакующий классификатор «в модели / не в модели» по выходным распределениям. *Выводы:* переобучение → утечка членства; MIA — базовый инструмент проверки «забыл или нет». **Взять:** метрика Efficacy в плане проекта (MIA Success Rate).

**Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting** — Yeom, Giacomelli, Fredrikson, Jha, 2018 (CSF) — https://arxiv.org/abs/1709.01604
*О чём:* теоретическая связь MIA-риска и переобучения. *Выводы:* членство выводимо, когда loss на объекте ниже порога; даёт простую loss-threshold атаку. **Взять:** дешёвый CPU-инструмент оценки unlearning для месяцев 1–2.

**Membership Inference Attacks From First Principles (LiRA)** — Carlini, Chien, Nasr, Song, Terzis, Tramèr, 2022 (IEEE S&P) — https://arxiv.org/abs/2112.03570
*О чём:* likelihood-ratio атака с калибровкой по «out-of-distribution» референсам. *Выводы:* сильнее пороговых атак; корректная оценка MIA требует референсных распределений, иначе метрики завышены. **Взять:** эталон алгоритма MIA при жёсткой оценке (месяц 4, неделя 1).

**MUSE: Machine Unlearning Six-Way Evaluation for Language Models** — Shi, Lee, Huang, et al., 2024/2025 — https://arxiv.org/abs/2407.06460 · https://muse-bench.github.io/
*О чём:* шесть требований: verbatim/ knowledge memorization, privacy leakage, utility, model robustness, забывание и сохранение. *Выводы:* на 7B-моделях (Harry Potter, новости) большинство методов убирают дословное воспроизведение, но плохо держат utility/robustness. **Взять:** готовый многогранный LLM-бенчмарк для месяца 4.

**RWKU: Benchmarking Real-World Knowledge Unlearning for LLMs** — Liu, Dou, Tan, Tian, Jiang, et al., 2024 — https://arxiv.org/abs/2406.10890
*О чём:* удаление знаний о реальных людях (200 сущностей), атаки при перефразировании, inversion/MIA, out-of-distribution. *Выводы:* «практическое» забывание существенно сложнее синтетического TOFU; нужны adversarial-проверки. **Взять:** дополнительный benchmark месяца 4.

**OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics** — 2025 (NeurIPS D&B) — https://arxiv.org/abs/2506.12618 · https://github.com/locuslab/open-unlearning
*О чём:* единый фреймворк: реализация методов (RMU, NPO/DPO, GA, …) × метрик (MIA, TOFU, WMDP) на общем интерфейсе. *Выводы:* результаты методов сильно зависят от выбора метрик; унификация снижает разброс воспроизводимости. **Взять:** готовый harness для честного сравнения OSU-LoRA с baselines — не писать оценку с нуля.

**TruthfulQA: Measuring How Models Mimic Human Falsehoods** — Lin, Hilton, Evans, 2021/2022 (ACL) — https://arxiv.org/abs/2109.07958
*О чём:* 817 вопросов, провоцирующих распространённые заблуждения. *Выводы:* truthful-метрика чувствительна к правдоподобным, но ложным ответам. **Взять:** метрика «галлюцинаций/правдивости» из плана месяца 4.

---

## 6. Геометрия весов и представлений моделей

Эта группа напрямую обслуживает `layer_alignment_and_transplant.ipynb` и обоснование OSU-LoRA.

**Similarity of Neural Network Representations Revisited (CKA)** — Kornblith, Norouzi, Lee, Hinton, 2019 (ICML) — https://arxiv.org/abs/1905.00414
*О чём:* CKA (через HSIC) — инвариантная к ортогональным преобразованиям мера сходства представлений. *Выводы:* корреляция нейронов по индексу неверна, т.к. базисы каналов произвольны; CKA корректно сравнивает «функцию», а не «координаты». **Взять:** уже использована в анализе (`linear_cka`) — эта работа обосновывает, почему S3 «разошёлся по весам, но совпал по CKA 0.94». Целевое чтение для интерпретации результатов.

**Linear Mode Connectivity and the Lottery Ticket Hypothesis** — Frankle, Dziugaite, Roy, Carbin, 2020 (ICML) — https://arxiv.org/abs/1912.05671
*О чём:* когда две сети линейно связаны (интерполяция весов не ухудшает loss) — зависит от согласованности базисов. *Выводы:* независимые SGD-решения могут лежать в разных «долинах»; выравнивание ведёт к connectivity. **Взять:** теоретическая рамка для наблюдения «base — центроид, base↔oracle ближе, чем oracle↔oracle».

**Git Re-Basin: Merging Models modulo Permutation Symmetries** — Ainsworth, Hayase, Srinivasa, 2022/2023 (ICLR) — https://arxiv.org/abs/2209.04836 · https://www.researchgate.net/publication/363501961
*О чём:* поиск перестановок нейронов, после которых модели можно усреднять/сливать без потерь. *Выводы:* большая часть различий весов — перестановочная симметрия; выравнивание снимает барьеры. **Взять:** мотивация экспериментов с перестановкой каналов и «аналитическим выравниванием» (Procrustes + жадное назначение) в layer-transplant-ноутбуке.

**ZipIt! Merging Models from Different Tasks without Training** — Stoica, Bolya, Bjorner, Ramesh, Hearn, Hoffman, 2023 (NeurIPS) — https://arxiv.org/abs/2305.03053
*О чём:* слияние моделей с разными инициализациями через сопоставление и «объединение» каналов. *Выводы:* даже разные таски имеют общие признаки, которые можно не обучая слить. **Взять:** техника сопоставления каналов; согласуется с «сверточный ствол универсален».

**Model Fusion via Optimal Transport (OT Fusion)** — Singh & Jaggi, 2019/2020 (NeurIPS) — https://arxiv.org/abs/1910.05653
*О чём:* выравнивание нейронов между моделями через оптимальный транспорт. *Выводы:* OT-выравнивание даёт слияние лучше наивного усреднения. **Взять:** альтернатива Procrustes для переноса/выравнивания стадий.

**Model soups: averaging weights of multiple fine-tuned models** — Wortsman, Ilharco, Gadre, et al., 2022 (ICML) — https://arxiv.org/abs/2203.05482
*О чём:* усреднение весов нескольких fine-tuned чекпоинтов повышает точность без доп. инференс-стоимости. *Выводы:* усреднение в весах работает, если модели близки. **Взять:** контекст для «база как центроид» и merge-based unlearning.

**Relative representations enable zero-shot latent space communication** — Moschella, Maiorca, Fumero, et al., 2022/2023 (ICLR) — https://arxiv.org/abs/2209.15430
*О чём:* представления относительно «якорей» инвариантны к изометриям/масштабам латентного пространства. *Выводы:* разные модели кодируют одно и то же по-разному, но с точностью до вращения — что видно как «ортогональные» веса при высоком CKA. **Взять:** объяснение случая S2 (`rel_dist ≈ 1.28`, но взаимные расстояния сохранены) и идея «клея» между стволом и головой.

---

## 7. Continual learning и ортогональные подпространства

Фундамент для **OSU-LoRA** (проекция градиентов забывания в подпространство, ортогональное градиентам сохранения).

**Overcoming catastrophic forgetting in neural networks (EWC)** — Kirkpatrick, Pascanu, Rabinowitz, et al., 2016/2017 (PNAS) — https://arxiv.org/abs/1612.00796
*О чём:* штраф по Fisher information удерживает важные для прошлых задач веса. *Выводы:* защита «важных» направлений позволяет учить новое без разрушения старого. **Взять:** концептуальный предок «сохраняющего» ограничения в unlearning и связь с Fisher Forgetting.

**Gradient Projection Memory for Continual Learning (GPM)** — Saha, Garg, Roy, 2021 (ICLR) — https://arxiv.org/abs/2103.09762
*О чём:* хранит базис подпространства, важного для прошлых задач, и проецирует туда градиенты новой задачи. *Выводы:* явное ортогональное подпространство резко снижает forgetting. **Взять:** прямой источник механики OSU-LoRA — как строить и поддерживать ортогональное подпространство.

**Orthogonal Subspace Learning for Language Model Continual Learning (O-LoRA)** — Wang, Zhang, Chen, Gui, Huang, et al., 2023 — https://arxiv.org/abs/2310.14152
*О чём:* каждый новый LoRA-адаптер учится в подпространстве, ортогональном предыдущим. *Выводы:* ортогональность адаптеров снижает катастрофическое забывание почти без потери качества. **Взять:** **непосредственный прототип OSU-LoRA** (ортогональные LoRA-подпространства) — переиспользовать идею, но развернуть её на «forget vs retain».

---

## 8. Редактирование знаний

**Locating and Editing Factual Associations in GPT (ROME)** — Meng, Bau, Andonian, Belinkov, 2022 (NeurIPS) — https://arxiv.org/abs/2202.05262
*О чём:* знание локализуется в MLP-слоях (mid-layer), редактируется rank-one обновлением. *Выводы:* факты хранятся локально и редактируемы без переобучения. **Взять:** контекст для «параметрической» работы с LLM-знанием и как родственный, но более грубый baseline (говорят «правку», а не «забывание»).

---

## 9. Как это ложится на фазы проекта

| Фаза / ноутбук | Опорная литература |
| :--- | :--- |
| Литературный обзор (н.1) | [1] все обзоры; [5] MIA-статьи |
| SISA на CPU (н.2) | SISA (1912.03817) |
| Influence Functions (н.3) | Koh & Liang (1703.04730); Certified Removal (1911.03030) |
| SISA vs IF + MIA (н.4) | Shokri (1610.05820), Yeom (1709.01604), LiRA (2112.03570) |
| GPU/ResNet (н.1) | — (инфра), плюс EWC (1612.00796) как контекст |
| GA + GD + KL (н.2) | Jang (2210.01504), NPO (2404.05868), SCRUB (2302.09880) |
| Fisher Forgetting (н.3) | Golatkar (1911.04933) |
| Trade-off / Pareto (н.4) | SCRUB (2302.09880), Deep Unlearning (2312.00761), Boundary Unlearning (2303.11570) |
| LLM: DPO/Task Arithmetic (н.1) | Task Arithmetic (2212.04089), NegMerge (2410.05583) |
| LLM: RMU / Toxigen (н.2) | WMDP+RMU (2403.03218), MUSE (2407.06460), Harry Potter (2310.02238) |
| OSU-LoRA (н.3) | GPM (2103.09762), O-LoRA (2310.14152), **SafeLoRA (2405.16833)**, LoRA Learns Less (2405.09673) |
| Сравнение с baselines (н.4) | OpenUnlearning (2506.12618), TOFU (2401.06121) |
| Жёсткая оценка (мес.4 н.1) | LiRA (2112.03570), TruthfulQA (2109.07958), Do Unlearning… (2410.08827) |
| Latent space / t-SNE + анализ слоёв (`layer_alignment_and_transplant.ipynb`) | CKA (1905.00414), LMC (1912.05671), Git Re-Basin (2209.04836), ZipIt (2305.03053), OT Fusion (1910.05653), Relative Representations (2209.15430), DAMP (2604.15166) |

### Ключевые внешние подтверждения гипотез проекта
- **Забывание сконцентрировано в поздних слоях / голове.** Согласуется с DAMP (depth-aware, 2604.15166) и с наблюдением `layer_alignment_and_transplant.ipynb`: сверточный ствол CKA 0.94–0.99, разница — в `S2_embedding`/`S1_logits`.
- **Различия в весах часто = смена базиса/перестановка.** Объясняется CKA (1905.00414), Git Re-Basin (2209.04836), Relative Representations (2209.15430) — прямо поддерживает эксперименты с Procrustes и перестановкой каналов.
- **Ортогональные подпространства как инструмент unlearning.** O-LoRA (2310.14152) и SafeLoRA (2405.16833) делают ровно то, что заложено в OSU-LoRA (проекция/ортогональность), но для смежных целей — это независимое подтверждение выбранного направления.
- **«Поведение ≠ знание».** Thaker et al. (2410.08827) и MUSE (2407.06460): методики надо оценивать атаками/зондами, а не только падением метрик на forget-промптах.
