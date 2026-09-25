# 🧠 AI Unlearning: Практико-ориентированный исследовательский проект

Добро пожаловать в репозиторий, посвященный исследованию и реализации методов **Machine Unlearning** (разучивания моделей). 
Проект нацелен на изучение, разработку и оценку методов удаления влияния конкретных данных из обученных моделей машинного обучения 
(включая Deep Learning и LLM) с сохранением общей полезности (utility) модели.

Проект сочетает в себе академический подход с требованиями production-ready архитектуры.

---

## 🎯 Цели проекта
1. Изучить современное состояние (SOTA) в области Exact и Approximate Unlearning.
2. Реализовать и сравнить базовые методы на CPU, а затем масштабировать их на GPU.
3. Исследовать специфические методы для Large Language Models (LLM).
4. Разработать и оценить собственный метод (на базе динамического LoRA и ортогональных подпространств).
5. Оценить экономическую и архитектурную целесообразность Unlearning в продакшене.

---

## 📅 План проекта (4 месяца)

### Месяц 1: Фундамент, SOTA-анализ и CPU-бейзлайны
* **Неделя 1:** Литературный обзор. Выбор метрик: Unlearning efficacy (Membership Inference Attacks - MIA), Utility (Accuracy/F1), Efficiency (время/память).
* **Неделя 2:** Реализация **SISA** (Sharded, Isolated, Sliced, and Aggregated) на CPU. Эксперименты: MLP и Logistic Regression на Adult Income, CIFAR-10.
* **Неделя 3:** Реализация **Influence Functions** (Koh & Liang). Обновление весов небольшой CNN через Hessian-vector products.
* **Неделя 4:** Сравнительный анализ SISA и Influence Functions. Оценка устойчивости к MIA-атакам.

### Месяц 2: Переход на GPU, Deep Learning и Gradient-based методы
* **Неделя 1:** Масштабирование пайплайна на PyTorch + CUDA. Обучение ResNet-18 и ViT-Tiny на CIFAR-100 / Tiny ImageNet. Настройка W&B / MLflow.
* **Неделя 2:** Реализация **Gradient Ascent (GA)** и его стабилизация через **GA + Gradient Descent (GD)** с KL-дивергенцией для предотвращения катастрофического забывания.
* **Неделя 3:** Изучение и реализация **Fisher Forgetting** (Hessian-free методы) для селективного обнуления весов.
* **Неделя 4:** Промежуточный анализ Trade-off "Unlearning vs Utility". Построение Pareto-фронтов.

### Месяц 3: LLM Unlearning и разработка собственного метода
* **Неделя 1:** Изучение специфики LLM Unlearning: DPO for Unlearning, Task Arithmetic / T-Merging.
* **Неделя 2:** Эксперименты с LLM (Llama-3-8B / Pythia-6.9B). Задача: удаление конкретных знаний (например, Toxigen dataset или защищенный авторским правом контент). Применение **RMU** (Representation Misdirection for Unlearning).
* **Неделя 3:** **Research Sprint:** Разработка собственного метода **OSU-LoRA** (Orthogonal Subspace Unlearning via Dynamic LoRA). Проецирование градиентов «забывания» в ортогональное подпространство относительно градиентов общей языковой связности.
* **Неделя 4:** Обучение, отладка и сравнение OSU-LoRA с базовыми методами (RMU, DPO-unlearning).

### Месяц 4: Прод-оценка, Финализация и Релиз
* **Неделя 1:** Жесткая оценка (Rigorous Evaluation): TruthfulQA (галлюцинации), Attack Success Rate (ASR), Retain Set Accuracy.
* **Неделя 2:** Архитектурный и экономический анализ (TCO). Сравнение стоимости полного ретрейнинга vs Unlearning-пайплайна. Проектирование API для Unlearning-as-a-Service.
* **Неделя 3:** Визуализация latent space (t-SNE/UMAP) до и после unlearning. Написание итогового отчета.
* **Неделя 4:** Open-source релиз: публикация кода, весов и пайплайнов.

---

## 📊 Метрики оценки
| Категория | Метрики |
| :--- | :--- |
| **Efficacy** | Membership Inference Attack (MIA) Success Rate, ROUGE/BLEU drop на забытых данных |
| **Utility** | Accuracy, F1-Score, TruthfulQA score на Retain Set (общие данные) |
| **Efficiency** | Время выполнения (часы), пиковое потребление VRAM, стоимость вычислений ($) |

---

## 🏗 Production Perspective (Важно)
В реальных продакшен-системах Machine Unlearning часто является **антипаттерном**. Если необходимо удалить <5% данных, полный ретрейнинг модели на дешевых spot-инстансах часто оказывается **дешевле и надежнее**, чем внедрение сложного approximate unlearning, который может деградировать бизнес-метрики.

**Unlearning оправдан в случаях:**
1. Ретрейнинг физически невозможен или чрезмерно дорог (модели, обучавшиеся месяцами на экзавбайтах данных).
2. Edge-устройства и Federated Learning, где исходные данные пользователя недоступны, но требуется соблюдение GDPR/CCPA.

---

## 📚 Актуальная литература (State-of-the-Art)

### Фундаментальные работы
1. Bourtoule, L., et al. (2021). *Machine Unlearning (SISA)*. IEEE S&P.
2. Koh, P. W., & Liang, P. (2017). *Understanding Black-box Predictions via Influence Functions*. ICML.
3. Guo, C., et al. (2020). *Certified Data Removal from Machine Learning Models*. ICML.

### Approximate Unlearning & Deep Learning
4. Golatkar, A., et al. (2020). *Eternal Sunshine of the Spotless Net: Fisher Forgetting*.
5. Zhao, J., et al. (2024). *A Survey of Machine Unlearning*.

### LLM Unlearning (Фронтир)
6. Li, C., et al. (2024). *Representation Misdirection for Unlearning (RMU)*.
7. Zhao, Y., et al. (2024). *DPO for LLM Unlearning / Forget-me-not*.
8. Ilharco, G., et al. (2023). *Editing Models with Task Arithmetic*.
9. Xu, Z., et al. (2024). *A Comprehensive Survey on Machine Unlearning for Large Language Models*.

### Методы оценки
10. Thomas, M., et al. (2024). *The Unlearning Benchmark: How to evaluate if an LLM actually forgot*.

---
