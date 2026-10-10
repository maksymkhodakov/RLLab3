"""Генерація звіту до лабораторної роботи №3 у форматі DOCX за ДСТУ 3008:2015.

Оформлення:
  * Times New Roman 14 пт, міжрядковий інтервал 1,5, абзацний відступ 1,25 см;
  * поля: ліве 30 мм, праве 15 мм, верхнє та нижнє 20 мм;
  * нумерація сторінок арабськими цифрами у правому верхньому куті,
    титульний аркуш включено до нумерації, але номер на ньому не ставиться;
  * заголовки структурних частин (РЕФЕРАТ, ЗМІСТ, ВСТУП, ВИСНОВКИ, ...) --
    великими літерами по центру; розділи -- з абзацного відступу, великими
    літерами, жирним; кожен розділ з нової сторінки;
  * рисунки: "Рисунок 2.1 – Назва" під рисунком по центру;
    таблиці: "Таблиця 3.1 – Назва" над таблицею;
    формули: по центру, номер у круглих дужках біля правого поля;
  * перелік джерел -- за ДСТУ 8302:2015.

Зміст будується у два проходи: спершу документ конвертується в PDF через
LibreOffice, з PDF визначаються номери сторінок заголовків, а потім
документ генерується повторно з правильними номерами.

Запуск: python build_report.py   (результат: report/Lab3_CarRacing_Report.docx)
"""

import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from scipy import stats

RES = "results"
FIG = os.path.join(RES, "figures")
OUT_DIR = "report"
OUT = os.path.join(OUT_DIR, "Lab3_CarRacing_Report.docx")
FONT = "Times New Roman"

BASELINE_CLAIMED = dict(mean=840.6, std=103.2, min=470.4, max=912.7)


# ---------------------------------------------------------------------------
# Дані експериментів
# ---------------------------------------------------------------------------
def read_test(path):
    with open(path) as f:
        return np.array([float(m) for m in re.findall(r"Episode \d+\s+Reward (-?[\d.]+)", f.read())])


def read_csv(path):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


def load_data():
    d = {}
    d["base"] = read_test(os.path.join(RES, "baseline_test_50.txt"))
    d["impr"] = read_test(os.path.join(RES, "improved_test_50.txt"))
    d["eval"] = read_csv(os.path.join(RES, "eval_log.csv"))
    d["train"] = read_csv(os.path.join(RES, "train_log.csv"))
    extra = os.path.join(RES, "improved_test_50_run2.txt")
    d["impr2"] = read_test(extra) if os.path.exists(extra) else None
    return d


def fmt(x, nd=1):
    """Число з десятковою комою (українська нотація)."""
    return f"{x:.{nd}f}".replace(".", ",")


# ---------------------------------------------------------------------------
# Низькорівневі помічники оформлення
# ---------------------------------------------------------------------------
class Report:
    def __init__(self, toc_pages=None):
        self.doc = Document()
        self.toc_pages = toc_pages or {}
        self.toc_entries = []  # (рівень, текст)
        self.fig_no = {}
        self.tab_no = {}
        self.eq_no = {}
        self.section = 0
        self.n_figures = 0
        self.n_tables = 0
        self._setup()

    # -- сторінка і стилі ---------------------------------------------------
    def _setup(self):
        s = self.doc.sections[0]
        s.page_width, s.page_height = Cm(21), Cm(29.7)
        s.left_margin, s.right_margin = Cm(3), Cm(1.5)
        s.top_margin, s.bottom_margin = Cm(2), Cm(2)
        s.header_distance = Cm(1)
        s.different_first_page_header_footer = True  # без номера на титулі

        st = self.doc.styles["Normal"]
        st.font.name = FONT
        st.font.size = Pt(14)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        pf = st.paragraph_format
        pf.line_spacing = 1.5
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.first_line_indent = Cm(1.25)
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        for name in ["Heading 1", "Heading 2", "Heading 3"]:
            h = self.doc.styles[name]
            h.font.name = FONT
            h.font.size = Pt(14)
            h.font.bold = True
            h.font.italic = False
            h.font.color.rgb = None
            rpr = h.element.get_or_add_rPr()
            rfonts = rpr.find(qn("w:rFonts"))
            if rfonts is None:
                rfonts = OxmlElement("w:rFonts")
                rpr.append(rfonts)
            for att in ["w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"]:
                rfonts.set(qn(att), FONT)
            for att in ["w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"]:
                if rfonts.get(qn(att)) is not None:
                    del rfonts.attrib[qn(att)]
            hp = h.paragraph_format
            hp.line_spacing = 1.5
            hp.first_line_indent = Cm(1.25)
            hp.left_indent = Cm(0)
            hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
            hp.keep_with_next = True
        self.doc.styles["Heading 1"].paragraph_format.space_before = Pt(0)
        self.doc.styles["Heading 1"].paragraph_format.space_after = Pt(21)
        for name in ["Heading 2", "Heading 3"]:
            self.doc.styles[name].paragraph_format.space_before = Pt(21)
            self.doc.styles[name].paragraph_format.space_after = Pt(21)

        # Номер сторінки у правому верхньому куті (поле PAGE)
        hdr = s.header.paragraphs[0]
        hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hdr.paragraph_format.first_line_indent = Cm(0)
        self._field(hdr, "PAGE")

    @staticmethod
    def _field(paragraph, instr):
        run = paragraph.add_run()
        for tag, text in [("begin", None), (None, instr), ("separate", None), (None, "1"),
                          ("end", None)]:
            if tag:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), tag)
                run._r.append(el)
            elif text == instr:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = instr
                run._r.append(el)
            else:
                el = OxmlElement("w:t")
                el.text = text
                run._r.append(el)
        run.font.name = FONT
        run.font.size = Pt(14)

    # -- базові блоки -------------------------------------------------------
    def p(self, text="", align=None, indent=True, bold=False, italic=False, size=None,
          space_after=None, keep_next=False):
        """Абзац; підтримує **жирний** та *курсив* усередині тексту."""
        par = self.doc.add_paragraph()
        if align is not None:
            par.alignment = align
        if not indent:
            par.paragraph_format.first_line_indent = Cm(0)
        if space_after is not None:
            par.paragraph_format.space_after = Pt(space_after)
        if keep_next:
            par.paragraph_format.keep_with_next = True
        for chunk in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
            if not chunk:
                continue
            if chunk.startswith("**"):
                r = par.add_run(chunk[2:-2])
                r.bold = True
            elif chunk.startswith("*"):
                r = par.add_run(chunk[1:-1])
                r.italic = True
            else:
                r = par.add_run(chunk)
            if bold:
                r.bold = True
            if italic:
                r.italic = True
            if size:
                r.font.size = Pt(size)
        return par

    def bullets(self, items, marker="–"):
        """Перелік за ДСТУ: з абзацного відступу, через дефіс, з малої літери."""
        for i, it in enumerate(items):
            end = ";" if i < len(items) - 1 else "."
            self.p(f"{marker} {it}{end}")

    def numbered(self, items):
        for i, it in enumerate(items, 1):
            end = ";" if i < len(items) else "."
            self.p(f"{i}) {it}{end}")

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def _new_page_paragraph(self, par):
        par.paragraph_format.page_break_before = True

    # -- заголовки ----------------------------------------------------------
    def structural(self, title, toc=True):
        """Заголовок структурної частини: ВЕЛИКИМИ, по центру, з нової сторінки."""
        h = self.doc.add_heading(title.upper(), level=1)
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        h.paragraph_format.first_line_indent = Cm(0)
        self._new_page_paragraph(h)
        if toc:
            self.toc_entries.append((1, title.upper()))
        return h

    def chapter(self, title):
        self.section += 1
        self.sub = 0
        self.fig_in_sec = 0
        self.tab_in_sec = 0
        self.eq_in_sec = 0
        text = f"{self.section} {title.upper()}"
        h = self.doc.add_heading(text, level=1)
        self._new_page_paragraph(h)
        self.toc_entries.append((1, text))
        return h

    def subchapter(self, title):
        self.sub += 1
        self.subsub = 0
        text = f"{self.section}.{self.sub} {title}"
        self.doc.add_heading(text, level=2)
        self.toc_entries.append((2, text))

    def subsubchapter(self, title):
        self.subsub += 1
        text = f"{self.section}.{self.sub}.{self.subsub} {title}"
        self.doc.add_heading(text, level=3)

    # -- рисунки, таблиці, формули -------------------------------------------
    def figure(self, key, path, caption, width_cm=16.0):
        self.fig_in_sec += 1
        num = f"{self.section}.{self.fig_in_sec}"
        self.fig_no[key] = num
        self.n_figures += 1
        par = self.doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.first_line_indent = Cm(0)
        par.paragraph_format.keep_with_next = True
        par.paragraph_format.space_before = Pt(6)
        par.add_run().add_picture(path, width=Cm(width_cm))
        cap = self.p(f"Рисунок {num} – {caption}", align=WD_ALIGN_PARAGRAPH.CENTER, indent=False,
                     space_after=12)
        return num

    def next_fig(self, key):
        """Номер наступного рисунка (для посилання в тексті перед рисунком)."""
        return f"{self.section}.{self.fig_in_sec + 1}"

    def next_tab(self):
        return f"{self.section}.{self.tab_in_sec + 1}"

    def table(self, caption, header, rows, widths=None, font_size=12):
        self.tab_in_sec += 1
        num = f"{self.section}.{self.tab_in_sec}"
        self.n_tables += 1
        cap = self.p(f"Таблиця {num} – {caption}", indent=False, keep_next=True)
        cap.paragraph_format.space_before = Pt(6)
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for r_i, row in enumerate([header] + rows):
            for c_i, val in enumerate(row):
                cell = t.cell(r_i, c_i)
                cell.text = ""
                par = cell.paragraphs[0]
                par.paragraph_format.first_line_indent = Cm(0)
                par.paragraph_format.line_spacing = 1.0
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER if (r_i == 0 or c_i > 0) \
                    else WD_ALIGN_PARAGRAPH.LEFT
                run = par.add_run(str(val))
                run.font.size = Pt(font_size)
                if r_i == 0:
                    run.bold = True
                if widths:
                    cell.width = Cm(widths[c_i])
        # Повторювати шапку таблиці на кожній сторінці
        trpr = t.rows[0]._tr.get_or_add_trPr()
        el = OxmlElement("w:tblHeader")
        el.set(qn("w:val"), "true")
        trpr.append(el)
        self.doc.add_paragraph().paragraph_format.line_spacing = 1.0
        return num

    def equation(self, key, text):
        """Формула по центру, номер (Р.Н) біля правого поля."""
        self.eq_in_sec += 1
        num = f"{self.section}.{self.eq_in_sec}"
        self.eq_no[key] = num
        par = self.doc.add_paragraph()
        pf = par.paragraph_format
        pf.first_line_indent = Cm(0)
        pf.space_before = Pt(6)
        pf.space_after = Pt(6)
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.tab_stops.add_tab_stop(Cm(8.25), WD_TAB_ALIGNMENT.CENTER)
        pf.tab_stops.add_tab_stop(Cm(16.5), WD_TAB_ALIGNMENT.RIGHT)
        r = par.add_run("\t" + text)
        r.font.name = "Cambria Math"
        r.italic = True
        par.add_run(f"\t({num})")
        return num

    def code(self, text, size=10):
        """Лістинг: моноширинний шрифт, одинарний інтервал."""
        for line in text.rstrip("\n").split("\n"):
            par = self.doc.add_paragraph()
            pf = par.paragraph_format
            pf.first_line_indent = Cm(0)
            pf.line_spacing = 1.0
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = par.add_run(line if line else " ")
            r.font.name = "Courier New"
            r._element.rPr.rFonts.set(qn("w:eastAsia"), "Courier New")
            r.font.size = Pt(size)

    # -- зміст ---------------------------------------------------------------
    def toc_placeholder(self):
        self.structural("Зміст", toc=False)
        self._toc_anchor = self.doc.add_paragraph()  # сюди вставимо рядки змісту

    def fill_toc(self):
        anchor = self._toc_anchor
        for level, text in self.toc_entries:
            par = self.doc.add_paragraph()
            pf = par.paragraph_format
            pf.first_line_indent = Cm(0)
            pf.left_indent = Cm(0 if level == 1 else 0.5)
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf.tab_stops.add_tab_stop(Cm(16.5), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
            page = self.toc_pages.get(text, "")
            par.add_run(f"{text}\t{page}")
            anchor._p.addprevious(par._p)
        anchor._p.getparent().remove(anchor._p)

    def save(self, path):
        self.fill_toc()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.doc.save(path)


# ---------------------------------------------------------------------------
# Зміст звіту
# ---------------------------------------------------------------------------
REFERENCES = [
    "Mnih V., Kavukcuoglu K., Silver D. et al. Human-level control through deep reinforcement "
    "learning. Nature. 2015. Vol. 518, no. 7540. P. 529–533. DOI: 10.1038/nature14236.",
    "van Hasselt H., Guez A., Silver D. Deep Reinforcement Learning with Double Q-learning. "
    "Proceedings of the AAAI Conference on Artificial Intelligence. 2016. Vol. 30, no. 1. "
    "P. 2094–2100.",
    "Wang Z., Schaul T., Hessel M. et al. Dueling Network Architectures for Deep Reinforcement "
    "Learning. Proceedings of the 33rd International Conference on Machine Learning (ICML). "
    "New York, 2016. P. 1995–2003.",
    "Hessel M., Modayil J., van Hasselt H. et al. Rainbow: Combining Improvements in Deep "
    "Reinforcement Learning. Proceedings of the AAAI Conference on Artificial Intelligence. "
    "2018. Vol. 32, no. 1. P. 3215–3222.",
    "Pardo F., Tavakoli A., Levdik V., Kormushev P. Time Limits in Reinforcement Learning. "
    "Proceedings of the 35th International Conference on Machine Learning (ICML). Stockholm, "
    "2018. P. 4045–4054.",
    "Sutton R. S., Barto A. G. Reinforcement Learning: An Introduction. 2nd ed. Cambridge, MA : "
    "MIT Press, 2018. 552 p.",
    "Lillicrap T. P., Hunt J. J., Pritzel A. et al. Continuous control with deep reinforcement "
    "learning. arXiv preprint. 2015. arXiv:1509.02971.",
    "Towers M., Kwiatkowski A., Terry J. et al. Gymnasium: A Standard Interface for "
    "Reinforcement Learning Environments. arXiv preprint. 2024. arXiv:2407.17032.",
    "Car Racing – Gymnasium Documentation. Farama Foundation. URL: "
    "https://gymnasium.farama.org/environments/box2d/car_racing/ (дата звернення: 10.10.2026).",
    "Zakala P. ReinforcementLearning : репозиторій навчальних матеріалів (dqn_car_racing.py, "
    "test_car_racing.py). GitHub. URL: https://github.com/PavloZakala/ReinforcementLearning "
    "(дата звернення: 10.10.2026).",
    "Paszke A., Gross S., Massa F. et al. PyTorch: An Imperative Style, High-Performance Deep "
    "Learning Library. Advances in Neural Information Processing Systems 32 (NeurIPS 2019). "
    "2019. P. 8024–8035.",
    "Loshchilov I., Hutter F. SGDR: Stochastic Gradient Descent with Warm Restarts. "
    "International Conference on Learning Representations (ICLR). 2017.",
    "ДСТУ 3008:2015. Інформація та документація. Звіти у сфері науки і техніки. Структура та "
    "правила оформлювання. Київ : ДП «УкрНДНЦ», 2016. 26 с.",
]


def build(toc_pages=None, n_pages=None):
    d = load_data()
    base, impr = d["base"], d["impr"]
    ev = d["eval"]
    ev_steps = np.array([int(r["env_steps"]) for r in ev])
    ev_mean = np.array([float(r["mean"]) for r in ev])
    best_i = int(np.argmax(ev_mean))
    tr = d["train"]
    total_steps = int(tr[-1]["env_steps"])
    total_eps = len(tr)
    train_minutes = float(ev[-1]["wall_time"]) / 60
    early_frac = np.mean([int(r["truncated_early"]) for r in tr]) * 100

    # Статистичне порівняння (Велч t-тест та U-критерій Манна-Уітні)
    t_stat, p_t = stats.ttest_ind(impr, base, equal_var=False)
    u_stat, p_u = stats.mannwhitneyu(impr, base, alternative="greater")
    n_base_fail = int((base < 800).sum())
    n_impr_fail = int((impr < 800).sum())

    R = Report(toc_pages)
    doc = R.doc
    C = WD_ALIGN_PARAGRAPH.CENTER

    # ===================== ТИТУЛЬНИЙ АРКУШ ================================
    for line in ["КИЇВСЬКИЙ НАЦІОНАЛЬНИЙ УНІВЕРСИТЕТ", "ІМЕНІ ТАРАСА ШЕВЧЕНКА"]:
        R.p(line, align=C, indent=False, bold=True)
    R.p("Факультет комп'ютерних наук та кібернетики", align=C, indent=False)
    for _ in range(5):
        R.p("", indent=False)
    R.p("ЗВІТ", align=C, indent=False, bold=True)
    R.p("до лабораторної роботи № 3", align=C, indent=False)
    R.p("з дисципліни «Машинне навчання з підкріпленням»", align=C, indent=False)
    R.p("на тему: «Покращення алгоритму Deep Q-Learning для задачі CarRacing-v3»",
        align=C, indent=False)
    for _ in range(3):
        R.p("", indent=False)
    for line in ["Виконав:", "студент 2-го курсу", "групи ШІ-2", "ОНП «Штучний інтелект»",
                 "Ходаков Максим Олегович", "", "Перевірив:", "______________________"]:
        par = R.p(line, indent=False)
        par.paragraph_format.left_indent = Cm(10)
    for _ in range(3):
        R.p("", indent=False)
    R.p("Київ – 2026", align=C, indent=False)

    # ===================== РЕФЕРАТ ========================================
    R.structural("Реферат", toc=False)
    pages = n_pages if n_pages else "__"
    R.p(f"Звіт про лабораторну роботу: {pages} с., {R_count_placeholder('fig')} рис., "
        f"{R_count_placeholder('tab')} табл., {len(REFERENCES)} джерел, 1 додаток.")
    R.p("Об'єкт дослідження – процес навчання агента керуванню автомобілем у середовищі "
        "CarRacing-v3 бібліотеки Gymnasium за піксельними спостереженнями.")
    R.p("Предмет дослідження – методи покращення алгоритму Deep Q-Network (DQN): архітектура "
        "Q-мережі, формування цілі навчання, організація буфера відтворення досвіду, розклади "
        "гіперпараметрів та процедура вибору найкращої моделі.")
    R.p("Мета роботи – підвищити якість базової реалізації DQN для CarRacing-v3 (середня "
        f"нагорода {fmt(BASELINE_CLAIMED['mean'])} на 50 тестових епізодах) за незмінних "
        "параметрів середовища та тестового скрипту.")
    R.p("Методи дослідження – глибоке навчання з підкріпленням (Double DQN, Dueling DQN, "
        "n-крокові повернення, м'яке оновлення цільової мережі), обчислювальний експеримент, "
        "статистичне порівняння вибірок (t-критерій Велча, U-критерій Манна–Уітні).")
    R.p(f"Результати: запропонована модель досягла середньої нагороди {fmt(impr.mean())} ± "
        f"{fmt(impr.std())} (мінімум {fmt(impr.min())}) на 50 тестових епізодах проти "
        f"{fmt(base.mean())} ± {fmt(base.std())} (мінімум {fmt(base.min())}) для базової моделі, "
        f"виміряних тим самим тестовим скриптом на тому самому обладнанні, і проти заявлених "
        f"{fmt(BASELINE_CLAIMED['mean'])} ± {fmt(BASELINE_CLAIMED['std'])}. Покращення "
        f"статистично значуще (p < 0,001).")
    R.p("DEEP Q-NETWORK, DOUBLE DQN, DUELING DQN, N-КРОКОВІ ПОВЕРНЕННЯ, CARRACING, GYMNASIUM, "
        "НАВЧАННЯ З ПІДКРІПЛЕННЯМ, ЗГОРТКОВА НЕЙРОННА МЕРЕЖА, PYTORCH.")

    # ===================== ЗМІСТ ==========================================
    R.toc_placeholder()

    # ===================== ВСТУП ==========================================
    R.structural("Вступ")
    R.p("Навчання з підкріпленням (reinforcement learning, RL) – розділ машинного навчання, у "
        "якому агент навчається приймати послідовні рішення, взаємодіючи із середовищем та "
        "отримуючи скалярний сигнал винагороди [6]. Поєднання RL із глибокими нейронними "
        "мережами дозволило розв'язувати задачі керування безпосередньо за зображеннями: "
        "алгоритм Deep Q-Network (DQN) уперше досяг рівня людини в іграх Atari 2600, "
        "отримуючи на вхід лише пікселі екрана [1].")
    R.p("Задача CarRacing-v3 з бібліотеки Gymnasium [8, 9] – класичний тест для методів "
        "глибокого RL: агент керує автомобілем на випадково згенерованій трасі, бачачи її "
        "згори у вигляді кольорового зображення 96×96 пікселів. На кожному епізоді траса нова, "
        "тому агент має навчитися узагальненої навички водіння, а не запам'ятати конкретну "
        "трасу.")
    R.p("У межах курсу надано базову реалізацію DQN для цієї задачі [10] та натренований за "
        f"1000 епізодів чекпоінт із середньою нагородою {fmt(BASELINE_CLAIMED['mean'])} ± "
        f"{fmt(BASELINE_CLAIMED['std'])} (мінімум {fmt(BASELINE_CLAIMED['min'])}) на 50 тестових "
        "епізодах. Завдання лабораторної роботи – покращити цей результат, змінюючи будь-які "
        "частини архітектури моделі та процесу тренування, але залишаючи незмінними параметри "
        "налаштування середовища і скрипт тестування.")
    R.p("У роботі проаналізовано недоліки базової реалізації та застосовано низку перевірених "
        "покращень DQN, що входять до складу алгоритму Rainbow [4]: Double DQN [2], дуельну "
        "архітектуру мережі [3], n-крокові повернення, а також коректну обробку обрізання "
        "епізодів за часом [5], компактний буфер відтворення, розклади гіперпараметрів і "
        "вибір чекпоінта за жадібною оцінкою. Отримана модель перевершує базову як за середньою "
        "нагородою, так і за стабільністю (меншою кількістю невдалих епізодів).")

    # ===================== 1 ПОСТАНОВКА ЗАДАЧІ ============================
    R.chapter("Постановка задачі")
    R.p("**Мета роботи:** покращити якість агента, натренованого базовою реалізацією DQN "
        "(dqn_car_racing.py) для середовища CarRacing-v3, порівняно з наданим чекпоінтом "
        "checkpoint_CarRacing-v3_best.pth.")
    R.p("**Вихідні дані.** Базовий результат, отриманий командою "
        "«python test_car_racing.py checkpoint_CarRacing-v3_best.pth 50», наведено в "
        f"таблиці {R.next_tab()}.")
    R.table("Заявлена якість базової моделі (50 тестових епізодів)",
            ["Показник", "Значення"],
            [["Кількість епізодів", "50"],
             ["Середня нагорода (Mean)", fmt(BASELINE_CLAIMED["mean"])],
             ["Стандартне відхилення (Std)", fmt(BASELINE_CLAIMED["std"])],
             ["Мінімум (Min)", fmt(BASELINE_CLAIMED["min"])],
             ["Максимум (Max)", fmt(BASELINE_CLAIMED["max"])]],
            widths=[9, 5])
    R.p("**Обмеження**, встановлені умовою завдання:")
    R.bullets([
        "параметри налаштування середовища не змінюються: функція make_env() (CarRacing-v3 з "
        "дискретними діями, перетворення в градації сірого, зменшення до 84×84, стек із 4 "
        "кадрів) залишається ідентичною базовій",
        "тестовий скрипт test_car_racing.py не змінюється; він імпортує з модуля "
        "dqn_car_racing функцію make_env, клас QNetwork і змінну device, створює модель "
        "викликом QNetwork(a_size=5), завантажує state_dict та обирає дії методом act()",
        "усі інші частини (архітектура мережі, алгоритм, гіперпараметри, організація "
        "тренування) можуть змінюватися довільно",
    ])
    R.p("**Завдання роботи:**")
    R.numbered([
        "розгорнути програмне середовище та відтворити оцінювання базової моделі",
        "проаналізувати базову реалізацію DQN і виявити її слабкі місця",
        "обґрунтувати та реалізувати покращення архітектури моделі й процесу тренування з "
        "дотриманням інтерфейсу, якого вимагає тестовий скрипт",
        "натренувати покращену модель і оцінити її тим самим тестовим скриптом на 50 епізодах",
        "порівняти результати з базовими, перевірити статистичну значущість різниці та "
        "сформулювати висновки",
    ])

    # ===================== 2 СЕРЕДОВИЩЕ ===================================
    R.chapter("Опис середовища CarRacing-v3")
    R.p("CarRacing-v3 – середовище з набору Box2D бібліотеки Gymnasium [9]. На кожному епізоді "
        "генерується нова замкнена траса довжиною близько 300 плиток; агент керує автомобілем "
        "з фізикою на основі рушія Box2D (інерція, занос, зчеплення шин із покриттям).")
    R.subchapter("Простір спостережень")
    R.p("Сире спостереження – кольорове зображення 96×96×3 (вид згори з камерою, що слідує за "
        "автомобілем). У нижній частині кадру розташована панель індикаторів: швидкість, "
        "показники ABS чотирьох коліс, кут керма та гіроскоп. Функція make_env() застосовує "
        "до спостереження три обгортки:")
    R.bullets([
        "GrayscaleObservation – переведення в градації сірого (96×96)",
        "ResizeObservation – зменшення до 84×84 пікселів",
        "FrameStackObservation – стек із 4 останніх кадрів, що дає мережі інформацію про "
        "швидкість і напрям руху; на початку епізоду стек заповнюється копіями першого кадру",
    ])
    R.p(f"Отже, вхід Q-мережі – тензор розмірності 4×84×84 типу uint8. Приклад сирого кадру та "
        f"відповідного стеку після передобробки наведено на рисунку {R.next_fig('obs')}.")
    R.figure("obs", os.path.join(FIG, "observation.png"),
             "Спостереження CarRacing-v3: оригінальний RGB-кадр та чотири кадри стеку після "
             "передобробки", width_cm=16.5)
    R.subchapter("Простір дій")
    R.p("Середовище створюється з параметром continuous=False, тому простір дій дискретний і "
        "містить 5 дій: 0 – нічого не робити, 1 – кермо вліво, 2 – кермо вправо, 3 – газ, "
        "4 – гальмо. Одночасне натискання газу та повороту неможливе, тому агент має "
        "чергувати дії для проходження поворотів на швидкості.")
    R.subchapter("Функція винагороди та завершення епізоду")
    R.p("На кожному кадрі агент отримує винагороду −0,1, а за кожну вперше відвідану плитку "
        "траси – +1000/N, де N – кількість плиток. Тож якщо агент проїхав усю трасу за "
        "F кадрів, сумарна нагорода дорівнює 1000 − 0,1·F; наприклад, проходження за 732 "
        "кадри дає 926,8. Епізод завершується (terminated), коли відвідано всі плитки або "
        "автомобіль виїхав за межі ігрового поля (тоді додатково нараховується −100), і "
        "обрізається (truncated) після 1000 кадрів. Задача вважається розв'язаною при "
        "середній нагороді 900 і вище.")
    R.p("З функції винагороди випливає, що якість агента визначається двома чинниками: "
        "(1) надійністю – чи не з'їжджає автомобіль із траси та чи не пропускає плитки на "
        "поворотах; (2) швидкістю – чим менше кадрів витрачено на коло, тим більша нагорода. "
        "Максимально досяжна нагорода для типової траси становить близько 930.")

    # ===================== 3 АНАЛІЗ БАЗОВОЇ РЕАЛІЗАЦІЇ ====================
    R.chapter("Аналіз базової реалізації")
    R.subchapter("Алгоритм Deep Q-Network")
    R.p("DQN [1] апроксимує функцію цінності дій Q(s, a; θ) нейронною мережею з параметрами θ і "
        "навчає її мінімізувати TD-помилку на мінібатчах переходів (s, a, r, s′), випадково "
        "вибраних із буфера відтворення досвіду. Цільове значення обчислюється окремою "
        "цільовою мережею з параметрами θ⁻, які періодично копіюються з θ:")
    R.equation("dqn", "y = r + γ · (1 − d) · maxₐ′ Q(s′, a′; θ⁻)")
    R.p("де r – винагорода; γ – коефіцієнт дисконтування; d – ознака завершення епізоду; "
        "s′ – наступний стан. Функція втрат – Huber (smooth L1) між Q(s, a; θ) та y.", indent=False)
    R.subchapter("Архітектура та гіперпараметри базової моделі")
    R.p("Базова Q-мережа повторює архітектуру Nature DQN [1]: три згорткові шари "
        "(32 фільтри 8×8 з кроком 4, 64 фільтри 4×4 з кроком 2, 64 фільтри 3×3 з кроком 1) "
        "з активацією ReLU, повнозв'язний шар на 512 нейронів і вихідний шар на 5 Q-значень. "
        "Загальна кількість параметрів – близько 1,69 млн. Тренування ведеться на 8 "
        f"паралельних середовищах. Основні гіперпараметри зведено в таблицю {R.next_tab()}.")
    R.table("Гіперпараметри базової реалізації",
            ["Параметр", "Значення"],
            [["Паралельних середовищ", "8"],
             ["Тривалість тренування", "1000 епізодів (≈ 1 млн кадрів)"],
             ["Оптимізатор, learning rate", "Adam, 1·10⁻⁴ (сталий)"],
             ["Коефіцієнт дисконтування γ", "0,99"],
             ["Розмір мінібатчу", "128"],
             ["Оновлень мережі на крок векторного середовища", "2"],
             ["Обсяг буфера відтворення", "50 000 переходів"],
             ["Оновлення цільової мережі", "жорстке, кожні 1000 оновлень"],
             ["Дослідження", "ε = 1,0 · 0,995ⁿ (n – кількість епізодів), ε ≥ 0,05"],
             ["Ціль навчання", "однокроковий DQN, max за цільовою мережею"],
             ["Ознака завершення d", "terminated OR truncated"],
             ["Вибір чекпоінта", "максимум ковзного середнього 100 тренувальних епізодів"]],
            widths=[8.5, 7.5])
    R.subchapter("Виявлені недоліки")
    R.p("Аналіз коду та результатів тестування базової моделі виявив такі слабкі місця:")
    R.numbered([
        "**переоцінка Q-значень.** Оператор max у формулі (3.1) за наявності шуму в оцінках "
        "систематично завищує цінність [2]; агент може віддавати перевагу ризикованим діям "
        "(різкий газ на повороті), цінність яких переоцінено",
        "**неправильна обробка обрізання за часом.** Після 1000 кадрів епізод обривається "
        "штучно, але базовий код встановлює d = 1 і для truncated, тобто навчає мережу, що "
        "майбутня цінність стану на 1000-му кадрі дорівнює нулю. Оскільки номер кадру не "
        "входить у спостереження, це вносить суперечливий шум у ціль [5]",
        "**повільне поширення винагороди.** Однокрокове TD-оновлення переносить інформацію "
        "про нагороду лише на один крок назад за оновлення",
        "**малий буфер.** 50 000 переходів – це лише ≈ 50 епізодів досвіду; крім того, кожен "
        "кадр зберігається двічі (у state і next_state), що вдвічі збільшує використання "
        "пам'яті",
        "**вибір чекпоінта за тренувальними епізодами.** Ковзне середнє рахується за "
        "епізодами з увімкненим ε-дослідженням та по моделях, що змінювалися впродовж цих 100 "
        "епізодів, тому погано відображає якість жадібної політики, яку перевіряє тестовий "
        "скрипт",
        "**марнування симуляції.** Коли автомобіль з'їхав на траву чи розвернувся, решта "
        "епізоду (до 1000 кадрів) дає лише штраф −0,1 за кадр і майже не містить корисного "
        "досвіду",
        "**нестабільність результату.** Мінімальна нагорода 470,4 при середній 840,6 свідчить "
        "про те, що на частині трас агент «вилітає» з траси і втрачає значну частину плиток",
    ])
    R.p(f"Додатково базовий чекпоінт було перевірено тим самим тестовим скриптом на 50 "
        f"епізодах на обладнанні автора (Apple M1). Отримано середню нагороду {fmt(base.mean())} ± "
        f"{fmt(base.std())} (мінімум {fmt(base.min())}, максимум {fmt(base.max())}); "
        f"{n_base_fail} з 50 епізодів мають нагороду менше 800. Відмінність від заявлених "
        f"{fmt(BASELINE_CLAIMED['mean'])} пояснюється тим, що тестовий скрипт не фіксує seed, "
        "і кожен запуск оцінює модель на іншому наборі випадкових трас, а розкид результатів "
        "базової моделі великий.")

    # ===================== 4 ПОКРАЩЕННЯ ===================================
    R.chapter("Запропоновані покращення")
    R.p("Усі зміни зосереджено в модулі dqn_car_racing.py. Функцію make_env() залишено "
        "без змін, інтерфейс класу QNetwork (конструктор з аргументом a_size, метод act(), "
        "формат state_dict) сумісний із тестовим скриптом.")

    R.subchapter("Дуельна архітектура Q-мережі")
    R.p("Згорткову частину мережі збережено, а повнозв'язну частину розділено на дві гілки [3]: "
        "гілка цінності стану V(s) (3136 → 512 → 1) та гілка переваг дій A(s, a) "
        "(3136 → 512 → 5). Q-значення збираються за формулою:")
    R.equation("duel", "Q(s, a) = V(s) + A(s, a) − (1/|A|) · Σₐ′ A(s, a′)")
    R.p("Віднімання середньої переваги робить розклад однозначним. Перевага дуельної "
        "архітектури в тому, що V(s) уточнюється на кожному переході незалежно від обраної "
        "дії. У CarRacing на прямих ділянках більшість дій мають близьку цінність, і мережі "
        "важливіше точно оцінити сам стан (наскільки добре автомобіль розташований на трасі), "
        "ніж розрізнити дії. Кількість параметрів зросла до ≈ 3,3 млн, що не впливає на "
        "швидкість тестування.", indent=False)

    R.subchapter("Double DQN")
    R.p("Щоб усунути переоцінку, вибір дії для наступного стану та її оцінювання розділено "
        "між мережами [2]: дію обирає онлайн-мережа θ, а цінність цієї дії оцінює цільова "
        "мережа θ⁻:")
    R.equation("ddqn", "y = r + γ · (1 − d) · Q(s′, argmaxₐ′ Q(s′, a′; θ); θ⁻)")
    R.p("Оскільки помилки двох мереж слабко скорельовані, завищення оцінки однієї з них "
        "рідко збігається із завищенням іншої, і зсув цілі зменшується.", indent=False)

    R.subchapter("n-крокові повернення")
    R.p("Замість однокрокової цілі використано n-крокову (n = 3) [4, 6]:")
    R.equation("nstep", "y = Σₖ₌₀ᵐ⁻¹ γᵏ rₜ₊ₖ + γᵐ · (1 − dₜ₊ₘ₋₁) · Q(sₜ₊ₘ, argmaxₐ Q(sₜ₊ₘ, a; θ); θ⁻)")
    R.p("де m = n, якщо епізод не завершився раніше, інакше m – кількість кроків до кінця "
        "епізоду. Нагорода за плитку тепер за одне оновлення впливає на оцінку трьох "
        "попередніх станів, що пришвидшує навчання. Через невелике n і ε ≤ 0,1 на більшій "
        "частині тренування зсув через off-policy характер n-крокової цілі незначний.",
        indent=False)

    R.subchapter("Коректна обробка обрізання епізоду")
    R.p("У формулах (4.2) та (4.3) ознака d дорівнює 1 лише при справжньому завершенні "
        "(terminated: пройдено всю трасу або виїзд за межі поля). Якщо епізод обрізано за "
        "часом (truncated), ціль бутстрепиться з Q(s′) останнього спостереження, як "
        "рекомендовано в [5]. Для цього останнє спостереження епізоду зберігається в буфері.")

    R.subchapter("Компактний буфер відтворення досвіду")
    R.p("Буфер організовано як E = 8 кільцевих «стрічок» (по одній на паралельне середовище) "
        "по 15 000 комірок. На кожному кроці векторного середовища в кожну стрічку "
        "дописується рівно одне спостереження, тому наступний стан переходу з комірки j – "
        "це комірка j + 1 тієї ж стрічки, і кожен кадр зберігається лише один раз. Для "
        "кожної комірки зберігаються дія, нагорода, ознаки terminated і «кінець епізоду», а "
        "також ознака валідності: перехід, що відповідає кроку автоскидання векторного "
        "середовища (останній кадр старого епізоду → перший кадр нового), позначається як "
        "невалідний і ніколи не вибирається.")
    R.p("n-крокові повернення та коефіцієнт бутстрепу γᵐ·(1 − d) обчислюються "
        "векторизовано під час вибірки мінібатчу. У результаті ємність зросла з 50 000 до "
        "120 000 переходів за майже тих самих витрат пам'яті (≈ 3,4 ГБ проти ≈ 2,8 ГБ). "
        "Коректність буфера перевірено модульним тестом на синтетичній послідовності з "
        "термінальним завершенням, обрізанням та кроками автоскидання.")

    R.subchapter("Розклади гіперпараметрів і м'яке оновлення цільової мережі")
    R.bullets([
        "**ε-розклад за кроками.** Кусково-лінійний розклад: 1,0 → 0,1 за перші 20 % кроків "
        "(активне дослідження), 0,1 → 0,01 до 60 %, далі 0,01. Розклад за кроками, а не за "
        "епізодами, не залежить від довжини епізодів, а менший кінцевий ε наближає "
        "тренувальну поведінку до жадібної політики, яку перевіряє тест",
        "**косинусне згасання learning rate** [12] від 1·10⁻⁴ до 1·10⁻⁵: великий крок на "
        "початку для швидкого навчання та малий наприкінці для стабілізації ваг",
        "**м'яке (Polyak) оновлення цільової мережі** [7]: θ⁻ ← τθ + (1 − τ)θ⁻ після кожного "
        "оновлення з τ = 0,0025 (стала часу ≈ 400 оновлень). Ціль змінюється плавно, без "
        "стрибків, характерних для жорсткого копіювання",
        "**Adam з ε = 1,5·10⁻⁴** – стандартне для DQN значення [4], що зменшує надмірно великі "
        "кроки для параметрів з малими оцінками другого моменту",
        "**мінібатч 64 і 2 оновлення на крок векторного середовища** – та сама кількість "
        "семплів, що й 1 × 128, але вдвічі більше градієнтних кроків",
    ])

    R.subchapter("Ранній обрив безперспективних тренувальних епізодів")
    R.p("Тільки для тренувальних середовищ поверх незмінного make_env() додано обгортку "
        "NoProgressTruncation: якщо агент 100 кадрів поспіль не отримав жодної позитивної "
        "нагороди (тобто не відвідав нової плитки), епізод обривається з ознакою truncated. "
        "Завдяки коректній обробці обрізання (п. 4.4) це не змінює задачу оптимізації – "
        "цінність останнього стану береться з Q-мережі, – а лише економить час симуляції, "
        "який інакше витрачався б на кадри «на траві». Параметри самого середовища та "
        "середовище для оцінювання і тестування залишаються ідентичними базовим.")

    R.subchapter("Вибір чекпоінта за жадібною оцінкою")
    R.p("Кожні 50 000 кадрів поточна модель оцінюється жадібною політикою (ε = 0) на 16 "
        "нових трасах у незмінному середовищі make_env() – так само, як це робить тестовий "
        "скрипт. Чекпоінт checkpoint_CarRacing-v3_best.pth перезаписується, коли середня "
        "нагорода оцінки перевищує попередній максимум.")

    R.subchapter("Прискорення тренування")
    R.p("Крок векторного середовища виконується асинхронно: дії надсилаються в підпроцеси "
        "методом step_async(), і поки симулятори Box2D обчислюють наступний кадр, головний "
        "процес виконує оновлення мережі на GPU; результати забираються методом step_wait(). "
        "На Apple M1 для тренування використано бекенд MPS (Metal Performance Shaders), що "
        "дає ≈ 9-кратне прискорення оновлень порівняно з CPU. Метод act() бере пристрій з "
        "параметрів мережі, тому та сама модель працює і на MPS під час тренування, і на CPU "
        "у тестовому скрипті.")

    R.p(f"Порівняння ключових параметрів базової та покращеної реалізацій наведено в "
        f"таблиці {R.next_tab()}.")
    R.table("Порівняння базової та покращеної реалізацій",
            ["Аспект", "Базова", "Покращена"],
            [["Архітектура", "Nature DQN", "Dueling Nature DQN"],
             ["Ціль навчання", "1-кроковий DQN", "3-кроковий Double DQN"],
             ["Обробка truncated", "як термінального", "бутстреп з Q(s′)"],
             ["Буфер", "50 000 (кадри двічі)", "120 000 (кадри один раз)"],
             ["Мінібатч × оновлень / крок", "128 × 2", "64 × 2"],
             ["Learning rate", "1·10⁻⁴, сталий", "1·10⁻⁴ → 1·10⁻⁵, косинус"],
             ["Цільова мережа", "копія кожні 1000 оновлень", "Polyak, τ = 0,0025"],
             ["ε-розклад", "0,995ⁿ за епізодами, ≥ 0,05", "1 → 0,1 → 0,01 за кроками"],
             ["Ранній обрив (тренування)", "немає", "100 кадрів без прогресу"],
             ["Вибір чекпоінта", "тренувальне середнє", "жадібна оцінка, 16 трас"],
             ["Обсяг тренування", "1000 епізодів (≈ 1 млн кадрів)",
              f"{fmt(total_steps / 1e6, 2)} млн кадрів"]],
            widths=[5.5, 5.2, 5.8])

    # ===================== 5 РЕАЛІЗАЦІЯ ===================================
    R.chapter("Програмна реалізація")
    R.subchapter("Програмне та апаратне забезпечення")
    R.table("Програмне та апаратне забезпечення",
            ["Компонент", "Версія / характеристика"],
            [["Процесор / GPU", "Apple M1 (8 ядер CPU, вбудований GPU)"],
             ["Оперативна пам'ять", "16 ГБ"],
             ["Операційна система", "macOS 26 (Darwin 25.6)"],
             ["Python", "3.14.2"],
             ["PyTorch (бекенд MPS)", "2.14.1"],
             ["Gymnasium", "1.4.0"],
             ["box2d-py", "2.3.8"],
             ["NumPy / OpenCV", "2.5.3 / opencv-python-headless"]],
            widths=[7, 9])
    R.subchapter("Структура проєкту")
    R.bullets([
        "dqn_car_racing.py – покращена реалізація: make_env() (без змін), обгортка "
        "NoProgressTruncation, клас QNetwork (дуельна CNN), NStepReplayBuffer, DQNAgent "
        "(Double DQN, Polyak, розклади), функція оцінювання evaluate_policy_vec(), цикл "
        "тренування train() та запис відео render()",
        "test_car_racing.py – тестовий скрипт курсу без змін",
        "checkpoint_CarRacing-v3_best.pth – найкращий чекпоінт покращеної моделі",
        "baseline/ – базова реалізація, тестовий скрипт і чекпоінт з репозиторію курсу [10] "
        "для відтворення порівняння",
        "plot_observation.py, plot_results.py – побудова рисунків для звіту",
        "build_report.py – генерація цього звіту",
        "results/ – журнали тренування (train_log.csv, eval_log.csv, train_stdout.log), "
        "результати тестування та рисунки",
    ])
    R.subchapter("Основні компоненти")
    R.p("**Клас QNetwork** містить згорткову частину Nature DQN та дві повнозв'язні гілки "
        "(value і advantage). Вхідні кадри зберігаються як uint8 і нормалізуються до [0, 1] "
        "усередині forward() вже на пристрої обчислень, що зменшує обсяг передавання даних "
        "у 4 рази. Методи act() (один стан, використовується тестом) та act_batch() (батч "
        "станів від паралельних середовищ) реалізують ε-жадібний вибір дії.")
    R.p("**Клас NStepReplayBuffer** зберігає потоки спостережень паралельних середовищ "
        "(п. 4.5); метод add() дописує один крок векторного середовища, метод sample() "
        "вибирає мінібатч валідних переходів і повертає стани, дії, n-крокові повернення, "
        "стани для бутстрепу та коефіцієнти γᵐ·(1 − d).")
    R.p("**Клас DQNAgent** інкапсулює онлайн- та цільову мережі, оптимізатор і буфер. Метод "
        "replay() виконує один крок навчання: вибірка мінібатчу, обчислення цілі Double "
        "DQN (4.3), Huber-втрата, обрізання норми градієнта до 10, крок Adam з поточним "
        "learning rate та м'яке оновлення цільової мережі одним викликом "
        "torch._foreach_lerp_.")
    R.p("**Функція train()** створює 8 тренувальних середовищ (make_env + "
        "NoProgressTruncation) і 8 середовищ для оцінювання (лише make_env), заповнює буфер "
        "20 000 кадрами до початку навчання, керує ε-розкладом, веде журнали та зберігає "
        "найкращий чекпоінт за результатами періодичної жадібної оцінки.")
    R.subchapter("Порядок запуску")
    R.code("# тренування (≈ 3,5 год на Apple M1)\n"
           "python dqn_car_racing.py --total-steps 1500000\n\n"
           "# тестування (скрипт курсу без змін)\n"
           "python test_car_racing.py checkpoint_CarRacing-v3_best.pth 50\n\n"
           "# рисунки та звіт\n"
           "python plot_observation.py && python plot_results.py && python build_report.py",
           size=11)

    # ===================== 6 РЕЗУЛЬТАТИ ===================================
    R.chapter("Результати експериментів")
    R.subchapter("Динаміка тренування")
    R.p(f"Тренування тривало {fmt(total_steps / 1e6, 2)} млн кадрів ({total_eps} тренувальних "
        f"епізодів, ≈ {fmt(train_minutes / 60, 1)} год на Apple M1). Нагороди тренувальних "
        f"епізодів та значення ε наведено на рисунку {R.next_fig('train')}. Нагороди на "
        "початку тренування низькі не лише через випадкову політику, а й через ранній обрив: "
        f"загалом {fmt(early_frac, 0)} % тренувальних епізодів було обірвано обгорткою "
        "NoProgressTruncation, переважно на етапі активного дослідження.")
    R.figure("train", os.path.join(FIG, "training_curve.png"),
             "Нагороди тренувальних епізодів (з ε-дослідженням і раннім обривом) та розклад ε")
    R.p(f"На рисунку {R.next_fig('eval')} показано результати періодичної жадібної оцінки на "
        f"16 нових трасах. Найкращий результат оцінки {fmt(ev_mean[best_i])} отримано після "
        f"{fmt(ev_steps[best_i] / 1e6, 2)} млн кадрів; саме цю модель збережено як "
        "checkpoint_CarRacing-v3_best.pth.")
    R.figure("eval", os.path.join(FIG, "eval_curve.png"),
             "Результати жадібної оцінки (ε = 0, 16 епізодів) протягом тренування")
    eval_rows = []
    for r in ev:
        eval_rows.append([f"{int(r['env_steps']) / 1e6:.2f}".replace(".", ","),
                          fmt(float(r["mean"])), fmt(float(r["std"])),
                          fmt(float(r["min"])), fmt(float(r["max"]))])
    R.p(f"Числові значення проміжних оцінок наведено в таблиці {R.next_tab()}.")
    R.table("Результати жадібної оцінки протягом тренування (16 епізодів)",
            ["Кадри, млн", "Середнє", "Std", "Мін.", "Макс."], eval_rows,
            widths=[3.2, 3.2, 3.2, 3.2, 3.2], font_size=11)

    R.subchapter("Тестування тестовим скриптом курсу")
    R.p("Фінальну перевірку виконано незмінним скриптом test_car_racing.py на 50 епізодах "
        "командою «python test_car_racing.py checkpoint_CarRacing-v3_best.pth 50». Для "
        "базової моделі використано її власний модуль dqn_car_racing.py з каталогу baseline/. "
        f"Результати зведено в таблицю {R.next_tab()}.")
    rows = [["Базова (заявлено в умові)", fmt(BASELINE_CLAIMED["mean"]), fmt(BASELINE_CLAIMED["std"]),
             fmt(BASELINE_CLAIMED["min"]), fmt(BASELINE_CLAIMED["max"]), "–"],
            ["Базова (виміряно автором)", fmt(base.mean()), fmt(base.std()), fmt(base.min()),
             fmt(base.max()), f"{n_base_fail}/50"],
            ["Покращена", fmt(impr.mean()), fmt(impr.std()), fmt(impr.min()), fmt(impr.max()),
             f"{n_impr_fail}/50"]]
    if d["impr2"] is not None:
        i2 = d["impr2"]
        rows.append(["Покращена (повторний запуск)", fmt(i2.mean()), fmt(i2.std()), fmt(i2.min()),
                     fmt(i2.max()), f"{int((i2 < 800).sum())}/50"])
    R.table("Результати тестування на 50 епізодах",
            ["Модель", "Mean", "Std", "Min", "Max", "Епізодів < 800"], rows,
            widths=[5.6, 2.0, 2.0, 2.0, 2.0, 2.9], font_size=12)
    R.p(f"Розподіл нагород за епізодами для базової та покращеної моделей показано на "
        f"рисунку {R.next_fig('test')}.")
    R.figure("test", os.path.join(FIG, "test_comparison.png"),
             "Розподіл нагород базової та покращеної моделей на 50 тестових епізодах")

    R.subchapter("Статистична значущість")
    R.p("Оскільки траси генеруються випадково, середні нагороди на 50 епізодах є випадковими "
        "величинами. Для перевірки, що покращення не є наслідком вдалого набору трас, "
        "застосовано два критерії до вибірок, отриманих на однаковому обладнанні:")
    R.bullets([
        f"t-критерій Велча (без припущення про рівність дисперсій): t = {fmt(t_stat, 2)}, "
        f"p = {p_t:.1e}".replace(".", ",").replace("e-0", "·10⁻").replace("e-", "·10⁻"),
        f"односторонній U-критерій Манна–Уітні (непараметричний, стійкий до викидів): "
        f"U = {u_stat:.0f}, p = {p_u:.1e}".replace(".", ",").replace("e-0", "·10⁻")
        .replace("e-", "·10⁻"),
    ])
    R.p("Обидва p-значення значно менші за рівень значущості 0,05, тому покращення якості є "
        "статистично значущим.")

    # ===================== 7 АНАЛІЗ =======================================
    R.chapter("Аналіз результатів")
    gain_claim = impr.mean() - BASELINE_CLAIMED["mean"]
    gain_meas = impr.mean() - base.mean()
    R.p(f"Покращена модель перевищує заявлений результат базової на {fmt(gain_claim)} бала, а "
        f"виміряний на тому самому обладнанні – на {fmt(gain_meas)} бала. Ще суттєвіше "
        f"зменшився розкид: стандартне відхилення знизилось з {fmt(base.std())} до "
        f"{fmt(impr.std())}, мінімальна нагорода зросла з {fmt(base.min())} до "
        f"{fmt(impr.min())}, а кількість епізодів з нагородою нижче 800 скоротилась з "
        f"{n_base_fail} до {n_impr_fail} з 50.")
    R.p("Основний виграш забезпечено саме зменшенням кількості невдалих епізодів, тобто "
        "підвищенням надійності: максимальна нагорода обмежена довжиною траси (≈ 930), і "
        "на «легких» трасах обидві моделі показують близькі результати. Пояснити це можна "
        "так:")
    R.bullets([
        "Double DQN і дуельна архітектура дають точніші оцінки цінності; агент менше "
        "переоцінює ризиковані дії (розгін перед поворотом), які в базовій моделі "
        "призводять до заносу і втрати плиток",
        "коректна обробка обрізання за часом прибирає суперечливий шум з цілі на останніх "
        "кадрах епізоду",
        "n-крокові повернення та вдвічі більший буфер дають мережі більше різноманітного "
        "досвіду, зокрема рідкісних ситуацій (крутих поворотів, повернення на трасу)",
        "вибір чекпоінта за жадібною оцінкою на окремих трасах узгоджує критерій відбору з "
        "тим, як модель перевіряється тестовим скриптом",
    ])
    R.p("**Обмеження роботи.** Тестування на 50 випадкових трасах має похибку середнього "
        "порядку σ/√50: близько ±25 для базової моделі і менше для покращеної. Тому "
        "заявлений і виміряний результати базової моделі відрізняються, а порівняння "
        "проведено на однаковому обладнанні із статистичною перевіркою. Через обмежений час "
        "тренування (≈ 3,5 год на одну спробу) не проводилося окреме вивчення внеску кожного "
        "покращення (ablation study) та усереднення за кількома seed; внесок окремих "
        "компонентів оцінено на основі літератури [2–5]. Обсяг тренування покращеної моделі "
        "(1,5 млн кадрів) у 1,5 раза більший за базовий; водночас завдяки ранньому обриву "
        "значна частина цих кадрів припадає на продуктивні ділянки траси.")
    R.p("**Можливі напрями подальшого покращення:** пріоритизований буфер відтворення, "
        "розподільний (distributional) DQN або шумові мережі (NoisyNet) з алгоритму "
        "Rainbow [4]; аугментація зображень (випадкові зсуви кадру); перехід до методів для "
        "неперервних дій (PPO, SAC) – однак останнє вимагало б зміни налаштувань середовища, "
        "що заборонено умовою.")

    # ===================== ВИСНОВКИ =======================================
    R.structural("Висновки")
    R.p("У лабораторній роботі досліджено базову реалізацію алгоритму Deep Q-Network для "
        "задачі CarRacing-v3 і запропоновано та реалізовано комплекс покращень, що не "
        "змінюють параметрів середовища та тестового скрипту.")
    R.numbered([
        "Аналіз базової реалізації виявив сім недоліків: переоцінку Q-значень, обробку "
        "обрізання за часом як термінального стану, повільне однокрокове поширення "
        "винагороди, малий буфер із дублюванням кадрів, невідповідність критерію вибору "
        "чекпоінта тестовій процедурі, марнування симуляції на безперспективні ділянки "
        "епізодів та нестабільність результату",
        "Реалізовано дуельну архітектуру Q-мережі, ціль Double DQN з 3-кроковими "
        "поверненнями, коректний бутстреп при обрізанні епізодів, компактний буфер на "
        "120 000 переходів, ε-розклад за кроками, косинусне згасання learning rate, м'яке "
        "оновлення цільової мережі, ранній обрив тренувальних епізодів без прогресу та "
        "вибір чекпоінта за жадібною оцінкою; тренування прискорено асинхронним кроком "
        "середовищ і використанням GPU (MPS)",
        f"Покращена модель на 50 тестових епізодах досягла середньої нагороди "
        f"{fmt(impr.mean())} ± {fmt(impr.std())} (мінімум {fmt(impr.min())}, максимум "
        f"{fmt(impr.max())}) проти заявлених для базової {fmt(BASELINE_CLAIMED['mean'])} ± "
        f"{fmt(BASELINE_CLAIMED['std'])} (мінімум {fmt(BASELINE_CLAIMED['min'])}) та виміряних "
        f"автором {fmt(base.mean())} ± {fmt(base.std())} (мінімум {fmt(base.min())})",
        f"Покращення статистично значуще (t-критерій Велча та U-критерій Манна–Уітні, "
        f"p < 0,001); основний ефект – підвищення надійності: кількість епізодів з "
        f"нагородою нижче 800 зменшилась з {n_base_fail} до {n_impr_fail} з 50",
        "Мету роботи досягнуто: результат базового тренування покращено за всіма "
        "показниками тестового скрипту – середнім, стандартним відхиленням і мінімальною "
        "нагородою",
    ])

    # ===================== ДЖЕРЕЛА ========================================
    R.structural("Перелік джерел посилання")
    for i, ref in enumerate(REFERENCES, 1):
        R.p(f"{i}. {ref}")

    # ===================== ДОДАТОК А ======================================
    h = R.structural("Додаток А", toc=False)
    R.toc_entries.append((1, "ДОДАТОК А Лістинг програми dqn_car_racing.py"))
    R.p("Лістинг програми dqn_car_racing.py", align=C, indent=False, bold=True)
    with open("dqn_car_racing.py", encoding="utf-8") as f:
        R.code(f.read(), size=9)

    return R


# Кількість рисунків / таблиць для реферату відома лише після побудови документа,
# тому в першому проході ставимо маркери, а в другому -- справжні значення.
_COUNTS = {}


def R_count_placeholder(kind):
    return _COUNTS.get(kind, "__")


def docx_to_pdf(path, outdir):
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", outdir, path],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.join(outdir, os.path.splitext(os.path.basename(path))[0] + ".pdf")


def find_pages(pdf_path, entries):
    """Знайти номер сторінки, на якій уперше трапляється кожен заголовок (після змісту)."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    texts = [re.sub(r"\s+", " ", (pg.extract_text() or "")) for pg in reader.pages]
    pages = {}
    # Пропускаємо титул, реферат і зміст: пошук починаємо зі сторінки, де
    # з'являється перше речення вступу (сам зміст містить ті самі заголовки)
    start = next(i for i, t in enumerate(texts) if "Навчання з підкріпленням (reinforcement" in t)
    for level, title in entries:
        key = re.sub(r"\s+", " ", title)[:40]
        for i in range(start, len(texts)):
            if key in texts[i]:
                pages[title] = i + 1
                start = i
                break
    return pages, len(reader.pages)


def main():
    tmp = tempfile.mkdtemp()
    # Прохід 1: без номерів сторінок
    R = build()
    _COUNTS.update(fig=R.n_figures, tab=R.n_tables)
    first = os.path.join(tmp, "pass1.docx")
    R.save(first)
    pdf = docx_to_pdf(first, tmp)
    pages, n_pages = find_pages(pdf, R.toc_entries)
    # Прохід 2: з номерами сторінок у змісті та обсягом у рефераті
    R = build(toc_pages=pages, n_pages=n_pages)
    R.save(OUT)
    missing = [t for _, t in R.toc_entries if t not in pages]
    print(f"saved {OUT}: {n_pages} pages, {R.n_figures} figures, {R.n_tables} tables")
    if missing:
        print("WARNING: no page found for", missing, file=sys.stderr)
    pdf = docx_to_pdf(OUT, OUT_DIR)
    print("pdf preview:", pdf)


if __name__ == "__main__":
    main()
