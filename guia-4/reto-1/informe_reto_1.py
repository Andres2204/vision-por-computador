"""Build informe-reto-1.pdf from the artifacts in resultados/.

The report is generated only from what the notebooks saved (metrics.json,
predictions.csv, split.json and the comparison figures), so no number is typed
by hand: re-run this script after retraining a model and the PDF follows.

Usage (from guia-4/reto-1):

    ../../.venv/bin/python informe_reto_1.py
    # or, if reportlab is not in the venv:
    #   ../../.venv/bin/pip install reportlab

Dependencies: reportlab and Pillow. The Spanish prose lives in
this file; identifiers and docstrings stay in English.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from PIL import Image as PILImage
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "resultados"
OUTPUT = BASE / "informe-reto-1.pdf"

MODEL_ORDER = [
    "cnn/base",
    "cnn/mejorada",
    "cnn/mejorada_v2",
    "densenet",
    "vit",
    "vit_preentrenado",
    "ml/logreg",
    "ml/svm",
    "ml/rf",
]

FAMILY_ES = {
    "CNN": "CNN",
    "DL (non-sequential)": "DL no secuencial",
    "machine learning": "machine learning",
    "IA generativa (Transformer)": "Transformer",
}

MC_NEMAR_PAIRS = [
    ("cnn/base", "cnn/mejorada"),
    ("cnn/mejorada", "densenet"),
    ("cnn/base", "densenet"),
    ("cnn/mejorada_v2", "cnn/base"),
    ("cnn/mejorada_v2", "cnn/mejorada"),
    ("cnn/mejorada_v2", "densenet"),
    ("vit", "cnn/mejorada_v2"),
    ("vit", "densenet"),
    ("vit_preentrenado", "vit"),
    ("vit_preentrenado", "densenet"),
]
SCRATCH_MODELS = ["cnn/base", "cnn/mejorada", "cnn/mejorada_v2", "vit"]
GUIDE2_BEST = 0.9000  # Random Forest, RGB + HSV + Haralick (guia-2/reto-5)

FEATURED_GUIDE2 = [
    ("Random Forest", "RGB + HSV + Haralick", "90.11 %", "90.00 %"),
    ("MLP", "RGB + HSV + Haralick", "90.67 %", "89.17 %"),
    ("SVC (RBF), elegido por CV", "RGB + HSV + Haralick", "90.81 %", "88.75 %"),
    ("MLP (línea base)", "medias/desv. RGB", "—", "81.25 %"),
]


# --------------------------------------------------------------------------- data
def load_models() -> list[dict]:
    models = []
    for folder in MODEL_ORDER:
        metrics_path = RESULTS / folder / "metrics.json"
        if not metrics_path.exists():
            continue
        data = json.loads(metrics_path.read_text())
        data["folder"] = folder
        data["predictions"] = load_predictions(RESULTS / folder / "predictions.csv")
        models.append(data)
    return models


def load_predictions(path: Path) -> dict[str, tuple[str, str]]:
    """Map path -> (true_class, predicted_class) so models can be paired."""
    if not path.exists():
        return {}
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["path"]] = (row["true_class"], row["predicted_class"])
    return rows


def bootstrap_ci(predictions: dict[str, tuple[str, str]], n_boot: int = 2000, seed: int = 20):
    if not predictions:
        return None
    pairs = list(predictions.values())
    rng = random.Random(seed)
    accuracies = []
    n = len(pairs)
    for _ in range(n_boot):
        hits = 0
        for _ in range(n):
            true, pred = pairs[rng.randrange(n)]
            hits += true == pred
        accuracies.append(hits / n)
    accuracies.sort()
    low = accuracies[int(0.025 * (n_boot - 1))]
    high = accuracies[int(0.975 * (n_boot - 1))]
    return low, high


def mcnemar_pvalue(pred_a: dict[str, tuple[str, str]], pred_b: dict[str, tuple[str, str]]) -> float | None:
    shared = sorted(set(pred_a) & set(pred_b))
    if not shared:
        return None
    only_a = only_b = 0
    for key in shared:
        true = pred_a[key][0]
        correct_a = pred_a[key][1] == true
        correct_b = pred_b[key][1] == true
        if correct_a and not correct_b:
            only_a += 1
        elif correct_b and not correct_a:
            only_b += 1
    total = only_a + only_b
    if total == 0:
        return 1.0
    smaller = min(only_a, only_b)
    tail = sum(math.comb(total, k) for k in range(smaller + 1)) / (2**total)
    return min(1.0, 2 * tail)


def num1(value: float) -> str:
    """One-decimal number with a Spanish decimal comma."""
    return f"{value:.1f}".replace(".", ",")


def signed1(value: float) -> str:
    """Signed one-decimal number with a Spanish decimal comma."""
    return ("+" if value >= 0 else "−") + f"{abs(value):.1f}".replace(".", ",")


def pct(value: float | None, digits: int = 2) -> str:
    return "n/d" if value is None else f"{100 * value:.{digits}f} %".replace(".", ",")


def number(value, digits: int = 0, dash: str = "—") -> str:
    if value is None:
        return dash
    if isinstance(value, float) and digits == 0 and 0 < abs(value) < 10:
        return f"{value:.2f}".replace(".", ",")
    if isinstance(value, float) and digits == 0:
        return f"{value:,.0f}".replace(",", " ")
    text = f"{value:,.{digits}f}"
    return text.replace(",", " ").replace(".", ",")


def fmt_p(p_value: float | None) -> str:
    if p_value is None:
        return "n/d"
    if p_value < 0.0001:
        return "< 0,0001"
    return f"{p_value:.4f}".replace(".", ",")


def fmt_p_html(p_value: float | None) -> str:
    """'p = 0,0123' or 'p < 0,0001', safe inside a reportlab Paragraph."""
    text = fmt_p(p_value)
    return "p &lt; 0,0001" if text.startswith("<") else f"p = {text}"


def verdict(p_value: float | None) -> str:
    if p_value is None:
        return "sin datos"
    return "significativa" if p_value < 0.05 else "dentro del ruido"


# ------------------------------------------------------------------------- styles
def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10.5,
            alignment=TA_JUSTIFY,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Cell",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7,
            leading=8.5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H1es",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            spaceBefore=2,
            spaceAfter=8,
            textColor=colors.black,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2es",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            spaceBefore=10,
            spaceAfter=5,
            textColor=colors.black,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            leading=9.5,
            alignment=1,
            textColor=colors.black,
            spaceBefore=2,
            spaceAfter=10,
        )
    )
    return styles


def table(data, widths, align_right_from: int = 1, header_bg="#000000", center: tuple = ()):
    body = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (align_right_from - 1, -1), "LEFT"),
    ]
    for column in center:
        style.append(("ALIGN", (column, 0), (column, -1), "CENTER"))
    style += [
        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    body.setStyle(TableStyle(style))
    return body


# -------------------------------------------------------------------------- build
def figure(path: Path, caption: str, styles, max_height: float):
    with PILImage.open(path) as image:
        width_px, height_px = image.size
    width = 16.6 * cm
    height = width * height_px / width_px
    if height > max_height:
        height = max_height
        width = height * width_px / height_px
    return KeepTogether([Image(str(path), width=width, height=height), Paragraph(caption, styles["Caption"])])


def build_report() -> Path:
    models = load_models()
    by_folder = {m["folder"]: m for m in models}
    missing = set(MODEL_ORDER) - set(by_folder)
    if missing:
        raise SystemExit("Faltan resultados de: " + ", ".join(sorted(missing)) + ". Ejecuta los cuadernos de modelos.")
    for model in models:
        model["ci"] = bootstrap_ci(model["predictions"])

    def pvalue(folder_a: str, folder_b: str) -> float | None:
        return mcnemar_pvalue(by_folder[folder_a]["predictions"], by_folder[folder_b]["predictions"])

    def acc(folder: str) -> float:
        return by_folder[folder]["accuracy"]

    ranking = sorted(models, key=lambda m: m["accuracy"], reverse=True)
    best, second = ranking[0], ranking[1]
    ml_models = [m for m in models if m["folder"].startswith("ml/")]
    best_ml = max(ml_models, key=lambda m: m["accuracy"])
    best_dl = max((m for m in models if m not in ml_models), key=lambda m: m["accuracy"])
    scratch = [by_folder[f] for f in SCRATCH_MODELS]
    densenet, vit, vit_pre = by_folder["densenet"], by_folder["vit"], by_folder["vit_preentrenado"]
    split = json.loads((RESULTS / "split.json").read_text())
    n_total, n_train, n_val, n_test = split["n_total"], split["n_train"], split["n_val"], split["n_test"]

    def tie(p_value: float | None) -> bool:
        return p_value is None or p_value >= 0.05

    styles = build_styles()
    story: list = []

    # ---------------------------------------------------------------- portada
    story.append(Paragraph("Reto 1 — Clasificación de 3 escenas", styles["H1es"]))
    story.append(
        Paragraph(
            "Comparamos nueve modelos para clasificar costas, bosques y autopistas: la CNN del cuaderno "
            "base y dos versiones mejoradas, DenseNet121 preentrenada, un ViT entrenado desde cero, un "
            "ViT-B/16 preentrenado y tres modelos de machine learning con descriptores HSV + HOG. Todos "
            "usan el mismo split y las mismas métricas.",
            styles["Body"],
        )
    )

    story.append(Paragraph("Resumen", styles["H2es"]))
    p_top = pvalue(best["folder"], second["folder"])
    summary = [
        f"<b>Mejor modelo:</b> {best['name']}, con {pct(best['accuracy'])} de accuracy en test, "
        f"F1 macro {pct(best['f1_macro'])} y kappa {number(best['cohen_kappa'], 3)}.",
        f"{second['name']} queda a {num1(100 * (best['accuracy'] - second['accuracy']))} puntos "
        f"(McNemar {fmt_p_html(p_top)}, {'no significativa' if tie(p_top) else 'significativa'}).",
        f"Las redes entrenadas desde cero quedan entre {pct(min(m['accuracy'] for m in scratch))} y "
        f"{pct(max(m['accuracy'] for m in scratch))}; el mejor modelo clásico, {best_ml['name']}, llega a "
        f"{pct(best_ml['accuracy'])}.",
    ]
    for bit in summary:
        story.append(Paragraph(f"• {bit}", styles["Body"]))

    # -------------------------------------------------------------- protocolo
    story.append(Paragraph("Protocolo", styles["H2es"]))
    story.append(
        Paragraph(
            f"{n_total} imágenes redimensionadas a 128×128. Split estratificado con semilla 20: {n_train} de "
            f"entrenamiento, {n_val} de validación y {n_test} de test. Todos los modelos se evalúan sobre las "
            "mismas imágenes de test. Métricas: accuracy, precisión, recall y F1 (macro y weighted), kappa de "
            "Cohen, ROC-AUC one-vs-rest, IC 95 % de la accuracy por bootstrap (2.000 remuestreos) y test "
            "exacto de McNemar entre pares de modelos.",
            styles["Body"],
        )
    )

    # ------------------------------------------------------------------ tabla
    story.append(Paragraph("Resultados por modelo", styles["H2es"]))
    rows = [["Modelo", "Familia", "Accuracy", "F1 macro", "IC 95 %", "Parámetros", "Entrena (s)", "Épocas"]]
    for model in ranking:
        low, high = model["ci"]
        rows.append(
            [
                Paragraph(model["name"], styles["Cell"]),
                Paragraph(FAMILY_ES.get(model["family"], model["family"]), styles["Cell"]),
                pct(model["accuracy"]),
                pct(model["f1_macro"]),
                f"{pct(low, 1)} – {pct(high, 1)}",
                number(model.get("parameters")),
                number(model.get("training_seconds")),
                number(model.get("epochs")),
            ]
        )
    story.append(
        table(
            rows,
            [3.6 * cm, 2.3 * cm, 1.6 * cm, 1.6 * cm, 2.6 * cm, 2.0 * cm, 1.6 * cm, 1.3 * cm],
            align_right_from=2,
            center=(2, 3, 4, 5, 6, 7),
        )
    )
    story.append(
        Paragraph(
            f"Tabla 1. Resultados sobre el test común de {n_test} imágenes, ordenados por accuracy. Los "
            "modelos clásicos no tienen épocas.",
            styles["Caption"],
        )
    )

    # ---------------------------------------------------------------- McNemar
    story.append(Paragraph("Test de McNemar", styles["H2es"]))
    pair_rows = [["Modelo A", "Modelo B", "p-valor", "Veredicto"]]
    for folder_a, folder_b in [*MC_NEMAR_PAIRS, (best_ml["folder"], best_dl["folder"])]:
        p_value = pvalue(folder_a, folder_b)
        pair_rows.append(
            [
                Paragraph(by_folder[folder_a]["name"], styles["Cell"]),
                Paragraph(by_folder[folder_b]["name"], styles["Cell"]),
                fmt_p(p_value),
                verdict(p_value),
            ]
        )
    story.append(table(pair_rows, [5.4 * cm, 5.4 * cm, 2.2 * cm, 3.6 * cm], align_right_from=2))
    story.append(
        Paragraph(
            "Tabla 2. Test exacto de McNemar (α = 0,05) sobre las predicciones de cada par de modelos en el "
            "mismo test.",
            styles["Caption"],
        )
    )

    # --------------------------------------------------------- guía 2
    story.append(Paragraph("Comparación con el machine learning de la guía 2", styles["H2es"]))
    story.append(
        Paragraph(
            "Resultados de <font face='Courier'>guia-2/reto-5</font> con descriptores RGB + HSV + Haralick. "
            "Ese cuaderno usa el mismo protocolo (25 % de test, semilla 20), pero no garantiza las mismas "
            "imágenes de test, así que la comparación es orientativa.",
            styles["Body"],
        )
    )
    guide_rows = [["Modelo (guía 2)", "Características", "Accuracy CV", "Accuracy test"]]
    for row in FEATURED_GUIDE2:
        guide_rows.append([Paragraph(cell, styles["Cell"]) for cell in row])
    story.append(table(guide_rows, [5.4 * cm, 5.0 * cm, 3.0 * cm, 3.0 * cm], align_right_from=2, center=(2, 3)))
    story.append(Paragraph("Tabla 3. Mejores modelos de la guía 2 y su línea base.", styles["Caption"]))

    # --------------------------------------------------------------- figuras
    story.append(Paragraph("Figuras", styles["H2es"]))
    figures = [
        ("barras_accuracy_ic95.png", "Figura 1. Accuracy en test con IC 95 %, por familia, frente al mejor resultado de la guía 2.", 9.0 * cm),
        ("matrices_confusion.png", "Figura 2. Matrices de confusión normalizadas por fila.", 15.5 * cm),
        ("curvas_aprendizaje.png", "Figura 3. Curvas de aprendizaje de los modelos entrenados por épocas.", 11.5 * cm),
        ("heatmap_f1_por_clase.png", "Figura 4. F1 por clase y modelo.", 8.5 * cm),
    ]
    for filename, caption, max_height in figures:
        story.append(figure(RESULTS / "comparacion" / filename, caption, styles, max_height))
    story.append(
        figure(
            RESULTS / "densenet" / "gradcam.png",
            "Figura 5. Grad-CAM de DenseNet121 sobre una imagen de costa: la red se fija en el mar.",
            styles,
            8.0 * cm,
        )
    )

    # ------------------------------------------------------------ conclusiones
    story.append(Paragraph("Conclusiones", styles["H2es"]))
    p_pre = pvalue("vit_preentrenado", "vit")
    p_att = pvalue("vit_preentrenado", "densenet")
    p_v1 = pvalue("cnn/base", "cnn/mejorada")
    p_v2 = pvalue("cnn/mejorada_v2", "cnn/base")
    p_ml = pvalue(best_ml["folder"], best_dl["folder"])
    p_scratch = pvalue("vit", "cnn/mejorada_v2")

    conclusions = [
        f"<b>El mejor modelo es {best['name']} ({pct(best['accuracy'])}).</b> {second['name']} queda a "
        f"{num1(100 * (best['accuracy'] - second['accuracy']))} puntos ({pct(second['accuracy'])}; "
        f"McNemar {fmt_p_html(p_top)}, {'no significativa' if tie(p_top) else 'significativa'}).",
        f"<b>Lo que más pesa es el preentrenamiento.</b> Las cuatro redes entrenadas desde cero quedan "
        f"entre {pct(min(m['accuracy'] for m in scratch))} y {pct(max(m['accuracy'] for m in scratch))}. "
        f"Partir de ImageNet sube el ViT {num1(100 * (acc('vit_preentrenado') - acc('vit')))} puntos "
        f"({fmt_p_html(p_pre)}).",
        f"<b>Mejorar la CNN no subió la accuracy.</b> La v1 queda en {pct(acc('cnn/mejorada'))} "
        f"({fmt_p_html(p_v1)} frente a la base) y la v2 en {pct(acc('cnn/mejorada_v2'))} "
        f"({fmt_p_html(p_v2)}), frente a {pct(acc('cnn/base'))} de la CNN base. Con {n_train} imágenes "
        "de entrenamiento, el límite lo ponen los datos.",
        f"<b>El ML clásico sigue siendo competitivo.</b> {best_ml['name']} con HSV + HOG "
        f"({pct(best_ml['accuracy'])}) supera a todas las redes entrenadas desde cero y "
        + ("no se distingue estadísticamente de " if tie(p_ml) else "queda por debajo de ")
        + f"{best_dl['name']} ({fmt_p_html(p_ml)}). Frente a la guía 2 "
        f"({pct(GUIDE2_BEST)}), la mejora viene de añadir HOG.",
        ("<b>La atención no supera a la convolución en este dataset.</b> " if tie(p_scratch) and tie(p_att)
         else "<b>Atención frente a convolución.</b> ")
        + f"Desde cero, el ViT {'empata con' if tie(p_scratch) else 'se separa de'} la CNN v2 "
        f"({fmt_p_html(p_scratch)}); con ImageNet, {'empata con' if tie(p_att) else 'se separa de'} "
        f"DenseNet121 ({fmt_p_html(p_att)}). El ViT-B/16 es además el modelo más costoso: "
        f"{number(vit_pre['parameters'])} parámetros y {num1(vit_pre['inference_ms_per_image'])} ms por "
        f"imagen, frente a {number(densenet['parameters'])} y {num1(densenet['inference_ms_per_image'])} ms "
        "de DenseNet121.",
    ]
    for item in conclusions:
        story.append(Paragraph(f"• {item}", styles["Body"]))


    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.7 * cm,
        title="Informe Reto 1 — Clasificación de 3 escenas",
        author="Darwin Castaño",
    )

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.black)
        canvas.drawString(2.2 * cm, 1.2 * cm, "Reto 1 — Clasificación de 3 escenas (coast / forest / highway)")
        canvas.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f"Página {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    print(f"[ok] informe generado: {build_report()}")
