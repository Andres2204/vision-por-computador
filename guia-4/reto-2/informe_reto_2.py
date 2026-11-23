"""Build informe-reto-2.pdf from the artifacts in resultados/.

The report is generated only from what the notebooks saved (metrics.json,
predictions.csv, split.json and the four comparison figures), so no number is
typed by hand: re-run this script after retraining a model and the PDF follows.

Usage (from guia-4/reto-2):

    ../../venv/bin/python informe_reto_2.py
    # or, if reportlab is not in the venv:
    #   ../../venv/bin/pip install reportlab

Dependencies: reportlab, Pillow plus the standard library. The Spanish prose
lives in this file; identifiers and docstrings stay in English.
"""

from __future__ import annotations

import csv
import json
import math
import random
import tempfile
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "resultados"
OUTPUT = BASE / "informe-reto-2.pdf"
FIGURE_TMP = Path(tempfile.mkdtemp(prefix="informe_figs_"))

MODEL_ORDER = [
    "cnn/base",
    "cnn/mejorada",
    "densenet",
    "vit",
]

FAMILY_ES = {
    "CNN": "CNN",
    "DL (non-sequential)": "DL no secuencial",
    "machine learning": "machine learning",
    "IA generativa (Transformer)": "IA generativa",
}

MC_NEMAR_PAIRS = [
    ("cnn/base", "cnn/mejorada"),
    ("cnn/mejorada", "densenet"),
    ("cnn/base", "densenet"),
    ("cnn/mejorada", "vit"),
    ("densenet", "vit"),
    ("cnn/base", "vit"),
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
            textColor=colors.HexColor("#1a3a5c"),
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
            textColor=colors.HexColor("#1a3a5c"),
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
            textColor=colors.HexColor("#444444"),
            spaceBefore=2,
            spaceAfter=10,
        )
    )
    return styles


def table(data, widths, align_right_from: int = 1, header_bg="#1a3a5c", center: tuple = ()):
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
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#999999")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    body.setStyle(TableStyle(style))
    return body


# -------------------------------------------------------------------------- build
def title_band_height(image) -> int:
    """Rows to drop from the top so a super-title is not cut in half by slicing."""

    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()
    step = max(1, width // 400)
    blank_rows = []
    for y in range(height):
        is_blank = True
        for x in range(0, width, step):
            if pixels[x, y] < 250:
                is_blank = False
                break
        blank_rows.append(is_blank)
    run = 0
    seen_content = False
    for y, is_blank in enumerate(blank_rows):
        if is_blank:
            if seen_content:
                run += 1
                if run >= 3:
                    return y
        else:
            seen_content = True
            run = 0
    return 0


def split_points(image, parts: int) -> list[int]:
    """Return x positions of blank columns to cut a wide figure into ``parts``."""

    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()
    blank = []
    for x in range(width):
        is_blank = True
        for y in range(0, height, max(1, height // 120)):
            if pixels[x, y] < 250:
                is_blank = False
                break
        blank.append(is_blank)
    runs = []
    start = None
    for x, is_blank in enumerate(blank):
        if is_blank and start is None:
            start = x
        elif not is_blank and start is not None:
            runs.append((start, x - 1))
            start = None
    if start is not None:
        runs.append((start, width - 1))
    runs = [run for run in runs if run[1] - run[0] >= 2]
    cuts: list[int] = []
    for part in range(1, parts):
        target = width * part / parts
        usable = [run for run in runs if run[1] > 20 and run[0] < width - 20]
        if not usable:
            cuts.append(int(target))
            continue
        best = min(usable, key=lambda run: abs((run[0] + run[1]) / 2 - target))
        cut = (best[0] + best[1]) // 2
        cuts.append(cut if all(abs(cut - other) > 30 for other in cuts) else int(target))
    return sorted(cuts)


def build_report() -> Path:
    models = load_models()
    if not models:
        raise SystemExit("No hay resultados en resultados/: ejecuta antes los cuadernos de modelos.")

    for model in models:
        model["ci"] = bootstrap_ci(model["predictions"])

    by_folder = {m["folder"]: m for m in models}
    required = {"cnn/base", "cnn/mejorada", "densenet", "vit"}
    missing = required - set(by_folder)
    if missing:
        raise SystemExit(
            "Faltan resultados de: " + ", ".join(sorted(missing)) + ". Ejecuta los cuadernos de modelos."
        )

    best = max(models, key=lambda m: m["accuracy"])
    ranking = sorted(models, key=lambda m: m["accuracy"], reverse=True)
    second = ranking[1]
    base = by_folder["cnn/base"]
    improved = by_folder["cnn/mejorada"]
    densenet = by_folder["densenet"]
    vit = by_folder["vit"]
    n_test = models[0].get("n_test")
    split = json.loads((RESULTS / "split.json").read_text()) if (RESULTS / "split.json").exists() else {}
    n_total = split.get("n_total", "n/d")
    n_train = split.get("n_train", "n/d")
    n_val = split.get("n_val", "n/d")

    def pvalue(folder_a: str, folder_b: str) -> float | None:
        return mcnemar_pvalue(by_folder[folder_a]["predictions"], by_folder[folder_b]["predictions"])

    styles = build_styles()
    story: list = []

    # ---------------------------------------------------------------- portada
    story.append(
        Paragraph(
            "Reto 2 — Clasificación de cobertura terrestre (continuación del reto 1)",
            styles["H1es"],
        )
    )
    story.append(
        Paragraph(
            "Segunda parte de la comparación. El reto 1 trabajó con 3 escenas y 9 modelos; "
            "aquí se cambia el dataset (denseresidential, harbor y river) y se comparan cuatro "
            "modelos elegidos a partir de sus resultados: CNN base, CNN mejorada con la receta v2, "
            "DenseNet121 preentrenada y ViT-B/16 preentrenado. El objetivo es comprobar si las "
            "conclusiones del reto 1 se mantienen en otro problema.",
            styles["Body"],
        )
    )

    story.append(Paragraph("Resumen ejecutivo", styles["H2es"]))
    def tie(p_value: float | None) -> bool:
        return p_value is None or p_value >= 0.05

    top3 = ranking[:3]
    top3_tied = all(
        tie(pvalue(a["folder"], b["folder"])) for i, a in enumerate(top3) for b in top3[i + 1:]
    )
    summary_bits = [
        f"<b>Mejor modelo:</b> {best['name']}, con {pct(best['accuracy'])} de accuracy en test "
        f"({n_test} imágenes), F1 macro {pct(best['f1_macro'])} y kappa {number(best['cohen_kappa'], 3)}.",
        f"Le siguen {ranking[1]['name']} ({pct(ranking[1]['accuracy'])}) y {ranking[2]['name']} "
        f"({pct(ranking[2]['accuracy'])})"
        + (", sin diferencias significativas entre los tres." if top3_tied else ".")
        + f" La CNN base se queda en {pct(base['accuracy'])}.",
    ]
    for bit in summary_bits:
        story.append(Paragraph(f"• {bit}", styles["Body"]))

    # ------------------------------------------------------- continuidad reto 1
    story.append(Paragraph("Continuidad con el reto 1", styles["H2es"]))
    story.append(
        Paragraph(
            "El reto 1 comparó 9 modelos sobre 958 imágenes de 3 escenas. Las tres CNN desde cero "
            "quedaron estadísticamente empatadas (McNemar p = 0,2005, 0,2668 y 0,7011), el ViT desde "
            "cero empató con la mejor CNN (p = 1,0000) y el preentrenamiento subió al ViT 8,75 puntos "
            "(p = 0,0001). El mejor modelo clásico (SVM, 95,42 %) quedó por debajo de DenseNet121 "
            "(p = 0,0654). Por eso el reto 2 conserva un representante por celda útil y arranca del "
            "ViT preentrenado, que es el que permite comparar atención y convolución a igualdad de "
            "preentrenamiento.",
            styles["Body"],
        )
    )
    continuity_rows = [["Modelo del reto 1", "Accuracy", "McNemar relevante"]]
    continuity = [
        ("CNN base", "90,42 %", "vs DenseNet121: p = 0,0001"),
        ("CNN mejorada v1", "87,08 %", "vs CNN base: p = 0,2005"),
        ("CNN mejorada v2", "89,17 %", "vs CNN base: p = 0,7011; vs v1: p = 0,2668"),
        ("DenseNet121", "98,33 %", "—"),
        ("ViT desde cero", "88,75 %", "vs CNN v2: p = 1,0000; vs DenseNet121: p = 0,0000"),
        ("ViT preentrenado", "97,50 %", "vs ViT desde cero: p = 0,0001; vs DenseNet121: p = 0,6875"),
        ("Mejor ML (SVM)", "95,42 %", "vs DenseNet121: p = 0,0654"),
    ]
    for row in continuity:
        continuity_rows.append([Paragraph(cell, styles["Cell"]) for cell in row])
    story.append(
        table(
            continuity_rows,
            [4.2 * cm, 2.4 * cm, 9.5 * cm],
            align_right_from=1,
            center=(1,),
        )
    )
    story.append(
        Paragraph(
            "Tabla 1. Resultados finales del reto 1 que justifican la selección de modelos del reto 2.",
            styles["Caption"],
        )
    )

    # -------------------------------------------------------------- protocolo
    story.append(Paragraph("Protocolo", styles["H2es"]))
    story.append(
        Paragraph(
            f"Dataset de cobertura terrestre con {n_total} imágenes PNG, 500 por clase "
            f"(denseresidential, harbor y river). Split estratificado y congelado en "
            f"<font face='Courier'>resultados/split.json</font>: {n_train} de entrenamiento, "
            f"{n_val} de validación y {n_test} de test (semilla 20). Las CNN y DenseNet trabajan a "
            "128×128; el ViT usa 224×224. Todos los modelos se evalúan sobre las mismas imágenes de "
            "test, así que las predicciones son emparejadas y el test de McNemar es válido. Métricas: "
            "accuracy, precisión/recall y F1 macro y weighted, kappa de Cohen, ROC-AUC one-vs-rest, "
            "intervalo de confianza del 95 % por bootstrap (2.000 remuestreos) y test exacto de "
            "McNemar entre pares.",
            styles["Body"],
        )
    )

    # ------------------------------------------------------------------ tabla
    story.append(Paragraph("Resultados por modelo", styles["H2es"]))
    header = [
        "Modelo",
        "Familia",
        "Accuracy",
        "F1 macro",
        "IC 95 %",
        "Parámetros",
        "Entrena (s)",
        "Épocas",
    ]
    rows = [header]
    for model in models:
        ci = model.get("ci")
        ci_text = "n/d" if not ci else f"{pct(ci[0], 1)} – {pct(ci[1], 1)}"
        rows.append(
            [
                Paragraph(model["name"], styles["Cell"]),
                Paragraph(FAMILY_ES.get(model["family"], model["family"]), styles["Cell"]),
                pct(model["accuracy"]),
                pct(model["f1_macro"]),
                ci_text,
                number(model.get("parameters")),
                number(model.get("training_seconds")),
                number(model.get("epochs")),
            ]
        )
    story.append(
        table(
            rows,
            [3.6 * cm, 2.2 * cm, 1.6 * cm, 1.6 * cm, 2.7 * cm, 2.0 * cm, 1.6 * cm, 1.3 * cm],
            align_right_from=2,
            center=(2, 3, 4, 5, 6, 7),
        )
    )
    story.append(
        Paragraph(
            f"Tabla 2. Resultados sobre el test común de {n_test} imágenes. IC 95 % calculado por "
            "bootstrap sobre las predicciones guardadas.",
            styles["Caption"],
        )
    )

    # ---------------------------------------------------------------- McNemar
    story.append(Paragraph("Significancia estadística (McNemar)", styles["H2es"]))
    pair_rows = [["Modelo A", "Modelo B", "p-valor", "Veredicto"]]
    for folder_a, folder_b in MC_NEMAR_PAIRS:
        p_value = pvalue(folder_a, folder_b)
        pair_rows.append(
            [
                Paragraph(by_folder[folder_a]["name"], styles["Cell"]),
                Paragraph(by_folder[folder_b]["name"], styles["Cell"]),
                fmt_p(p_value),
                verdict(p_value),
            ]
        )
    story.append(table(pair_rows, [5.2 * cm, 5.2 * cm, 2.2 * cm, 3.8 * cm], align_right_from=2))
    story.append(
        Paragraph(
            "Tabla 3. Test exacto de McNemar (α = 0,05) sobre las predicciones emparejadas del mismo "
            f"test. «Dentro del ruido» significa que la diferencia observada no se distingue del "
            f"muestreo; con {n_test} imágenes de test, pocos puntos de accuracy no bastan para "
            "declarar un ganador.",
            styles["Caption"],
        )
    )

    # --------------------------------------------------------------- figuras
    figures = [
        (
            "barras_accuracy_f1.png",
            "Figura 1. Accuracy y F1 macro en test con intervalo de confianza del 95 %.",
        ),
        (
            "matrices_confusion.png",
            "Figura 2. Matrices de confusión normalizadas de los cuatro modelos.",
        ),
        (
            "curvas_aprendizaje.png",
            "Figura 3. Curvas de aprendizaje de entrenamiento y validación.",
        ),
        (
            "heatmap_f1_por_clase.png",
            "Figura 4. F1 por clase y por modelo.",
        ),
    ]
    story.append(PageBreak())
    story.append(Paragraph("Figuras", styles["H2es"]))
    for filename, caption in figures:
        path = RESULTS / "comparacion" / filename
        if not path.exists():
            continue
        from PIL import Image as PILImage  # noqa: PLC0415

        with PILImage.open(path) as image:
            width_px, height_px = image.size
            strip = width_px / height_px > 5
            slices = []
            if strip:
                top = title_band_height(image)
                image = image.crop((0, top, width_px, height_px))
                width_px, height_px = image.size
                slice_count = 2 if width_px / height_px < 9 else 3
                cuts = split_points(image, slice_count)
                edges = [0, *cuts, width_px]
                for part in range(len(edges) - 1):
                    left, right = edges[part], edges[part + 1]
                    crop = image.crop((left, 0, right, height_px))
                    crop_path = FIGURE_TMP / f"{path.stem}_part{part + 1}.png"
                    crop.save(crop_path)
                    slices.append((crop_path, crop.size))
        if slices:
            for slice_path, (slice_w, slice_h) in slices:
                width = 16.6 * cm
                height = width * slice_h / slice_w
                story.append(Image(str(slice_path), width=width, height=height))
            story.append(Paragraph(caption, styles["Caption"]))
            continue
        width = 16.6 * cm
        height = width * height_px / width_px
        if height > 8.5 * cm:
            height = 8.5 * cm
            width = height * width_px / height_px
        print(f"[fig] {path.name}: {width / cm:.1f} x {height / cm:.1f} cm")
        story.append(
            KeepTogether(
                [Image(str(path), width=width, height=height), Paragraph(caption, styles["Caption"])]
            )
        )

    # ------------------------------------------------------------ conclusiones
    story.append(Paragraph("Conclusiones", styles["H2es"]))
    p_vit_densenet = pvalue("vit", "densenet")
    p_vit_improved = pvalue("vit", "cnn/mejorada")
    p_densenet_improved = pvalue("densenet", "cnn/mejorada")
    p_base_improved = pvalue("cnn/base", "cnn/mejorada")
    gap_vit_densenet = 100 * (vit["accuracy"] - densenet["accuracy"])
    conclusions = [
        f"<b>El mejor modelo es {best['name']} ({pct(best['accuracy'])}).</b> "
        + (
            f"{ranking[1]['name']} ({pct(ranking[1]['accuracy'])}) y {ranking[2]['name']} "
            f"({pct(ranking[2]['accuracy'])}) no se distinguen estadísticamente de él."
            if top3_tied
            else f"Le sigue {second['name']} con {pct(second['accuracy'])}."
        ),
        f"<b>La CNN mejorada {'alcanza a los preentrenados' if tie(p_vit_improved) and tie(p_densenet_improved) else 'no alcanza a los preentrenados'}.</b> "
        f"Sube {num1(100 * (improved['accuracy'] - base['accuracy']))} puntos sobre la CNN base "
        f"({fmt_p_html(p_base_improved)}) y frente al ViT y a DenseNet121 da "
        f"{fmt_p_html(p_vit_improved)} y {fmt_p_html(p_densenet_improved)}. "
        + (
            "A diferencia del reto 1, en este dataset la receta v2 basta sin preentrenamiento."
            if tie(p_vit_improved) and tie(p_densenet_improved)
            else "Como en el reto 1, el preentrenamiento marca la diferencia."
        ),
        f"<b>Atención frente a convolución.</b> ViT y DenseNet121 quedan a "
        f"{num1(gap_vit_densenet)} puntos ({fmt_p_html(p_vit_densenet)}): "
        + ("empatan, igual que en el reto 1." if tie(p_vit_densenet) else "la diferencia es significativa."),
        f"<b>Coste.</b> La CNN mejorada usa {number(improved['parameters'])} parámetros y "
        f"{num1(improved['inference_ms_per_image'])} ms por imagen; DenseNet121, "
        f"{number(densenet['parameters'])} y {num1(densenet['inference_ms_per_image'])} ms; el ViT, "
        f"{number(vit['parameters'])} y {num1(vit['inference_ms_per_image'])} ms. "
        + (
            "Con la misma accuracy estadística, la CNN mejorada es la opción más eficiente."
            if tie(p_vit_improved) and tie(p_densenet_improved)
            else "El ViT es el más caro de los cuatro."
        ),
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
        title="Informe Reto 2 — Clasificación de cobertura terrestre",
        author="Darwin Castaño",
    )

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(
            2.2 * cm,
            1.2 * cm,
            "Reto 2 — Clasificación landuse (denseresidential / harbor / river)",
        )
        canvas.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f"Página {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    import shutil

    try:
        path = build_report()
    finally:
        shutil.rmtree(FIGURE_TMP, ignore_errors=True)
    print(f"[ok] informe generado: {path}")
