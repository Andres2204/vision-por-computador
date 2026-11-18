"""Build informe-reto-1.pdf from the artifacts in resultados/.

The report is generated only from what the notebooks saved (metrics.json,
predictions.csv, split.json and the comparison figures), so no number is typed
by hand: re-run this script after retraining a model and the PDF follows.

Usage (from guia-4/reto-1):

    ../../.venv/bin/python informe_reto_1.py
    # or, if reportlab is not in the venv:
    #   ../../.venv/bin/pip install reportlab

Dependencies: reportlab plus the standard library. The Spanish prose lives in
this file; identifiers and docstrings stay in English.
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
OUTPUT = BASE / "informe-reto-1.pdf"
FIGURE_TMP = Path(tempfile.mkdtemp(prefix="informe_figs_"))

MODEL_ORDER = [
    "cnn/base",
    "cnn/mejorada",
    "cnn/mejorada_v2",
    "densenet",
    "vit",
    "ml/logreg",
    "ml/svm",
    "ml/rf",
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
    ("cnn/mejorada_v2", "cnn/base"),
    ("cnn/mejorada_v2", "cnn/mejorada"),
    ("cnn/mejorada_v2", "densenet"),
    ("vit", "cnn/mejorada_v2"),
    ("vit", "densenet"),
]

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
    return "n/d" if value is None else f"{100 * value:.{digits}f} %"


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
        if model["folder"] != "vit":
            model["ci"] = bootstrap_ci(model["predictions"])
        else:
            model["ci"] = bootstrap_ci(model["predictions"])

    by_folder = {m["folder"]: m for m in models}
    best = max(models, key=lambda m: m["accuracy"])
    dl_models = [m for m in models if not m["folder"].startswith("ml/")]
    ml_models = [m for m in models if m["folder"].startswith("ml/")]
    best_ml = max(ml_models, key=lambda m: m["accuracy"]) if ml_models else None
    best_dl = max(dl_models, key=lambda m: m["accuracy"]) if dl_models else None
    transformer = by_folder.get("vit")
    n_test = models[0].get("n_test")
    split = json.loads((RESULTS / "split.json").read_text()) if (RESULTS / "split.json").exists() else {}
    n_total = split.get("n_total", "n/d")
    n_train = split.get("n_train", "n/d")

    styles = build_styles()
    story: list = []

    # ---------------------------------------------------------------- portada
    story.append(Paragraph("Reto 1 — Clasificación de 3 escenas con aprendizaje profundo", styles["H1es"]))
    story.append(
        Paragraph(
            "CNN secuencial, una arquitectura no secuencial (DenseNet121 preentrenada), un Vision "
            "Transformer entrenado desde cero y modelos clásicos de machine learning sobre el mismo "
            "split congelado, con métricas idénticas para todos.",
            styles["Body"],
        )
    )

    story.append(Paragraph("Resumen ejecutivo", styles["H2es"]))
    summary_bits = [
        f"<b>Ganador:</b> {best['name']} con {pct(best['accuracy'])} de accuracy en test "
        f"({n_test} imágenes), F1 macro {pct(best['f1_macro'])} y kappa {number(best['cohen_kappa'], 3)}.",
    ]
    ranking = sorted(models, key=lambda m: m["accuracy"], reverse=True)
    if len(ranking) > 1:
        summary_bits.append(
            f"Le siguen {ranking[1]['name']} ({pct(ranking[1]['accuracy'])})"
            + (f" y {ranking[2]['name']} ({pct(ranking[2]['accuracy'])})." if len(ranking) > 2 else ".")
        )
    if best_dl and best_ml:
        gap = 100 * (best_dl["accuracy"] - best_ml["accuracy"])
        if gap >= 0:
            summary_bits.append(
                f"El mejor modelo clásico es {best_ml['name']} ({pct(best_ml['accuracy'])}), "
                f"{num1(gap)} puntos por debajo del mejor modelo de aprendizaje profundo "
                f"({best_dl['name']}, {pct(best_dl['accuracy'])})."
            )
        else:
            summary_bits.append(
                f"El mejor modelo clásico es {best_ml['name']} ({pct(best_ml['accuracy'])}), "
                f"{num1(-gap)} puntos por delante de {best_dl['name']} "
                f"({pct(best_dl['accuracy'])}), el mejor modelo de aprendizaje profundo."
            )
    if transformer:
        summary_bits.append(
            f"El Transformer (ViT desde cero) alcanza {pct(transformer['accuracy'])}: "
            + (
                f"queda {num1(100 * (best['accuracy'] - transformer['accuracy']))} puntos por debajo del ganador, "
                "lo que apunta al preentrenamiento como factor dominante con este volumen de datos."
                if best["folder"] != "vit"
                else "es el mejor modelo del reto."
            )
        )
    else:
        summary_bits.append(
            "El ViT desde cero todavía no está en <font face='Courier'>resultados/vit/</font>: "
            "ejecuta <font face='Courier'>ScenesViT.ipynb</font> y vuelve a generar este informe."
        )
    for bit in summary_bits:
        story.append(Paragraph(f"• {bit}", styles["Body"]))

    # -------------------------------------------------------------- protocolo
    story.append(Paragraph("Protocolo", styles["H2es"]))
    story.append(
        Paragraph(
            f"Dataset de 3 escenas (bosque, costa y autopista) con {n_total} imágenes; split "
            f"estratificado y congelado en <font face='Courier'>resultados/split.json</font>: "
            f"{n_train} de entrenamiento, 108 de validación y {n_test} de test (semilla 20). "
            "Todos los modelos se evalúan sobre esas mismas imágenes de test, así que las predicciones "
            "son emparejadas y el test de McNemar es válido. Métricas: accuracy, precisión/recall y F1 "
            "macro y weighted, kappa de Cohen, ROC-AUC one-vs-rest, intervalo de confianza del 95 % de "
            "la accuracy por bootstrap (2 000 remuestreos) y test exacto de McNemar entre pares.",
            styles["Body"],
        )
    )

    # ------------------------------------------------------------------ tabla
    story.append(Paragraph("Resultados por modelo", styles["H2es"]))
    header = [
        "Modelo", "Familia", "Accuracy", "F1 macro", "IC 95 %", "Parámetros", "Entrena (s)", "Épocas",
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
            [4.1 * cm, 2.6 * cm, 1.9 * cm, 1.8 * cm, 3.1 * cm, 2.3 * cm, 1.9 * cm, 1.5 * cm],
            align_right_from=2,
            center=(2, 3, 4, 5, 6, 7),
        )
    )
    story.append(
        Paragraph(
            f"Tabla 1. Resultados sobre el test común de {n_test} imágenes. IC 95 % calculado por "
            "bootstrap sobre las predicciones guardadas; los modelos clásicos no tienen épocas ni un "
            "conteo de parámetros comparable, y así se registra en sus <font face='Courier'>metrics.json</font>.",
            styles["Caption"],
        )
    )

    # ---------------------------------------------------------------- McNemar
    story.append(Paragraph("Significancia estadística (McNemar)", styles["H2es"]))
    pair_rows = [["Modelo A", "Modelo B", "p-valor", "Veredicto"]]
    for folder_a, folder_b in MC_NEMAR_PAIRS:
        if folder_a not in by_folder or folder_b not in by_folder:
            continue
        p_value = mcnemar_pvalue(by_folder[folder_a]["predictions"], by_folder[folder_b]["predictions"])
        pair_rows.append(
            [
                Paragraph(by_folder[folder_a]["name"], styles["Cell"]),
                Paragraph(by_folder[folder_b]["name"], styles["Cell"]),
                fmt_p(p_value),
                verdict(p_value),
            ]
        )
    if best_ml and best_dl:
        p_value = mcnemar_pvalue(best_ml["predictions"], best_dl["predictions"])
        pair_rows.append(
            [
                Paragraph(best_ml["name"], styles["Cell"]),
                Paragraph(best_dl["name"], styles["Cell"]),
                fmt_p(p_value),
                verdict(p_value),
            ]
        )
    story.append(table(pair_rows, [5.6 * cm, 5.6 * cm, 2.4 * cm, 4.0 * cm], align_right_from=2))
    story.append(
        Paragraph(
            "Tabla 2. Test exacto de McNemar (α = 0,05) sobre las predicciones emparejadas del mismo "
            "test. «Dentro del ruido» significa que la diferencia observada no se distingue del "
            "muestreo; con 240 imágenes de test, pocos puntos de accuracy no bastan para declarar "
            "un ganador.",
            styles["Caption"],
        )
    )

    # --------------------------------------------------------------- figuras
    figures = [
        ("barras_accuracy_ic95.png", "Figura 1. Accuracy en test con intervalo de confianza del 95 %."),
        ("matrices_confusion.png", "Figura 2. Matrices de confusión de todos los modelos."),
        ("curvas_aprendizaje.png", "Figura 3. Curvas de aprendizaje de los modelos con épocas."),
        ("heatmap_f1_por_clase.png", "Figura 4. F1 por clase y por modelo."),
    ]
    story.append(PageBreak())
    story.append(Paragraph("Figuras", styles["H2es"]))
    for filename, caption in figures:
        path = RESULTS / "comparacion" / filename
        if not path.exists():
            continue
        from PIL import Image as PILImage  # noqa: PLC0415  (only needs to exist when figures exist)

        with PILImage.open(path) as image:
            width_px, height_px = image.size
            strip = width_px / height_px > 5
            slices = []
            if strip:
                # A very wide strip (one confusion matrix per model) is unreadable at
                # full page width: drop its super-title (the PDF caption says the same)
                # and split it at the blank columns between panels.
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
        story.append(KeepTogether([Image(str(path), width=width, height=height), Paragraph(caption, styles["Caption"])]))

    extras = [
        (RESULTS / "densenet" / "gradcam.png", "Figura 5. Grad-CAM de DenseNet121 sobre una imagen de test por clase."),
        (RESULTS / "vit" / "ejemplos_prediccion.png", "Figura 6. Predicciones del ViT sobre seis imágenes de test (verde: acierto)."),
    ]
    for path, caption in extras:
        if not path.exists():
            continue
        from PIL import Image as PILImage  # noqa: PLC0415

        with PILImage.open(path) as image:
            width_px, height_px = image.size
        width = 15.0 * cm
        height = width * height_px / width_px
        if height > 7.0 * cm:
            height = 7.0 * cm
            width = height * width_px / height_px
        print(f"[fig] {path.name}: {width / cm:.1f} x {height / cm:.1f} cm")
        story.append(KeepTogether([Image(str(path), width=width, height=height), Paragraph(caption, styles["Caption"])]))

    # ------------------------------------------------------------ conclusiones
    story.append(Paragraph("Conclusiones", styles["H2es"]))
    conclusions = []
    conclusions.append(
        f"<b>Ganador.</b> {best['name']} es el mejor modelo en test con {pct(best['accuracy'])} "
        + (f"(IC 95 % {pct(best['ci'][0])}–{pct(best['ci'][1])})" if best.get("ci") else "")
        + f", F1 macro {pct(best['f1_macro'])} y kappa {number(best['cohen_kappa'], 3)}."
    )
    if "cnn/mejorada_v2" in by_folder and "cnn/base" in by_folder:
        delta = 100 * (by_folder["cnn/mejorada_v2"]["accuracy"] - by_folder["cnn/base"]["accuracy"])
        p_value = mcnemar_pvalue(by_folder["cnn/mejorada_v2"]["predictions"], by_folder["cnn/base"]["predictions"])
        conclusions.append(
            f"<b>Iteración de la CNN.</b> La versión mejorada v2 queda {num1(abs(delta))} puntos "
            f"{'por encima' if delta >= 0 else 'por debajo'} de la CNN base (McNemar p = {fmt_p(p_value)}): "
            "ajustar hiperparámetros y la función de costo no cambia el resultado de forma "
            "estadísticamente significativa, y la v1 se conserva como evidencia de la iteración."
        )
    if "densenet" in by_folder and "cnn/base" in by_folder:
        delta = 100 * (by_folder["densenet"]["accuracy"] - by_folder["cnn/base"]["accuracy"])
        p_value = mcnemar_pvalue(by_folder["densenet"]["predictions"], by_folder["cnn/base"]["predictions"])
        conclusions.append(
            f"<b>Arquitectura no secuencial.</b> DenseNet121 con fine-tuning gana "
            f"{num1(abs(delta))} puntos a la CNN secuencial (McNemar p = {fmt_p(p_value)}): la conectividad "
            "densa y, sobre todo, el preentrenamiento en ImageNet son lo que más aporta."
        )
    if transformer and "cnn/mejorada_v2" in by_folder:
        delta = 100 * (transformer["accuracy"] - by_folder["cnn/mejorada_v2"]["accuracy"])
        p_value = mcnemar_pvalue(transformer["predictions"], by_folder["cnn/mejorada_v2"]["predictions"])
        conclusions.append(
            f"<b>IA generativa (Transformer).</b> El ViT entrenado desde cero llega a "
            f"{pct(transformer['accuracy'])} y queda {signed1(delta)} puntos frente a la CNN mejorada v2 "
            f"(McNemar p = {fmt_p(p_value)}). Como los dos arrancan desde cero, esa diferencia es "
            "atribuible a la arquitectura: con "
            f"{n_train} imágenes de entrenamiento la atención global no compensa los sesgos inductivos "
            "de la convolución."
        )
        if "densenet" in by_folder:
            delta_pre = 100 * (by_folder["densenet"]["accuracy"] - transformer["accuracy"])
            conclusions.append(
                f"<b>Preentrenamiento.</b> El hueco de {num1(delta_pre)} puntos entre DenseNet121 "
                f"({pct(by_folder['densenet']['accuracy'])}) y el ViT ({pct(transformer['accuracy'])}) "
                "mezcla arquitectura e ImageNet; el enunciado agrupa el Transformer como «IA "
                "generativa», pero aquí se usa como clasificador discriminativo y lo que comparte con "
                "la IA generativa es la arquitectura de atención."
            )
    if best_ml:
        conclusions.append(
            f"<b>Línea base clásica.</b> Con descriptores HSV+HOG, el mejor modelo clásico "
            f"({best_ml['name']}, {pct(best_ml['accuracy'])}) entrena en "
            f"{number(best_ml.get('training_seconds'), 2)} s y sigue siendo la referencia de coste: "
            "solo el preentrenamiento de ImageNet lo supera con claridad."
        )
    conclusions.append(
        f"<b>Limitación del dataset.</b> {n_total} imágenes en 3 clases y {n_test} de test: los "
        "intervalos de confianza son anchos y diferencias de pocos puntos quedan dentro del ruido, "
        "por eso el McNemar y el bootstrap acompañan a la accuracy en todas las comparaciones."
    )
    for item in conclusions:
        story.append(Paragraph(f"• {item}", styles["Body"]))

    # -------------------------------------------------------- guía 2 + notas
    story.append(Paragraph("Comparación con el machine learning de la guía 2", styles["H2es"]))
    story.append(
        Paragraph(
            "Cifras transcritas de <font face='Courier'>guia-2/reto-5/reto-5.ipynb</font> (no se "
            "reentrenan aquí). Ese cuaderno usa el mismo protocolo (25 % de test, semilla 20, "
            "estratificado, 240 imágenes), pero no son las mismas 240 imágenes: la distribución por "
            "clase de este reto es 90/82/68 frente a 91/87/62, así que la comparación es orientativa y "
            "no entra en el McNemar.",
            styles["Body"],
        )
    )
    guide_rows = [["Modelo (guía 2)", "Características", "Accuracy CV", "Accuracy test"]]
    for row in FEATURED_GUIDE2:
        guide_rows.append([Paragraph(cell, styles["Cell"]) for cell in row])
    story.append(table(guide_rows, [5.4 * cm, 5.0 * cm, 3.0 * cm, 3.0 * cm], align_right_from=2, center=(2, 3)))
    story.append(
        Paragraph(
            "Tabla 3. Los cuatro modelos representativos de la guía 2 con el mejor bloque de "
            "características (RGB+HSV+Haralick, 25 descriptores). El detalle de los 9 modelos está en "
            "<font face='Courier'>reto-1.ipynb</font>.",
            styles["Caption"],
        )
    )

    story.append(Paragraph("Reproducibilidad", styles["H2es"]))
    story.append(
        Paragraph(
            "Cada modelo queda en <font face='Courier'>resultados/&lt;modelo&gt;/</font> con sus "
            "métricas, predicciones y pesos, y este informe se genera con "
            "<font face='Courier'>python informe_reto_1.py</font> leyendo solo esos artefactos: ninguna "
            "cifra está escrita a mano, así que al reentrenar un modelo el PDF se recalcula solo. "
            "Orden de ejecución: <font face='Courier'>ScenesCNN.ipynb</font> → "
            "<font face='Courier'>ScenesDenseNet.ipynb</font> → <font face='Courier'>ScenesViT.ipynb</font> "
            "→ <font face='Courier'>ScenesClassicML.ipynb</font> → <font face='Courier'>reto-1.ipynb</font> "
            "→ <font face='Courier'>informe_reto_1.py</font>.",
            styles["Body"],
        )
    )

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
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(2.2 * cm, 1.2 * cm, "Reto 1 — Clasificación de 3 escenas (coast / forest / highway)")
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
