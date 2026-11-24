"""Build informe-reto-4.pdf from the artifacts in resultados/.

Usage (from guia-4/reto-4):

    ../../.venv/bin/python informe_reto_4.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Table, TableStyle

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "resultados"
OUTPUT = BASE / "informe-reto-4.pdf"

VERSIONS = ["Ruidosa", "SwinIR σ=25", "SwinIR σ=50", "CDAE", "CDAE + SwinIR σ=15", "CDAE + SwinIR Real-SR"]
IMAGES = [(1, "5120_512"), (59, "5632_4608"), (67, "5632_8704")]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def signed(value: float, digits: int = 2) -> str:
    return ("+" if value >= 0 else "−") + num(abs(value), digits)


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5,
                              leading=13.5, alignment=TA_JUSTIFY, spaceAfter=6))
    styles.add(ParagraphStyle(name="Cell", parent=styles["BodyText"], fontName="Helvetica", fontSize=7, leading=8.5))
    styles.add(ParagraphStyle(name="H1es", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=15,
                              leading=18, spaceBefore=2, spaceAfter=8, textColor=colors.black))
    styles.add(ParagraphStyle(name="H2es", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
                              leading=14, spaceBefore=10, spaceAfter=5, textColor=colors.black))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=7.5,
                              leading=9.5, alignment=1, textColor=colors.black, spaceBefore=2, spaceAfter=10))
    return styles


def table(data, widths):
    body = Table(data, colWidths=widths, repeatRows=1)
    body.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return body


def figure(path: Path, caption: str, styles, max_height: float):
    with PILImage.open(path) as image:
        w, h = image.size
    width = 16.6 * cm
    height = width * h / w
    if height > max_height:
        height = max_height
        width = height * w / h
    return KeepTogether([Image(str(path), width=width, height=height), Paragraph(caption, styles["Caption"])])


def build_report() -> Path:
    summary = {row["Versión"]: {k: float(v) for k, v in row.items() if k != "Versión"}
               for row in read_csv(RESULTS / "resumen.csv")}
    per_image = read_csv(RESULTS / "metricas_por_imagen.csv")
    s = summary
    noisy, cdae, dn15, realsr, dn25, dn50 = (s["Ruidosa"], s["CDAE"], s["CDAE + SwinIR σ=15"],
                                             s["CDAE + SwinIR Real-SR"], s["SwinIR σ=25"], s["SwinIR σ=50"])
    best = max(VERSIONS, key=lambda v: s[v]["PSNR"])

    styles = build_styles()
    story: list = []

    story.append(Paragraph("Reto 4 — Restauración de imágenes SAR con SwinIR", styles["H1es"]))
    story.append(Paragraph(
        "Aplicamos SwinIR preentrenado, sin entrenar ni reentrenar, a imágenes SAR con speckle y a las mismas "
        "imágenes filtradas con el CDAE del reto 3. Siguiendo el artículo <i>Analysis of the effects of applying "
        "a restoration process to despeckled SAR imagery</i>, revisamos si SwinIR recupera el detalle que pierde "
        "el filtro, con métricas frente al ground truth y con inspección visual.", styles["Body"]))

    story.append(Paragraph("Resumen", styles["H2es"]))
    for bit in [
        f"<b>Mejor resultado en métricas:</b> {best}, con {num(s[best]['PSNR'])} dB de PSNR y SSIM "
        f"{num(s[best]['SSIM'], 3)}. Ninguna versión con SwinIR lo supera.",
        f"SwinIR sobre la imagen ruidosa la mejora (σ=50: {signed(dn50['PSNR'] - noisy['PSNR'])} dB), "
        "pero deja speckle con σ=25 e inventa texturas con σ=50.",
        f"Sobre la imagen filtrada, el denoising suaviza más (SSIM {num(dn15['SSIM'], 3)}) y el Real-SR "
        f"devuelve nitidez (ratio de gradientes {num(cdae["GRAD_RATIO"])} a {num(realsr['GRAD_RATIO'])}) "
        "pero con detalle inventado y artefactos de cuadrícula.",
    ]:
        story.append(Paragraph(f"• {bit}", styles["Body"]))

    story.append(Paragraph("Protocolo", styles["H2es"]))
    story.append(Paragraph(
        "<b>Imágenes.</b> Tres imágenes de 512×512 de la validación oficial del <i>SAR despeckling filters "
        "dataset</i> (índices 1, 59 y 67, las mismas del análisis visual del reto 3), con zonas homogéneas, "
        "cultivos, carreteras y zona urbana. El speckle es real (Sentinel-1) y el ground truth es un promedio "
        "multitemporal.", styles["Body"]))
    story.append(Paragraph(
        "<b>Modelos.</b> SwinIR-M con los pesos oficiales de <i>grayscale denoising</i> (σ = 15, 25 y 50, "
        "entrenados con ruido gaussiano) y de <i>real-world super-resolution</i> ×2 con GAN (entrenado con "
        "desenfoque, ruido y JPEG). La salida ×2 se reduce a 512×512 con <font face='Courier'>INTER_AREA</font> "
        "para compararla con el GT. El CDAE es el modelo ajustado del reto 3, usado solo para predecir.",
        styles["Body"]))
    story.append(Paragraph(
        "<b>Métricas</b> (las del reto 3, por imagen y frente al GT): PSNR y SSIM; ENL = μ²/σ² en las 3 ventanas "
        "de 20×20 más homogéneas del GT, reportado como ENL / ENL<sub>GT</sub> (ideal 1; mayor indica "
        "sobresuavizado) con la mediana; ratio de gradientes Sobel (menor que 1, bordes perdidos; mayor que 1, "
        "ruido o bordes de más) y correlación de bordes con el GT.", styles["Body"]))

    story.append(Paragraph("Resultados", styles["H2es"]))
    rows = [["Versión", "PSNR (dB)", "SSIM", "ENL / ENL_GT (ideal 1)", "Ratio gradientes (ideal 1)", "Corr. bordes"]]
    for v in VERSIONS:
        m = s[v]
        rows.append([Paragraph(f"<b>{v}</b>" if v == best else v, styles["Cell"]), num(m["PSNR"]),
                     num(m["SSIM"], 3), num(m["ENL_RATIO"]), num(m["GRAD_RATIO"]), num(m["GRAD_CORR"], 3)])
    story.append(table(rows, [4.4 * cm, 2.0 * cm, 1.7 * cm, 2.9 * cm, 3.0 * cm, 2.4 * cm]))
    story.append(Paragraph("Tabla 1. Media de las 3 imágenes (ENL / ENL_GT con mediana).", styles["Caption"]))

    rows = [["Versión"] + [f"Imagen {i}" for i, _ in IMAGES]]
    for v in VERSIONS:
        values = {r["Imagen"]: float(r["PSNR"]) for r in per_image if r["Versión"] == v}
        rows.append([Paragraph(v, styles["Cell"])] + [num(values[name]) for _, name in IMAGES])
    story.append(table(rows, [4.4 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm]))
    story.append(Paragraph("Tabla 2. PSNR (dB) por imagen.", styles["Caption"]))

    story.append(figure(RESULTS / "metricas.png",
                        "Figura 1. PSNR, SSIM, ENL / ENL_GT (escala log) y ratio de gradientes. Las líneas "
                        "discontinuas marcan el valor ideal.", styles, 5 * cm))
    for n, (i, name) in enumerate(IMAGES, start=2):
        story.append(figure(RESULTS / f"comparacion_{i}.png",
                            f"Figura {n}. Imagen {i} ({name}): imagen completa, zoom de la zona con más bordes "
                            "(recuadro amarillo), |error| frente al GT y bordes Sobel.", styles, 11.5 * cm))

    story.append(Paragraph("Observaciones visuales", styles["H2es"]))
    for bit in [
        "<b>SwinIR σ=25</b> quita poco: el grano del speckle sigue en las zonas homogéneas.",
        "<b>SwinIR σ=50</b> limpia el speckle y deja rectas las calles y los bloques urbanos, pero en cultivos "
        "y vegetación aparecen trazos curvos y manchas lisas que no están en el GT.",
        "<b>CDAE</b> es el más fiel en intensidad pero borroso: pierde las líneas de las parcelas y la "
        "separación entre edificios, y en la zona saturada de la imagen 59 baja el blanco a gris.",
        "<b>CDAE + SwinIR σ=15</b> queda igual o más borroso que el CDAE.",
        "<b>CDAE + SwinIR Real-SR</b> recupera contornos de edificios, la carretera de la imagen 67 y el grano "
        "del terreno, pero con textura en bloques y un patrón de cuadrícula en la zona saturada de la imagen 59.",
    ]:
        story.append(Paragraph(f"• {bit}", styles["Body"]))

    story.append(Paragraph("Conclusiones", styles["H2es"]))
    for bit in [
        f"<b>El CDAE sigue siendo el mejor en métricas</b> ({num(cdae['PSNR'])} dB, SSIM {num(cdae['SSIM'], 3)}).",
        f"<b>El denoising de SwinIR no está hecho para el speckle</b>, que es multiplicativo: con σ=25 el ENL "
        f"queda en {num(dn25['ENL_RATIO'])} veces el del GT (speckle residual) y con σ=50 en "
        f"{num(dn50['ENL_RATIO'])} veces (sobresuavizado y texturas inventadas).",
        f"<b>Aplicar denoising después del filtro empeora</b>: el SSIM baja de {num(cdae['SSIM'], 3)} a "
        f"{num(dn15['SSIM'], 3)} y el ratio de gradientes de {num(cdae['GRAD_RATIO'])} a "
        f"{num(dn15['GRAD_RATIO'])}.",
        f"<b>El Real-SR devuelve nitidez pero no fidelidad</b>: el ratio de gradientes sube a "
        f"{num(realsr['GRAD_RATIO'])} y el ENL queda en {num(realsr['ENL_RATIO'])}, pero el PSNR baja "
        f"{num(cdae['PSNR'] - realsr['PSNR'])} dB y la correlación de bordes cae a "
        f"{num(realsr['GRAD_CORR'], 3)}, la más baja: los bordes que añade no coinciden con los reales.",
        "<b>SwinIR preentrenado sin reentrenar no restaura imágenes SAR</b>: mejora la apariencia a costa de "
        "inventar detalle. Con 3 imágenes y un GT multitemporal, los números son indicativos y la inspección "
        "visual pesa tanto como las métricas.",
    ]:
        story.append(Paragraph(f"• {bit}", styles["Body"]))

    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.7 * cm,
                            title="Informe Reto 4 — Restauración SAR con SwinIR", author="Darwin Castaño")

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.black)
        canvas.drawString(2.2 * cm, 1.2 * cm, "Reto 4 — Restauración de imágenes SAR con SwinIR")
        canvas.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f"Página {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    print(f"[ok] informe generado: {build_report()}")
