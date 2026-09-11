
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import tensorflow as tf
import pandas as pd
import plotly.express as px
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from io import BytesIO
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fashion Vision AI | ANN Report",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(59,130,246,.12), transparent 28%),
        radial-gradient(circle at 90% 10%, rgba(168,85,247,.10), transparent 28%),
        #080d18;
}
.main .block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 30px;
    border-radius: 24px;
    background: linear-gradient(135deg, #111827, #0f172a);
    border: 1px solid rgba(148,163,184,.22);
    box-shadow: 0 12px 40px rgba(0,0,0,.25);
    margin-bottom: 24px;
}
.hero h1 { margin: 0; font-size: 42px; }
.hero p { color: #9ca3af; margin: 8px 0 0; font-size: 16px; }
.card {
    background: rgba(15,23,42,.82);
    border: 1px solid rgba(148,163,184,.18);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 16px;
}
.metric-card {
    background: rgba(15,23,42,.86);
    border: 1px solid rgba(148,163,184,.16);
    border-radius: 16px;
    padding: 16px;
    text-align: center;
}
.metric-label {
    color: #94a3b8;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .7px;
}
.metric-value { font-size: 25px; font-weight: 800; margin-top: 5px; }
.prediction-card {
    background: linear-gradient(135deg, rgba(30,41,59,.96), rgba(15,23,42,.96));
    border: 1px solid rgba(96,165,250,.30);
    border-radius: 20px;
    padding: 24px;
}
.prediction-name { font-size: 34px; font-weight: 850; margin: 4px 0; }
.prediction-confidence { color: #cbd5e1; font-size: 18px; }
.section-title { font-size: 22px; font-weight: 800; margin: 22px 0 12px; }
.badge {
    display:inline-block;
    padding:5px 10px;
    border-radius:999px;
    background:rgba(59,130,246,.14);
    border:1px solid rgba(96,165,250,.25);
    color:#bfdbfe;
    font-size:12px;
    margin-right:5px;
}
.footer {
    text-align:center;
    color:#64748b;
    padding:25px 0 10px;
    margin-top:35px;
    border-top:1px solid rgba(148,163,184,.12);
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ANN.keras")


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load ANN.keras.")
    st.code(str(e))
    st.warning("Keep ANN.keras in the same folder as app.py and use a compatible TensorFlow environment.")
    st.stop()


# ============================================================
# CLASS DEFINITIONS
# ============================================================

class_names = [
    "T-shirt / Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

class_descriptions = {
    "T-shirt / Top": "A lightweight upper-body fashion category. The model associates the image pattern with the Fashion-MNIST T-shirt/top class.",
    "Trouser": "A lower-body garment category representing trousers or pants. The prediction is based on the learned 28×28 grayscale visual pattern.",
    "Pullover": "A long-sleeved upper-body garment category. The ANN identifies visual characteristics associated with the Fashion-MNIST pullover class.",
    "Dress": "A one-piece clothing category. The model identifies the image as visually similar to the Fashion-MNIST dress examples.",
    "Coat": "An outerwear category representing coats or similar heavier garments. The ANN prediction comes from learned pixel-level patterns.",
    "Sandal": "An open footwear category. The model recognizes the image as visually similar to the Fashion-MNIST sandal class.",
    "Shirt": "An upper-body shirt category. The ANN maps the processed grayscale pattern to the learned Fashion-MNIST shirt class.",
    "Sneaker": "A footwear category representing athletic or casual sneakers. The prediction is based on learned image patterns.",
    "Bag": "An accessory category representing bags. The ANN identifies the uploaded image as similar to Fashion-MNIST bag examples.",
    "Ankle Boot": "A closed footwear category with a boot-like silhouette. The ANN maps the image pattern to the Fashion-MNIST ankle-boot class."
}


# ============================================================
# HELPERS
# ============================================================

def prepare_image(image):
    """Convert uploaded image to the ANN's expected grayscale format."""
    image = ImageOps.exif_transpose(image)
    original = image.copy()

    arr = np.array(image)

    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Unsupported image structure.")

    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) == 4:
        sample = np.expand_dims(normalized, axis=(0, -1))
    elif len(input_shape) == 3:
        sample = np.expand_dims(normalized, axis=0)
    else:
        raise ValueError(
            f"Unsupported ANN input shape: {input_shape}. "
            "Expected a 28x28 image input."
        )

    return original, arr, gray, resized, normalized, sample


def get_probabilities(raw_prediction):
    """Return a valid probability vector, applying softmax when the model outputs logits."""
    raw = np.asarray(raw_prediction).reshape(-1).astype(np.float64)

    if len(raw) != len(class_names):
        raise ValueError(
            f"Model returned {len(raw)} outputs, but {len(class_names)} class names are configured."
        )

    total = raw.sum()

    if (
        np.all(raw >= 0)
        and np.all(raw <= 1)
        and np.isclose(total, 1.0, atol=1e-3)
    ):
        return raw

    exp_values = np.exp(raw - np.max(raw))
    return exp_values / exp_values.sum()


def confidence_level(confidence):
    if confidence >= 85:
        return "Very High"
    if confidence >= 70:
        return "High"
    if confidence >= 50:
        return "Moderate"
    return "Low"


def build_report_pdf(
    original_image,
    uploaded_name,
    predicted_class,
    confidence,
    probabilities,
    image_width,
    image_height,
    image_format,
    file_size_kb,
    gray,
    resized,
    normalized,
    model_input_shape,
    model_output_shape,
    parameter_count,
    top_indices,
    brightness,
    contrast,
):
    """Create a professional Fashion-MNIST style AI analysis report."""

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Fashion Vision AI Analysis Report",
        author="Fashion Vision AI"
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=24,
        leading=29,
        alignment=TA_CENTER,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#64748B"),
        alignment=TA_CENTER,
        spaceAfter=14
    )

    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        spaceBefore=10,
        spaceAfter=8
    )

    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        spaceBefore=8,
        spaceAfter=5
    )

    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        spaceAfter=6
    )

    small = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B")
    )

    story = []

    generated_at = datetime.now().strftime("%d %B %Y, %I:%M %p")

    story.append(Paragraph("FASHION VISION AI", title_style))
    story.append(Paragraph("ANN IMAGE CLASSIFICATION & VISUAL ANALYSIS REPORT", subtitle_style))
    story.append(Paragraph(f"<b>Generated:</b> {generated_at}", subtitle_style))
    story.append(Spacer(1, 5))

    # Image
    image_buffer = BytesIO()
    original_image.convert("RGB").save(image_buffer, format="JPEG", quality=92)
    image_buffer.seek(0)

    story.append(Paragraph("1. Visual Overview", h1))
    story.append(RLImage(image_buffer, width=80 * mm, height=80 * mm))
    story.append(Spacer(1, 6))

    meta_data = [
        ["File", uploaded_name],
        ["Original dimensions", f"{image_width} × {image_height} px"],
        ["Format", image_format],
        ["File size", f"{file_size_kb:.2f} KB"],
        ["Brightness", f"{brightness:.2f} / 255"],
        ["Contrast", f"{contrast:.2f}"],
    ]

    meta_table = Table(meta_data, colWidths=[48 * mm, 125 * mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1E293B")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)

    # Prediction
    story.append(Paragraph("2. AI Prediction", h1))

    prediction_data = [
        ["Predicted class", predicted_class],
        ["Confidence", f"{confidence:.2f}%"],
        ["Confidence level", confidence_level(confidence)],
        ["Model", "Artificial Neural Network (ANN)"],
    ]

    pred_table = Table(prediction_data, colWidths=[48 * mm, 125 * mm])
    pred_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E0F2FE")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BAE6FD")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        f"<b>AI-generated class description:</b> {class_descriptions[predicted_class]}",
        body
    ))

    if confidence >= 85:
        interpretation = "The ANN shows strong confidence in the predicted class."
    elif confidence >= 70:
        interpretation = "The ANN shows a reasonably strong preference for the predicted class, while alternative classes should still be considered."
    elif confidence >= 50:
        interpretation = "The prediction is moderate. The top alternatives are relatively important when interpreting this result."
    else:
        interpretation = "The prediction has low confidence. This image may differ significantly from the patterns learned by the ANN."

    story.append(Paragraph(f"<b>Confidence interpretation:</b> {interpretation}", body))

    # Top 3
    story.append(Paragraph("3. Top-3 Classification Results", h1))

    top_table_data = [["Rank", "Class", "Probability"]]
    for rank, idx in enumerate(top_indices, 1):
        top_table_data.append([
            str(rank),
            class_names[idx],
            f"{probabilities[idx] * 100:.2f}%"
        ])

    top_table = Table(top_table_data, colWidths=[25 * mm, 105 * mm, 43 * mm])
    top_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (-1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(top_table)

    # Full probability table with visual bars
    story.append(Paragraph("4. Complete Probability Distribution", h1))

    prob_table_data = [["Class", "Probability", "Visual confidence"]]

    for idx in np.argsort(probabilities)[::-1]:
        pct = float(probabilities[idx] * 100)
        bar_count = min(30, max(1, int(pct / 3.34)))
        bar = "█" * bar_count
        prob_table_data.append([
            class_names[idx],
            f"{pct:.2f}%",
            bar
        ])

    prob_table = Table(prob_table_data, colWidths=[50 * mm, 30 * mm, 93 * mm])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ("FONTNAME", (2, 1), (2, -1), "Helvetica"),
        ("TEXTCOLOR", (2, 1), (2, -1), colors.HexColor("#2563EB")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(prob_table)

    # Preprocessing
    story.append(PageBreak())
    story.append(Paragraph("5. ANN Image Processing Pipeline", h1))

    pipeline = [
        ["Stage", "Operation", "Result"],
        ["1", "Uploaded image", f"{image_width} × {image_height} px"],
        ["2", "Orientation correction", "EXIF-aware"],
        ["3", "Color conversion", "Grayscale"],
        ["4", "Resize", "28 × 28 pixels"],
        ["5", "Normalization", "Pixel values ÷ 255"],
        ["6", "Batch preparation", str(tuple(model.input_shape) if not isinstance(model.input_shape, list) else model.input_shape)],
        ["7", "ANN inference", str(model_output_shape)],
        ["8", "Decision", predicted_class],
    ]

    pipeline_table = Table(pipeline, colWidths=[20 * mm, 65 * mm, 88 * mm])
    pipeline_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(pipeline_table)
    story.append(Spacer(1, 12))

    # Processed image
    processed_buffer = BytesIO()
    normalized_uint8 = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    Image.fromarray(normalized_uint8).save(processed_buffer, format="PNG")
    processed_buffer.seek(0)

    story.append(Paragraph("Processed ANN Input — 28 × 28", h2))
    story.append(RLImage(processed_buffer, width=65 * mm, height=65 * mm))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        f"<b>Normalized pixel range:</b> {normalized.min():.4f} to {normalized.max():.4f}<br/>"
        f"<b>Grayscale matrix shape:</b> {gray.shape}<br/>"
        f"<b>Final resized shape:</b> {resized.shape}",
        body
    ))

    # Model details
    story.append(Paragraph("6. Model Information", h1))

    model_data = [
        ["Property", "Value"],
        ["Architecture", "Artificial Neural Network"],
        ["Input shape", str(model_input_shape)],
        ["Output shape", str(model_output_shape)],
        ["Classes", str(len(class_names))],
        ["Trainable parameters", f"{parameter_count:,}"],
        ["Framework", "TensorFlow / Keras"],
    ]

    model_table = Table(model_data, colWidths=[60 * mm, 113 * mm])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("7. Professional Interpretation", h1))
    story.append(Paragraph(
        f"The Fashion Vision AI system analyzed <b>{uploaded_name}</b> using a trained ANN. "
        f"The leading classification is <b>{predicted_class}</b> with a confidence of "
        f"<b>{confidence:.2f}%</b>. The result is based on the visual patterns learned by "
        f"the model from its training data after grayscale conversion, 28×28 resizing and pixel normalization.",
        body
    ))
    story.append(Paragraph(
        "This report is an AI model-analysis document and should be interpreted as a classification result, "
        "not as a human expert garment inspection or product-authentication certificate.",
        small
    ))

    # Footer on every page
    def add_footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(16 * mm, 8 * mm, "Fashion Vision AI • ANN Image Classification")
        canvas.drawRightString(
            A4[0] - 16 * mm,
            8 * mm,
            f"Page {doc_obj.page}"
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
    <h1>🧠 Fashion Vision AI</h1>
    <p>
        Professional ANN-powered Fashion-MNIST image analysis with
        interactive probability intelligence and downloadable AI reports.
    </p>
    <div style="margin-top:14px;">
        <span class="badge">TensorFlow</span>
        <span class="badge">ANN</span>
        <span class="badge">OpenCV</span>
        <span class="badge">Streamlit</span>
        <span class="badge">Plotly</span>
        <span class="badge">PDF Reporting</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ AI Model")

    st.write("**Architecture:** Artificial Neural Network")
    st.write("**Input:** 28 × 28 grayscale")
    st.write("**Output:** 10 classes")
    st.write("**Normalization:** Pixel / 255")

    st.divider()

    st.subheader("📁 Supported Images")
    st.write("JPG • JPEG • PNG • WEBP • BMP • TIFF • TIF • JFIF")

    st.divider()

    st.subheader("📊 Report Includes")
    st.write("✓ Original image")
    st.write("✓ AI prediction")
    st.write("✓ Confidence score")
    st.write("✓ Top-3 results")
    st.write("✓ All class probabilities")
    st.write("✓ Processing pipeline")
    st.write("✓ Model information")
    st.write("✓ Image statistics")
    st.write("✓ 28×28 ANN input")
    st.write("✓ Professional PDF report")


# ============================================================
# UPLOAD
# ============================================================

st.markdown('<div class="section-title">📤 Upload Fashion Image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image for AI classification",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "jfif"],
    help="The ANN converts the uploaded image into a 28×28 grayscale input."
)


if uploaded_file is None:
    st.info("Upload an image to start the complete AI analysis.")
    st.markdown("""
    <div class="card">
        <b>🚀 Analysis workflow</b><br><br>
        Upload → Preprocess → ANN Prediction → Probability Analysis →
        Image Description → Professional PDF Report
    </div>
    """, unsafe_allow_html=True)

else:
    try:
        # ====================================================
        # PREPROCESS
        # ====================================================

        image = Image.open(uploaded_file)

        (
            original,
            arr,
            gray,
            resized,
            normalized,
            sample
        ) = prepare_image(image)

        # ====================================================
        # IMAGE STATS
        # ====================================================

        width, height = original.size
        file_size_kb = uploaded_file.size / 1024
        image_format = original.format or "Unknown"

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        aspect_ratio = width / height if height else 0

        # ====================================================
        # PREDICT
        # ====================================================

        raw_prediction = model.predict(sample, verbose=0)
        probabilities = get_probabilities(raw_prediction)

        predicted_index = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_index]
        confidence = float(probabilities[predicted_index] * 100)

        top_indices = np.argsort(probabilities)[::-1][:3]

        # ====================================================
        # KPI
        # ====================================================

        st.markdown('<div class="section-title">📊 Image Intelligence</div>', unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)

        metrics = [
            ("Prediction", predicted_class),
            ("Confidence", f"{confidence:.2f}%"),
            ("Image Size", f"{width}×{height}"),
            ("Brightness", f"{brightness:.1f}"),
            ("Aspect Ratio", f"{aspect_ratio:.2f}")
        ]

        for col, (label, value) in zip([c1, c2, c3, c4, c5], metrics):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ====================================================
        # IMAGE + PREDICTION
        # ====================================================

        st.markdown('<div class="section-title">🖼️ Visual Analysis</div>', unsafe_allow_html=True)

        left, right = st.columns([1.05, 1])

        with left:
            st.markdown('<div class="card"><b>Original Uploaded Image</b></div>', unsafe_allow_html=True)
            st.image(original, use_container_width=True)

            with st.expander("🔬 View ANN Preprocessed Image"):
                st.image(
                    normalized,
                    caption="28 × 28 grayscale + normalized input",
                    width=320,
                    clamp=True
                )

        with right:
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div style="color:#94a3b8;font-size:12px;letter-spacing:1px;">
                        ANN CLASSIFICATION
                    </div>
                    <div class="prediction-name">{predicted_class}</div>
                    <div class="prediction-confidence">
                        Confidence: <b>{confidence:.2f}%</b>
                    </div>
                    <div style="margin-top:12px;color:#cbd5e1;">
                        Confidence level: <b>{confidence_level(confidence)}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(min(confidence / 100, 1.0))

            st.markdown("#### 📝 AI Image Description")
            st.write(class_descriptions[predicted_class])

            if confidence >= 85:
                st.success("The ANN has a very strong preference for this class.")
            elif confidence >= 70:
                st.info("The ANN has a strong preference for this class.")
            elif confidence >= 50:
                st.warning("The ANN prediction is moderate; review the alternative classes.")
            else:
                st.warning("The ANN is uncertain. The image may be outside the strongest learned patterns.")

        # ====================================================
        # TOP 3
        # ====================================================

        st.markdown('<div class="section-title">🏆 Top-3 Predictions</div>', unsafe_allow_html=True)

        top_cols = st.columns(3)

        for rank, (col, idx) in enumerate(zip(top_cols, top_indices), start=1):
            pct = float(probabilities[idx] * 100)
            with col:
                st.metric(
                    f"#{rank} {class_names[idx]}",
                    f"{pct:.2f}%"
                )
                st.progress(float(probabilities[idx]))

        # ====================================================
        # INTERACTIVE PLOTLY CHART
        # ====================================================

        st.markdown('<div class="section-title">📈 Interactive Probability Intelligence</div>', unsafe_allow_html=True)

        probability_df = pd.DataFrame({
            "Class": class_names,
            "Probability": probabilities * 100
        }).sort_values("Probability", ascending=True)

        probability_df["Status"] = np.where(
            probability_df["Class"] == predicted_class,
            "Predicted Class",
            "Alternative"
        )

        fig = px.bar(
            probability_df,
            x="Probability",
            y="Class",
            orientation="h",
            text="Probability",
            color="Status",
            hover_data={"Probability": ":.2f"},
            title="ANN Class Probability Distribution",
            labels={
                "Probability": "Probability (%)",
                "Class": "Fashion Class",
                "Status": "Classification"
            }
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            height=560,
            xaxis=dict(range=[0, max(100, float(probability_df["Probability"].max()) + 8)]),
            legend_title="",
            margin=dict(l=10, r=30, t=60, b=10)
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "scrollZoom": True
            }
        )

        st.caption("Hover over a bar for exact probability. Use the chart toolbar to zoom or reset the view.")

        # ====================================================
        # PROBABILITY TABLE
        # ====================================================

        st.markdown("#### 📋 Detailed Probability Table")

        display_df = probability_df.sort_values(
            "Probability",
            ascending=False
        ).copy()

        display_df["Probability"] = display_df["Probability"].map(
            lambda x: f"{x:.2f}%"
        )

        st.dataframe(
            display_df[["Class", "Probability", "Status"]],
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # PROCESSING PIPELINE
        # ====================================================

        st.markdown('<div class="section-title">🔄 ANN Processing Pipeline</div>', unsafe_allow_html=True)

        p1, p2, p3, p4, p5, p6 = st.columns(6)

        steps = [
            ("1", "Upload", "Original image"),
            ("2", "Grayscale", "Color → gray"),
            ("3", "Resize", "28 × 28"),
            ("4", "Normalize", "÷ 255"),
            ("5", "ANN", "Inference"),
            ("6", "Decision", predicted_class),
        ]

        for col, (num, title, detail) in zip([p1, p2, p3, p4, p5, p6], steps):
            with col:
                st.markdown(
                    f"""
                    <div class="card" style="text-align:center;min-height:105px;">
                        <div style="font-size:12px;color:#60a5fa;">STEP {num}</div>
                        <b>{title}</b><br>
                        <span style="color:#94a3b8;font-size:12px;">{detail}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ====================================================
        # MODEL + IMAGE DETAILS
        # ====================================================

        d1, d2 = st.columns(2)

        with d1:
            with st.expander("🧠 Model Details"):
                st.write("**Model:** Artificial Neural Network")
                st.write(f"**Input shape:** `{model.input_shape}`")
                st.write(f"**Output shape:** `{model.output_shape}`")
                st.write(f"**Parameters:** `{model.count_params():,}`")
                st.write("**Classes:** 10")
                st.write("**Framework:** TensorFlow / Keras")

        with d2:
            with st.expander("🧪 Preprocessing Details"):
                st.write(f"**Original array shape:** `{arr.shape}`")
                st.write(f"**Grayscale shape:** `{gray.shape}`")
                st.write(f"**Resized shape:** `{resized.shape}`")
                st.write(f"**Final model input shape:** `{sample.shape}`")
                st.write(f"**Pixel range:** `{normalized.min():.4f} → {normalized.max():.4f}`")
                st.write(f"**Contrast (std):** `{contrast:.2f}`")

        # ====================================================
        # PDF REPORT
        # ====================================================

        st.markdown('<div class="section-title">📄 Professional AI Report</div>', unsafe_allow_html=True)

        pdf_bytes = build_report_pdf(
            original_image=original,
            uploaded_name=uploaded_file.name,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            image_width=width,
            image_height=height,
            image_format=image_format,
            file_size_kb=file_size_kb,
            gray=gray,
            resized=resized,
            normalized=normalized,
            model_input_shape=model.input_shape,
            model_output_shape=model.output_shape,
            parameter_count=model.count_params(),
            top_indices=top_indices,
            brightness=brightness,
            contrast=contrast,
        )

        report_filename = (
            f"Fashion_Vision_Report_"
            f"{predicted_class.replace('/', '-').replace(' ', '_')}.pdf"
        )

        st.download_button(
            "📥 Download Professional PDF Report",
            data=pdf_bytes,
            file_name=report_filename,
            mime="application/pdf",
            use_container_width=True
        )

        st.success(
            "Your professional AI report is ready. It contains the uploaded image, "
            "prediction, confidence, top-3 results, all probabilities, preprocessing "
            "pipeline, model details and AI interpretation."
        )

    except Exception as e:
        st.error("❌ The image could not be processed.")
        with st.expander("Show technical error"):
            st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    🧠 <b>Fashion Vision AI</b><br>
    ANN Image Classification • TensorFlow • OpenCV • Plotly • Streamlit • ReportLab
</div>
""", unsafe_allow_html=True)
