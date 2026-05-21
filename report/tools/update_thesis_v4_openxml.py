from __future__ import annotations

import copy
import re
import shutil
import struct
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report"
SRC = REPORT / "Appendix_6_filled_ST4001_senior_thesis_draft_v3_residual.docx"
DST = REPORT / "Appendix_6_filled_ST4001_senior_thesis_draft_v4_standardized.docx"
FIG_DIR = REPORT / "figures" / "standardized"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)
ET.register_namespace("", REL_NS)


def qn(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


def w(tag: str) -> str:
    return qn("w", tag)


def rel(tag: str) -> str:
    return f"{{{REL_NS}}}{tag}"


def ct(tag: str) -> str:
    return f"{{{CT_NS}}}{tag}"


def set_attr(el: ET.Element, prefix: str, name: str, value: str | int) -> None:
    el.set(qn(prefix, name), str(value))


def paragraph_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.findall(".//w:t", NS))


def make_rpr(bold: bool = False, size: int = 12) -> ET.Element:
    rpr = ET.Element(w("rPr"))
    fonts = ET.SubElement(rpr, w("rFonts"))
    set_attr(fonts, "w", "ascii", "Times New Roman")
    set_attr(fonts, "w", "hAnsi", "Times New Roman")
    set_attr(fonts, "w", "eastAsia", "Times New Roman")
    if bold:
        ET.SubElement(rpr, w("b"))
    sz = ET.SubElement(rpr, w("sz"))
    set_attr(sz, "w", "val", size * 2)
    szcs = ET.SubElement(rpr, w("szCs"))
    set_attr(szcs, "w", "val", size * 2)
    return rpr


def make_p(text: str, *, bold: bool = False, size: int = 12, align: str = "both") -> ET.Element:
    p = ET.Element(w("p"))
    ppr = ET.SubElement(p, w("pPr"))
    spacing = ET.SubElement(ppr, w("spacing"))
    set_attr(spacing, "w", "after", 120)
    set_attr(spacing, "w", "line", 360)
    set_attr(spacing, "w", "lineRule", "auto")
    jc = ET.SubElement(ppr, w("jc"))
    set_attr(jc, "w", "val", align)

    r = ET.SubElement(p, w("r"))
    r.append(make_rpr(bold=bold, size=size))
    t = ET.SubElement(r, w("t"))
    if text.startswith(" ") or text.endswith(" "):
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return p


def make_table(caption: str, headers: list[str], rows: list[list[str]]) -> list[ET.Element]:
    elems: list[ET.Element] = [make_p(caption, bold=True, size=10, align="center")]
    tbl = ET.Element(w("tbl"))
    tbl_pr = ET.SubElement(tbl, w("tblPr"))
    borders = ET.SubElement(tbl_pr, w("tblBorders"))
    for name in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        border = ET.SubElement(borders, w(name))
        set_attr(border, "w", "val", "single")
        set_attr(border, "w", "sz", 4)
        set_attr(border, "w", "space", 0)
        set_attr(border, "w", "color", "auto")

    def cell(value: str, bold: bool = False, align: str = "left") -> ET.Element:
        tc = ET.Element(w("tc"))
        tc_pr = ET.SubElement(tc, w("tcPr"))
        width = ET.SubElement(tc_pr, w("tcW"))
        set_attr(width, "w", "w", 0)
        set_attr(width, "w", "type", "auto")
        tc.append(make_p(value, bold=bold, size=9, align=align))
        return tc

    tr = ET.SubElement(tbl, w("tr"))
    for h in headers:
        tr.append(cell(h, bold=True, align="center"))

    for row in rows:
        tr = ET.SubElement(tbl, w("tr"))
        for value in row:
            tr.append(cell(value))

    elems.append(tbl)
    elems.append(make_p("", size=6, align="left"))
    return elems


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def make_image_p(rel_id: str, image_name: str, width_px: int, height_px: int, doc_pr_id: int, max_width_pt: int) -> ET.Element:
    max_width_emu = int(max_width_pt / 72 * 914400)
    height_emu = int(max_width_emu * height_px / width_px)

    p = ET.Element(w("p"))
    ppr = ET.SubElement(p, w("pPr"))
    jc = ET.SubElement(ppr, w("jc"))
    set_attr(jc, "w", "val", "center")
    r = ET.SubElement(p, w("r"))
    drawing = ET.SubElement(r, w("drawing"))
    inline = ET.SubElement(drawing, qn("wp", "inline"), {"distT": "0", "distB": "0", "distL": "0", "distR": "0"})
    ET.SubElement(inline, qn("wp", "extent"), {"cx": str(max_width_emu), "cy": str(height_emu)})
    ET.SubElement(inline, qn("wp", "effectExtent"), {"l": "0", "t": "0", "r": "0", "b": "0"})
    ET.SubElement(inline, qn("wp", "docPr"), {"id": str(doc_pr_id), "name": f"Picture {doc_pr_id}"})
    c_nv = ET.SubElement(inline, qn("wp", "cNvGraphicFramePr"))
    ET.SubElement(c_nv, qn("a", "graphicFrameLocks"), {"noChangeAspect": "1"})

    graphic = ET.SubElement(inline, qn("a", "graphic"))
    graphic_data = ET.SubElement(graphic, qn("a", "graphicData"), {"uri": NS["pic"]})
    pic = ET.SubElement(graphic_data, qn("pic", "pic"))
    nv = ET.SubElement(pic, qn("pic", "nvPicPr"))
    ET.SubElement(nv, qn("pic", "cNvPr"), {"id": "0", "name": image_name})
    ET.SubElement(nv, qn("pic", "cNvPicPr"))
    blip_fill = ET.SubElement(pic, qn("pic", "blipFill"))
    ET.SubElement(blip_fill, qn("a", "blip"), {qn("r", "embed"): rel_id})
    stretch = ET.SubElement(blip_fill, qn("a", "stretch"))
    ET.SubElement(stretch, qn("a", "fillRect"))
    sp_pr = ET.SubElement(pic, qn("pic", "spPr"))
    xfrm = ET.SubElement(sp_pr, qn("a", "xfrm"))
    ET.SubElement(xfrm, qn("a", "off"), {"x": "0", "y": "0"})
    ET.SubElement(xfrm, qn("a", "ext"), {"cx": str(max_width_emu), "cy": str(height_emu)})
    prst = ET.SubElement(sp_pr, qn("a", "prstGeom"), {"prst": "rect"})
    ET.SubElement(prst, qn("a", "avLst"))
    return p


def figure_elements(rel_id: str, image_path: Path, media_name: str, caption: str, doc_pr_id: int, max_width_pt: int) -> list[ET.Element]:
    width, height = png_size(image_path)
    return [
        make_image_p(rel_id, media_name, width, height, doc_pr_id, max_width_pt),
        make_p(caption, bold=True, size=10, align="center"),
    ]


def replace_text_in_document(root: ET.Element, replacements: list[tuple[str, str]]) -> None:
    for p in root.findall(".//w:p", NS):
        full = paragraph_text(p)
        if not full:
            continue
        updated = full
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != full:
            texts = p.findall(".//w:t", NS)
            if texts:
                texts[0].text = updated
                for t in texts[1:]:
                    t.text = ""


def body_children(root: ET.Element) -> list[ET.Element]:
    body = root.find("w:body", NS)
    if body is None:
        raise RuntimeError("Missing document body")
    return list(body)


def find_body_index(root: ET.Element, needle: str) -> int:
    body = root.find("w:body", NS)
    if body is None:
        raise RuntimeError("Missing document body")
    for idx, child in enumerate(list(body)):
        if child.tag == w("p") and needle in paragraph_text(child):
            return idx
    raise ValueError(f"Could not find anchor paragraph: {needle}")


def insert_before(root: ET.Element, needle: str, elems: list[ET.Element]) -> None:
    body = root.find("w:body", NS)
    if body is None:
        raise RuntimeError("Missing document body")
    idx = find_body_index(root, needle)
    for offset, elem in enumerate(elems):
        body.insert(idx + offset, elem)


def next_rid(rels_root: ET.Element) -> int:
    nums = []
    for item in rels_root.findall("rel:Relationship", {"rel": REL_NS}):
        rid = item.attrib.get("Id", "")
        m = re.fullmatch(r"rId(\d+)", rid)
        if m:
            nums.append(int(m.group(1)))
    return max(nums, default=0) + 1


def ensure_png_content_type(content_root: ET.Element) -> None:
    for item in content_root.findall("ct:Default", {"ct": CT_NS}):
        if item.attrib.get("Extension") == "png":
            return
    ET.SubElement(content_root, ct("Default"), {"Extension": "png", "ContentType": "image/png"})


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(SRC)

    with zipfile.ZipFile(SRC, "r") as zin:
        document_xml = zin.read("word/document.xml")
        rels_xml = zin.read("word/_rels/document.xml.rels")
        content_xml = zin.read("[Content_Types].xml")
        existing_names = set(zin.namelist())

    document_root = ET.fromstring(document_xml)
    rels_root = ET.fromstring(rels_xml)
    content_root = ET.fromstring(content_xml)

    replacements = [
        ("Table 1 shows the current main results", "The main quantitative results table shows the current main results"),
        ("Table 1. Quantitative results on the held-out L506 test patient.", "Table 5.1. Quantitative results on the held-out L506 test patient."),
        ("Figure 1. Metric comparison of low-dose input and denoising models on the held-out L506 patient.", "Figure 5.1. Metric comparison of low-dose input and denoising models on the held-out L506 patient."),
        ("Figure 2. Representative qualitative result for slice L506_88.", "Figure 5.2. Representative qualitative result for slice L506_88."),
        ("Figure 3. Training loss trend from the reproduced CTformer run.", "Figure 5.3. Training loss trend from the reproduced CTformer run."),
        ("Figure 4. Residual noise analysis for L506_88.", "Figure 6.1. Original residual noise overview for L506_88."),
        ("Table 2. Residual noise statistics for representative slice L506_88.", "Table 6.1. Residual noise statistics for representative slice L506_88."),
        ("Figure 5. Histogram of removed residual values for L506_88.", "Figure 6.2. Histogram of removed residual values for L506_88."),
        ("Figure 6. Radially averaged residual power spectrum for L506_88.", "Figure 6.3. Radially averaged residual power spectrum for L506_88."),
    ]
    replace_text_in_document(document_root, replacements)

    insert_before(
        document_root,
        "1.4 Thesis Organization",
        [
            make_p(
                "In this thesis, the contribution is deliberately framed as an engineering research contribution rather than a simple benchmark reproduction. The first contribution is a complete paired LDCT denoising pipeline: data preparation, patient-level splitting, model training, checkpoint management, tiled inference, visual comparison, and quantitative evaluation are connected into one reproducible workflow. The second contribution is the adaptation of a Restormer-style restoration Transformer, named CTRestormer in this thesis, to full-resolution CT denoising under limited local GPU memory. The third contribution is the residual noise analysis protocol, which compares the true low-dose residual with the residual removed by each model. This analysis helps identify whether a method wins mainly by suppressing noise-like texture or by over-smoothing anatomical edges.",
                size=12,
            ),
            make_p(
                "These contributions are important because LDCT denoising should not be judged only by global PSNR or SSIM. A model can obtain a strong pixel metric while still producing clinically undesirable smoothing or structure leakage. Therefore, this thesis evaluates both image fidelity and residual behavior. The final discussion treats CTRestormer not as an isolated neural network, but as a practical denoising system whose output, residual maps, and failure risks can be inspected.",
                size=12,
            ),
        ],
    )

    insert_before(
        document_root,
        "Chapter 3. Methodology",
        [
            make_p("2.5 Literature Gap and Thesis Position", bold=True, size=13, align="left"),
            make_p(
                "The reviewed literature shows three limitations that motivate the design of this thesis. First, CNN-based LDCT denoising methods such as RED-CNN are effective and computationally stable, but their local convolutional operators may require deep stacks to model long-range anatomical context. Second, Transformer-based image restoration methods improve global context modeling, but their memory consumption can become problematic for 512 x 512 CT slices. Third, many LDCT denoising papers report PSNR, SSIM, and RMSE, while giving less attention to what signal is removed from the noisy image. SSIM itself was designed to measure structural similarity rather than clinical correctness [8], so it should be interpreted together with qualitative and residual evidence.",
                size=12,
            ),
            make_p(
                "This thesis is positioned at the intersection of these gaps. It keeps RED-CNN as a strong CNN baseline, uses CTformer as a Transformer-based LDCT reference, and adapts the Restormer restoration idea to CT denoising. The attention mechanism is motivated by the general Transformer framework [9] and later vision Transformer developments [10], while the residual analysis is motivated by the need to understand denoising behavior beyond aggregate metrics. The discussion of future Noise2Noise-style work is also connected to blind or self-supervised denoising methods such as Noise2Self and Noise2Void [11], [12].",
                size=12,
            ),
        ],
    )

    insert_before(
        document_root,
        "Chapter 4. Experimental Setup",
        [
            make_p("3.5 Implementation and Reproducibility Details", bold=True, size=13, align="left"),
            make_p(
                "The implementation is designed around paired supervised restoration. Each input slice is a low-dose CT image and each target slice is the corresponding normal-dose CT image. During training, patches are sampled from the paired slices to reduce GPU memory usage and to increase the diversity of local anatomical patterns observed by the network. During evaluation, the full 512 x 512 slice is reconstructed through tiled inference. This is necessary because Transformer-based restoration models may exceed local GPU memory when a complete CT slice is processed at once.",
                size=12,
            ),
            make_p(
                "For CTRestormer, the network predicts a restored image through a residual restoration design. The residual design is useful because the expected output should preserve most anatomical structures from the input while removing dose-induced noise. The training therefore encourages the model to learn a correction between LDCT and NDCT rather than hallucinating an entirely new image. In the evaluation stage, PSNR, SSIM, and RMSE are computed on the held-out L506 patient. The residual analysis additionally computes input minus target, input minus prediction, and prediction minus target, making it possible to compare the removed residual with the estimated true noise.",
                size=12,
            ),
        ],
    )

    config_elems: list[ET.Element] = [make_p("4.4 Dataset and Configuration Summary", bold=True, size=13, align="left")]
    config_elems.extend(
        make_table(
            "Table 4.1. Dataset split and evaluation protocol.",
            ["Item", "Setting", "Purpose"],
            [
                ["Dataset", "AAPM-Mayo 3mm B30 paired LDCT/NDCT slices", "Supervised low-dose to normal-dose CT restoration"],
                ["Patients", "L067, L096, L109, L143, L192, L286, L291, L310, L333, L506", "Patient-level organization of paired slices"],
                ["Test patient", "L506", "Held-out evaluation to avoid slice-level leakage"],
                ["Image size", "512 x 512", "Full CT slice denoising and tiled inference"],
                ["Metrics", "PSNR, SSIM, RMSE, residual correlation, edge leakage", "Image quality and residual behavior evaluation"],
            ],
        )
    )
    config_elems.extend(
        make_table(
            "Table 4.2. Model and inference configuration summary.",
            ["Component", "Configuration", "Reason"],
            [
                ["RED-CNN", "CNN denoising baseline", "Representative convolutional LDCT denoiser"],
                ["CTformer", "Transformer LDCT baseline checkpoint", "Comparison with a Transformer-specific CT model"],
                ["CTRestormer", "Restormer-style encoder-decoder with residual output", "Proposed practical restoration model for LDCT"],
                ["Training", "Patch-based training with memory-aware settings", "Allows training under local GPU limitations"],
                ["Inference", "Tiled full-slice inference", "Avoids out-of-memory errors for 512 x 512 slices"],
            ],
        )
    )
    config_elems.append(
        make_p(
            "These tables are included to make the experimental setup easier to audit. They also separate experimental design from experimental results, which improves thesis readability and makes the later metric tables easier to interpret.",
            size=12,
        )
    )
    insert_before(document_root, "Chapter 5. Experimental Results", config_elems)

    ensure_png_content_type(content_root)
    rid = next_rid(rels_root)
    media_items: list[tuple[str, Path]] = []
    figure_specs = [
        ("L506_88_prediction_comparison_standard.png", "Figure 6.4. Standardized prediction comparison for L506_88 using a shared CT display window.", 430),
        ("L506_88_removed_residuals_standard.png", "Figure 6.5. Removed residual maps computed as input minus prediction. The panels use a shared diverging residual scale to compare removal strength.", 430),
        ("L506_88_prediction_errors_standard.png", "Figure 6.6. Prediction error maps computed as prediction minus target for L506_88.", 430),
        ("L506_88_residual_metrics_bar_standard.png", "Figure 6.7. Residual correlation and edge leakage comparison for the representative L506_88 slice.", 390),
    ]
    figure_elems: list[ET.Element] = [
        make_p("6.2.2 Standardized Residual Figure Presentation", bold=True, size=13, align="left"),
        make_p(
            "The original residual overview is useful for a quick inspection, but it is visually dense. For thesis presentation, the residual analysis is therefore split into standardized figures. The prediction comparison, removed residual maps, prediction error maps, and residual metric bars are separated so that each figure has one visual purpose. The same CT display window is used for prediction comparison, and shared residual scales are used for residual maps where appropriate.",
            size=12,
        ),
        make_p(
            "The standardized comparison in Figure 6.4 compares the low-dose input, normal-dose target, and denoised predictions. Figure 6.5 focuses only on what each model removes from the low-dose input. This distinction is important: a visually smooth denoised image is not necessarily better if the removed residual contains anatomical edges. Figure 6.6 therefore shows the prediction error relative to the normal-dose target, while Figure 6.7 summarizes residual correlation and edge leakage numerically.",
            size=12,
        ),
    ]
    doc_pr_id = 1000
    for file_name, caption, max_width in figure_specs:
        image_path = FIG_DIR / file_name
        media_name = f"v4_{file_name}"
        while f"word/media/{media_name}" in existing_names:
            media_name = f"v4_{rid}_{file_name}"
        rel_id = f"rId{rid}"
        rid += 1
        ET.SubElement(
            rels_root,
            rel("Relationship"),
            {
                "Id": rel_id,
                "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                "Target": f"media/{media_name}",
            },
        )
        media_items.append((media_name, image_path))
        figure_elems.extend(figure_elements(rel_id, image_path, media_name, caption, doc_pr_id, max_width))
        doc_pr_id += 1
    insert_before(document_root, "6.3 Noise Texture and Frequency Analysis", figure_elems)

    insert_before(
        document_root,
        "Chapter 7. Conclusion",
        [
            make_p("6.5 Limitations of the Current Residual Study", bold=True, size=13, align="left"),
            make_p(
                "The residual analysis strengthens the thesis because it checks model behavior beyond PSNR and SSIM. However, the current residual visual figures are based on a representative slice, L506_88. Therefore, they should be interpreted as a detailed case study rather than complete proof over the entire test patient. A stronger final version can extend the same residual metrics to all L506 slices and report the mean and standard deviation of residual correlation, removed residual standard deviation, and edge leakage.",
                size=12,
            ),
            make_p(
                "Another limitation is that residual similarity to input minus target is only an approximation of desirable denoising. The normal-dose target is treated as the reference, but clinical image quality also depends on lesion visibility, tissue boundary preservation, and reader confidence. Future work should therefore combine residual maps with ROI-based measurements and, if possible, clinical task-based evaluation.",
                size=12,
            ),
        ],
    )

    insert_before(
        document_root,
        "Appendix",
        [
            make_p('[8] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity," IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, 2004.', size=12, align="left"),
            make_p('[9] A. Vaswani et al., "Attention is all you need," Advances in Neural Information Processing Systems, 2017, pp. 5998-6008.', size=12, align="left"),
            make_p('[10] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," International Conference on Learning Representations, 2021.', size=12, align="left"),
            make_p('[11] J. Batson and L. Royer, "Noise2Self: Blind denoising by self-supervision," International Conference on Machine Learning, 2019.', size=12, align="left"),
            make_p('[12] A. Krull, T.-O. Buchholz, and F. Jug, "Noise2Void: Learning denoising from single noisy images," Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.', size=12, align="left"),
            make_p('[13] E. Kang, W. Chang, J. Yoo, and J. C. Ye, "Deep convolutional framelet denoising for low-dose CT via wavelet residual network," IEEE Transactions on Medical Imaging, vol. 37, no. 6, pp. 1358-1369, 2018.', size=12, align="left"),
        ],
    )

    replacements_xml = {
        "word/document.xml": ET.tostring(document_root, encoding="utf-8", xml_declaration=True),
        "word/_rels/document.xml.rels": ET.tostring(rels_root, encoding="utf-8", xml_declaration=True),
        "[Content_Types].xml": ET.tostring(content_root, encoding="utf-8", xml_declaration=True),
    }

    tmp = DST.with_suffix(".tmp.docx")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(SRC, "r") as zin, zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = replacements_xml.get(info.filename)
            if data is None:
                data = zin.read(info.filename)
            zout.writestr(info, data)
        for media_name, image_path in media_items:
            zout.write(image_path, f"word/media/{media_name}")

    shutil.move(str(tmp), str(DST))
    print(f"Updated thesis DOCX: {DST}")
    print("PDF export skipped because Microsoft Word automation was not available in the current sandbox.")


if __name__ == "__main__":
    main()
