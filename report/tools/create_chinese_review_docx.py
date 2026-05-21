from __future__ import annotations

import re
import struct
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report"
SRC_MD = REPORT / "ST4001_senior_thesis_chinese_review_draft.md"
OUT_DOCX = REPORT / "ST4001_senior_thesis_chinese_review_draft.docx"
FIG_DIR = REPORT / "figures" / "standardized"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    return struct.unpack(">II", data[16:24])


def clean_inline(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("\\(", "(").replace("\\)", ")")
    return text


def rpr(size: int = 24, bold: bool = False) -> str:
    bold_xml = "<w:b/>" if bold else ""
    return (
        "<w:rPr>"
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
        'w:eastAsia="SimSun" w:cs="Times New Roman"/>'
        f"{bold_xml}<w:sz w:val=\"{size}\"/><w:szCs w:val=\"{size}\"/>"
        "</w:rPr>"
    )


def paragraph(text: str, *, size: int = 24, bold: bool = False, align: str = "both", before: int = 0, after: int = 120) -> str:
    text = clean_inline(text)
    return (
        "<w:p><w:pPr>"
        f'<w:spacing w:before="{before}" w:after="{after}" w:line="360" w:lineRule="auto"/>'
        f'<w:jc w:val="{align}"/>'
        "</w:pPr><w:r>"
        f"{rpr(size=size, bold=bold)}<w:t>{escape(text)}</w:t>"
        "</w:r></w:p>"
    )


def heading(text: str, level: int) -> str:
    text = clean_inline(text)
    if level == 1:
        return paragraph(text, size=32, bold=True, align="center", before=240, after=180)
    if level == 2:
        return paragraph(text, size=28, bold=True, align="left", before=260, after=140)
    return paragraph(text, size=25, bold=True, align="left", before=180, after=100)


def table(rows: list[list[str]]) -> str:
    out = [
        "<w:tbl><w:tblPr><w:tblBorders>",
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>',
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>',
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>',
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/>',
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/>',
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/>',
        "</w:tblBorders></w:tblPr>",
    ]
    for r, row in enumerate(rows):
        out.append("<w:tr>")
        for cell in row:
            out.append("<w:tc><w:tcPr><w:tcW w:w=\"0\" w:type=\"auto\"/></w:tcPr>")
            out.append(paragraph(cell.strip(), size=20, bold=(r == 0), align="center" if r == 0 else "left", after=60))
            out.append("</w:tc>")
        out.append("</w:tr>")
    out.append("</w:tbl>")
    out.append(paragraph("", size=8, after=40))
    return "".join(out)


def image_para(rid: str, name: str, image_path: Path, doc_pr_id: int, caption: str, max_width_pt: int = 430) -> str:
    width_px, height_px = png_size(image_path)
    width_emu = int(max_width_pt / 72 * 914400)
    height_emu = int(width_emu * height_px / width_px)
    drawing = f"""
<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing>
<wp:inline distT="0" distB="0" distL="0" distR="0">
<wp:extent cx="{width_emu}" cy="{height_emu}"/>
<wp:effectExtent l="0" t="0" r="0" b="0"/>
<wp:docPr id="{doc_pr_id}" name="Picture {doc_pr_id}"/>
<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
<a:graphic><a:graphicData uri="{PIC_NS}">
<pic:pic>
<pic:nvPicPr><pic:cNvPr id="0" name="{escape(name)}"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{width_emu}" cy="{height_emu}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
</pic:pic>
</a:graphicData></a:graphic>
</wp:inline>
</w:drawing></w:r></w:p>
"""
    return drawing + paragraph(caption, size=20, bold=True, align="center", after=120)


def parse_markdown(md: str) -> tuple[str, list[tuple[str, Path]]]:
    blocks: list[str] = []
    media: list[tuple[str, Path]] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("# "):
            blocks.append(heading(line[2:].strip(), 1))
            i += 1
            continue
        if line.startswith("## "):
            blocks.append(heading(line[3:].strip(), 2))
            i += 1
            continue
        if line.startswith("### "):
            blocks.append(heading(line[4:].strip(), 3))
            i += 1
            continue
        if line.startswith(">"):
            blocks.append(paragraph(line.lstrip("> ").strip(), size=21, align="left"))
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", lines[i + 1].strip()):
            rows: list[list[str]] = []
            rows.append([clean_inline(c.strip()) for c in line.strip("|").split("|")])
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([clean_inline(c.strip()) for c in lines[i].strip("|").split("|")])
                i += 1
            blocks.append(table(rows))
            continue
        blocks.append(paragraph(line, size=24, align="both"))
        i += 1

    figure_specs = [
        ("L506_88_prediction_comparison_standard.png", "附图 1. L506_88 输入、target 与模型预测结果标准化对比。"),
        ("L506_88_removed_residuals_standard.png", "附图 2. 模型移除残差图，显示 input minus prediction。"),
        ("L506_88_prediction_errors_standard.png", "附图 3. 预测误差图，显示 prediction minus target。"),
        ("L506_88_residual_metrics_bar_standard.png", "附图 4. Residual correlation 与 edge leakage 柱状图。"),
    ]
    blocks.append(heading("附录：标准化 residual 图组", 2))
    for idx, (file_name, caption) in enumerate(figure_specs, start=1):
        path = FIG_DIR / file_name
        rid = f"rId{idx + 10}"
        media.append((rid, path))
        blocks.append(image_para(rid, file_name, path, idx, caption, max_width_pt=430 if idx < 4 else 390))

    return "\n".join(blocks), media


def main() -> None:
    md = SRC_MD.read_text(encoding="utf-8")
    body_xml, media = parse_markdown(md)
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}" xmlns:wp="{WP_NS}" xmlns:a="{A_NS}" xmlns:pic="{PIC_NS}">
<w:body>
{body_xml}
<w:sectPr>
<w:pgSz w:w="11906" w:h="16838"/>
<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
</w:sectPr>
</w:body>
</w:document>
"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    image_rels = []
    for rid, path in media:
        image_rels.append(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{path.name}"/>'
        )
    doc_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        + "\n".join(image_rels)
        + "\n</Relationships>\n"
    )

    if OUT_DOCX.exists():
        OUT_DOCX.unlink()
    with zipfile.ZipFile(OUT_DOCX, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        for _, path in media:
            z.write(path, f"word/media/{path.name}")
    print(f"Chinese review DOCX written: {OUT_DOCX}")


if __name__ == "__main__":
    main()
