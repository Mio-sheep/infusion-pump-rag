"""手工构造最小可用 PDF 的测试夹具。

不依赖任何第三方库：直接拼 PDF 的原始语法（对象、xref 表都自己算偏移）。
只能放单字节字符（Helvetica），所以夹具是英文的 —— 中文需要嵌入 CID 字体，
手写不现实；中文路径的验证靠 pypdf 自身的提取能力。
"""

from __future__ import annotations


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(
    pages: list[list[tuple[str, float, float]]],
    title: str = "",
) -> bytes:
    """按 (文字, 字号, y坐标) 生成一个多页 PDF。x 坐标固定为 72。"""
    n_pages = len(pages)
    # 对象编号：1=Catalog 2=Pages 3=Info(可选) 然后每页两个：Page 和 Contents，最后 Font
    font_id = 3 + n_pages * 2 + 1
    info_id: int | None = 3 if title else None
    first_page_id = 4 if title else 3

    objects: dict[int, bytes] = {}
    kids = " ".join(f"{first_page_id + i * 2} 0 R" for i in range(n_pages))
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode()
    if info_id is not None:
        objects[info_id] = f"<< /Title ({_escape(title)}) >>".encode()

    for i, lines in enumerate(pages):
        page_id = first_page_id + i * 2
        content_id = page_id + 1
        info_ref = f"/Info {info_id} 0 R " if info_id is not None else ""
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"{info_ref}"
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()
        stream_lines = []
        for text, size, y in lines:
            stream_lines.append(f"BT /F1 {size:g} Tf 72 {y:g} Td ({_escape(text)}) Tj ET")
        stream = "\n".join(stream_lines).encode("latin-1")
        objects[content_id] = b"<< /Length %d >>\nstream\n%s\nendstream" % (
            len(stream),
            stream,
        )

    objects[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = len(out)
        out += f"{obj_id} 0 obj\n".encode()
        out += objects[obj_id]
        out += b"\nendobj\n"

    n_objects = max(objects) + 1
    xref_pos = len(out)
    out += f"xref\n0 {n_objects}\n".encode()
    out += b"0000000000 65535 f \n"
    for obj_id in range(1, n_objects):
        if obj_id in offsets:
            out += f"{offsets[obj_id]:010d} 00000 n \n".encode()
        else:
            out += b"0000000000 65535 f \n"
    root_ref = "/Root 1 0 R"
    info_ref = f"/Info {info_id} 0 R" if info_id is not None else ""
    out += f"trailer << /Size {n_objects} {root_ref} {info_ref} >>\n".encode()
    out += f"startxref\n{xref_pos}\n%%EOF".encode()
    return bytes(out)
