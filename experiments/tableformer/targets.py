"""Compile exact logical table grids into visual-order OTSL targets."""

def otsl(table, *, max_tokens=1024):
    if table.get("schema") != "heocr_table.v1" or table.get("direction") not in {"rtl", "ltr"}:
        raise ValueError("expected a directional heocr_table.v1 table")
    if table.get("dev_only") or table.get("claude_drafted") or table.get("page_breaks") or table.get("blocks_meta"):
        raise ValueError("training requires a single approved physical table fragment")
    pages = table.get("pages")
    if pages is not None and (len(pages) != 2 or pages[0] != pages[1]):
        raise ValueError("table fragment must be page-local")
    rows, cols = table["rows"], table["cols"]
    headers = table.get("header_rows", 0)
    if any(type(n) is not int for n in (rows, cols, headers)) or not (0 <= headers <= rows and rows > 0 and cols > 0):
        raise ValueError("invalid table shape")
    # The final end symbol also consumes one decoder target position.
    if rows * (cols + 1) + 1 > max_tokens:
        raise ValueError("OTSL sequence exceeds the decoder position limit")
    grid = [[None] * cols for _ in range(rows)]
    for cell in table["cells"]:
        r, c, rs, cs = cell["r"], cell["c"], cell["rowspan"], cell["colspan"]
        if (any(type(n) is not int for n in (r, c, rs, cs))
                or not (0 <= r < r + rs <= rows and 0 <= c < c + cs <= cols)):
            raise ValueError("cell lies outside the table")
        if type(cell.get("mask", False)) is not bool or cell.get("mask", False):
            raise ValueError("masked cells require a separately validated physical fragment")
        text = cell["text"]
        if not isinstance(text, str):
            raise ValueError("expected a cell string")
        if table["direction"] == "rtl":
            c = cols - c - cs
        if r < headers:
            anchor = "ched"
        elif cell.get("role") in {"row_header", "section_header"}:
            anchor = {"row_header": "rhed", "section_header": "srow"}[cell["role"]]
        else:
            anchor = "fcel" if text.strip() else "ecel"
        for i in range(r, r + rs):
            for j in range(c, c + cs):
                if grid[i][j] is not None:
                    raise ValueError("overlapping table cells")
                grid[i][j] = (anchor if (i, j) == (r, c) else
                              "lcel" if i == r else "ucel" if j == c else "xcel")
    if any(tag is None for row in grid for tag in row):
        raise ValueError("uncovered grid position; include explicit blank cells")
    return [tag for row in grid for tag in (*row, "nl")]
