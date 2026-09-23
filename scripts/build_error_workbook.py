from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = ROOT / "outputs" / "field_predictions.npz"
SAVED_TRUTH_PATH = ROOT / "outputs" / "field_ground_truth.npz"
TRUTH_PATH = ROOT / "u_stack.npz"
THREAD_ID = "019fb973-f9ab-7fa3-ae22-6b338a8db385"
OUTPUT_DIR = ROOT / "outputs" / THREAD_ID
OUTPUT_PATH = OUTPUT_DIR / "u_error_heatmap_mse.xlsx"

NAVY = "17365D"
BLUE = "1F4E78"
LIGHT_BLUE = "D9EAF7"
PALE_BLUE = "EAF3F8"
GREEN = "70AD47"
GOLD = "FFC000"
WHITE = "FFFFFF"
TEXT = "1F2937"
MUTED = "5B6573"
GRID = "D9E1F2"


def interpolate_color(value: float, lower: float, upper: float) -> str:
    """Approximate viridis with purple -> teal -> yellow."""
    if not math.isfinite(value):
        return "FFFFFF"
    if upper <= lower:
        position = 0.5
    else:
        position = min(1.0, max(0.0, (value - lower) / (upper - lower)))
    stops = ((0.0, (68, 1, 84)), (0.5, (33, 145, 140)), (1.0, (253, 231, 37)))
    if position <= 0.5:
        left_p, left_rgb = stops[0]
        right_p, right_rgb = stops[1]
    else:
        left_p, left_rgb = stops[1]
        right_p, right_rgb = stops[2]
    fraction = (position - left_p) / (right_p - left_p)
    rgb = tuple(round(a + (b - a) * fraction) for a, b in zip(left_rgb, right_rgb))
    return "".join(f"{channel:02X}" for channel in rgb)


def fill_cache(lower: float, upper: float, bins: int = 128):
    fills = [
        PatternFill("solid", fgColor=interpolate_color(lower + (upper - lower) * i / (bins - 1), lower, upper))
        for i in range(bins)
    ]

    def pick(value: float) -> PatternFill:
        if upper <= lower:
            return fills[bins // 2]
        index = round((min(upper, max(lower, float(value))) - lower) / (upper - lower) * (bins - 1))
        return fills[index]

    return pick


def style_title(ws, title: str, subtitle: str, last_col: int = 8):
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=last_col)
    ws["A1"] = title
    ws["A1"].font = Font(name="Aptos Display", size=18, bold=True, color=WHITE)
    ws["A1"].fill = PatternFill("solid", fgColor=NAVY)
    ws["A1"].alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 28
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=last_col)
    ws["A2"] = subtitle
    ws["A2"].font = Font(name="Aptos", size=10, italic=True, color=MUTED)
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[2].height = 30


def add_full_field_sheet(wb, name: str, field: np.ndarray, x: np.ndarray, y: np.ndarray,
                         scale_min: float, scale_max: float, subtitle: str):
    ws = wb.create_sheet(name)
    height, width = field.shape
    last_col = width + 1
    last_row = height + 4
    style_title(ws, name, subtitle, min(last_col, 10))
    ws["A4"] = "y \\ x"
    ws["A4"].font = Font(bold=True, color=WHITE)
    ws["A4"].fill = PatternFill("solid", fgColor=BLUE)
    ws["A4"].alignment = Alignment(horizontal="center")
    for col, x_value in enumerate(x, start=2):
        cell = ws.cell(4, col, float(x_value))
        cell.font = Font(size=7, bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=BLUE)
        cell.alignment = Alignment(horizontal="center", text_rotation=90)
        ws.column_dimensions[get_column_letter(col)].width = 2.2
    ws.column_dimensions["A"].width = 8
    choose_fill = fill_cache(scale_min, scale_max)
    # Reverse y so the worksheet view matches Matplotlib origin='lower'.
    for display_row, array_row in enumerate(range(height - 1, -1, -1), start=5):
        y_cell = ws.cell(display_row, 1, float(y[array_row]))
        y_cell.font = Font(size=8, bold=True, color=WHITE)
        y_cell.fill = PatternFill("solid", fgColor=BLUE)
        y_cell.alignment = Alignment(horizontal="center")
        ws.row_dimensions[display_row].height = 6
        for col, value in enumerate(field[array_row], start=2):
            cell = ws.cell(display_row, col, float(value))
            cell.fill = choose_fill(float(value))
            cell.number_format = "0.0000"
            cell.font = Font(size=1, color=interpolate_color(float(value), scale_min, scale_max))
    data_range = f"B5:{get_column_letter(last_col)}{last_row}"
    ws.conditional_formatting.add(
        data_range,
        ColorScaleRule(start_type="min", start_color="440154", mid_type="percentile", mid_value=50,
                       mid_color="21918C", end_type="max", end_color="FDE725"),
    )
    ws.freeze_panes = "B5"
    ws.sheet_view.zoomScale = 20
    ws.sheet_view.showGridLines = False
    ws.auto_filter.ref = f"A4:{get_column_letter(last_col)}{last_row}"
    return ws


def add_formula_field_sheet(wb, name: str, truth: np.ndarray, pred: np.ndarray, x: np.ndarray, y: np.ndarray,
                            formula_kind: str, scale_min: float, scale_max: float, subtitle: str):
    ws = wb.create_sheet(name)
    height, width = truth.shape
    last_col = width + 1
    last_row = height + 4
    style_title(ws, name, subtitle, min(last_col, 10))
    ws["A4"] = "y \\ x"
    ws["A4"].font = Font(bold=True, color=WHITE)
    ws["A4"].fill = PatternFill("solid", fgColor=BLUE)
    ws["A4"].alignment = Alignment(horizontal="center")
    for col, x_value in enumerate(x, start=2):
        cell = ws.cell(4, col, float(x_value))
        cell.font = Font(size=7, bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=BLUE)
        cell.alignment = Alignment(horizontal="center", text_rotation=90)
        ws.column_dimensions[get_column_letter(col)].width = 2.2
    ws.column_dimensions["A"].width = 8
    choose_fill = fill_cache(scale_min, scale_max)
    for display_row, array_row in enumerate(range(height - 1, -1, -1), start=5):
        y_cell = ws.cell(display_row, 1, float(y[array_row]))
        y_cell.font = Font(size=8, bold=True, color=WHITE)
        y_cell.fill = PatternFill("solid", fgColor=BLUE)
        y_cell.alignment = Alignment(horizontal="center")
        ws.row_dimensions[display_row].height = 6
        for col, (true_value, pred_value) in enumerate(zip(truth[array_row], pred[array_row]), start=2):
            coordinate = f"{get_column_letter(col)}{display_row}"
            if formula_kind == "absolute":
                formula = f"=ABS('U True'!{coordinate}-'U Pred'!{coordinate})"
                numeric_value = abs(float(true_value) - float(pred_value))
            elif formula_kind == "squared":
                formula = f"=('U True'!{coordinate}-'U Pred'!{coordinate})^2"
                numeric_value = (float(true_value) - float(pred_value)) ** 2
            else:
                raise ValueError(formula_kind)
            cell = ws.cell(display_row, col, formula)
            cell.fill = choose_fill(numeric_value)
            cell.number_format = "0.000000"
            cell.font = Font(size=1, color=interpolate_color(numeric_value, scale_min, scale_max))
    data_range = f"B5:{get_column_letter(last_col)}{last_row}"
    ws.conditional_formatting.add(
        data_range,
        ColorScaleRule(start_type="min", start_color="440154", mid_type="percentile", mid_value=50,
                       mid_color="21918C", end_type="max", end_color="FDE725"),
    )
    ws.freeze_panes = "B5"
    ws.sheet_view.zoomScale = 20
    ws.sheet_view.showGridLines = False
    return ws


def add_triptych(wb, truth, pred, abs_error, scale_min, scale_max, error_max):
    ws = wb.create_sheet("Triptych", 0)
    sample_rows = np.linspace(0, truth.shape[0] - 1, 66, dtype=int)
    sample_cols = np.linspace(0, truth.shape[1] - 1, 150, dtype=int)
    panel_width = len(sample_cols)
    gap = 5
    starts = [2, 2 + panel_width + gap, 2 + 2 * (panel_width + gap)]
    titles = ["u true", "u pred", "|error|"]
    matrices = [truth, pred, abs_error]
    ranges = [(scale_min, scale_max), (scale_min, scale_max), (0.0, error_max)]
    last_col = starts[-1] + panel_width - 1
    style_title(
        ws,
        "U-field error triptych",
        "Cell-based, downsampled display of frame 40. Full-resolution numeric fields and formulas are on the detail sheets.",
        last_col,
    )
    for start_col, title, matrix, (lower, upper) in zip(starts, titles, matrices, ranges):
        end_col = start_col + panel_width - 1
        ws.merge_cells(start_row=4, start_column=start_col, end_row=4, end_column=end_col)
        title_cell = ws.cell(4, start_col, title)
        title_cell.font = Font(name="Aptos Display", size=14, bold=True, color=WHITE)
        title_cell.fill = PatternFill("solid", fgColor=BLUE)
        title_cell.alignment = Alignment(horizontal="center")
        choose_fill = fill_cache(lower, upper)
        for display_index, array_row in enumerate(sample_rows[::-1], start=5):
            ws.row_dimensions[display_index].height = 5
            for offset, array_col in enumerate(sample_cols):
                value = float(matrix[array_row, array_col])
                cell = ws.cell(display_index, start_col + offset, value)
                cell.fill = choose_fill(value)
                cell.number_format = "0.000"
                cell.font = Font(size=1, color=interpolate_color(value, lower, upper))
        data_range = f"{get_column_letter(start_col)}5:{get_column_letter(end_col)}{4 + len(sample_rows)}"
        ws.conditional_formatting.add(
            data_range,
            ColorScaleRule(start_type="min", start_color="440154", mid_type="percentile", mid_value=50,
                           mid_color="21918C", end_type="max", end_color="FDE725"),
        )
        for col in range(start_col, end_col + 1):
            ws.column_dimensions[get_column_letter(col)].width = 1.25
    ws["A5"] = "y=197"
    ws["A5"].font = Font(size=8, color=MUTED)
    ws.cell(4 + len(sample_rows), 1, "y=0").font = Font(size=8, color=MUTED)
    for start_col in starts:
        ws.cell(5 + len(sample_rows), start_col, "x=0").font = Font(size=8, color=MUTED)
        ws.cell(5 + len(sample_rows), start_col + panel_width - 1, "x=448").font = Font(size=8, color=MUTED)
    ws.sheet_view.zoomScale = 20
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "B5"


def add_audit_sheet(wb, frame_index, time_value, truth, pred, x, y):
    ws = wb.create_sheet("MSE Audit", 1)
    height, width = truth.shape
    last_col_letter = get_column_letter(width + 1)
    last_row = height + 4
    squared_range = f"'Squared Error'!B5:{last_col_letter}{last_row}"
    style_title(
        ws,
        "MSE audit trail",
        "Every squared error is an Excel formula. The summary below exposes the exact hand calculation: sum the squared errors and divide by the number of points.",
        8,
    )
    labels = [
        (4, "Source prediction file", "outputs/field_predictions.npz"),
        (5, "Source truth file", "u_stack.npz"),
        (6, "Frame index", int(frame_index)),
        (7, "Time", float(time_value)),
        (8, "Grid size", f"{height} × {width}"),
    ]
    for row, label, value in labels:
        ws.cell(row, 1, label).font = Font(bold=True, color=TEXT)
        ws.cell(row, 2, value)

    ws["A10"] = "Manual MSE calculation"
    ws["A10"].font = Font(size=13, bold=True, color=WHITE)
    ws["A10"].fill = PatternFill("solid", fgColor=BLUE)
    ws.merge_cells("A10:D10")
    audit_rows = [
        (11, "Number of grid points (N)", f"=COUNT({squared_range})", height * width),
        (12, "Sum of squared errors (SSE)", f"=SUM({squared_range})", float(np.sum((truth - pred) ** 2))),
        (13, "MSE = SSE / N", "=B13/B12", float(np.mean((truth - pred) ** 2))),
        (14, "Direct Excel check", f"=AVERAGE({squared_range})", float(np.mean((truth - pred) ** 2))),
        (15, "RMSE", "=SQRT(B13)", float(np.sqrt(np.mean((truth - pred) ** 2)))),
        (16, "Maximum absolute error", "=MAX('Absolute Error'!B5:" + last_col_letter + str(last_row) + ")", float(np.max(np.abs(truth - pred)))),
    ]
    ws["A11"], ws["B11"], ws["C11"] = "Quantity", "Excel formula / result", "Verified value"
    for cell in ws[11]:
        if cell.column <= 3:
            cell.font = Font(bold=True, color=WHITE)
            cell.fill = PatternFill("solid", fgColor=BLUE)
    for row, label, formula, verified in audit_rows:
        destination_row = row + 1
        ws.cell(destination_row, 1, label)
        ws.cell(destination_row, 2, formula)
        ws.cell(destination_row, 3, verified)
        ws.cell(destination_row, 2).fill = PatternFill("solid", fgColor=PALE_BLUE)
        ws.cell(destination_row, 3).fill = PatternFill("solid", fgColor="E2F0D9")
        ws.cell(destination_row, 2).number_format = "0.0000000000E+00"
        ws.cell(destination_row, 3).number_format = "0.0000000000E+00"
    # Correct the point count display to an integer.
    ws["B12"].number_format = "#,##0"
    ws["C12"].number_format = "#,##0"

    ws["A20"] = "How to reproduce it by hand"
    ws["A20"].font = Font(size=13, bold=True, color=WHITE)
    ws["A20"].fill = PatternFill("solid", fgColor=BLUE)
    ws.merge_cells("A20:D20")
    instructions = [
        "1. On 'U True' and 'U Pred', take values at the same x/y cell.",
        "2. Subtract predicted from true. 'Absolute Error' shows ABS(difference).",
        "3. Square the raw difference. Every cell on 'Squared Error' does this.",
        "4. Add all squared-error cells to obtain SSE.",
        "5. Divide SSE by N. The result equals the AVERAGE of 'Squared Error'.",
    ]
    for index, instruction in enumerate(instructions, start=21):
        ws.cell(index, 1, instruction)
        ws.merge_cells(start_row=index, start_column=1, end_row=index, end_column=6)
    ws["A28"] = "One-cell example"
    ws["A28"].font = Font(size=13, bold=True, color=WHITE)
    ws["A28"].fill = PatternFill("solid", fgColor=BLUE)
    ws.merge_cells("A28:F28")
    ws.append([])
    example_row = 29
    headers = ["x", "y", "True", "Predicted", "Difference", "Squared error"]
    for col, header in enumerate(headers, start=1):
        cell = ws.cell(example_row, col, header)
        cell.font = Font(bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=BLUE)
    ws.cell(30, 1, float(x[0]))
    ws.cell(30, 2, float(y[-1]))
    ws.cell(30, 3, "='U True'!B5")
    ws.cell(30, 4, "='U Pred'!B5")
    ws.cell(30, 5, "=C30-D30")
    ws.cell(30, 6, "=E30^2")
    for col in range(3, 7):
        ws.cell(30, col).number_format = "0.000000"
    for col, width_value in enumerate([28, 44, 22, 22, 22, 22], start=1):
        ws.column_dimensions[get_column_letter(col)].width = width_value
    for row in range(4, 31):
        ws.row_dimensions[row].height = 22
    ws.freeze_panes = "A4"
    ws.sheet_view.showGridLines = False


def main():
    with np.load(PREDICTIONS_PATH) as predictions:
        pred = np.asarray(predictions["U_pred"], dtype=np.float64)
        frame_index = int(predictions["frame_index"])
        time_value = float(predictions["time"])
        x = np.asarray(predictions["x"], dtype=np.float64)
        y = np.asarray(predictions["y"], dtype=np.float64)
    if SAVED_TRUTH_PATH.is_file():
        with np.load(SAVED_TRUTH_PATH) as saved_truth:
            truth = np.asarray(saved_truth["U_true"], dtype=np.float64)
    else:
        with np.load(TRUTH_PATH) as source:
            truth = np.asarray(source["U"][frame_index], dtype=np.float64)
    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: truth {truth.shape}, prediction {pred.shape}")
    abs_error = np.abs(truth - pred)
    squared_error = (truth - pred) ** 2
    common_min = float(min(np.min(truth), np.min(pred)))
    common_max = float(max(np.max(truth), np.max(pred)))
    error_max = float(np.max(abs_error))
    squared_max = float(np.max(squared_error))

    wb = Workbook()
    wb.remove(wb.active)
    wb.calculation.fullCalcOnLoad = True
    wb.calculation.forceFullCalc = True
    wb.calculation.calcMode = "auto"
    add_triptych(wb, truth, pred, abs_error, common_min, common_max, error_max)
    add_audit_sheet(wb, frame_index, time_value, truth, pred, x, y)
    add_full_field_sheet(
        wb, "U True", truth, x, y, common_min, common_max,
        f"Ground-truth U at frame {frame_index}. Rows run from y={y[-1]:g} at top to y={y[0]:g} at bottom.",
    )
    add_full_field_sheet(
        wb, "U Pred", pred, x, y, common_min, common_max,
        f"Predicted U at frame {frame_index}. Uses the same color scale as U True.",
    )
    add_formula_field_sheet(
        wb, "Absolute Error", truth, pred, x, y, "absolute", 0.0, error_max,
        "Each cell is =ABS('U True' cell - 'U Pred' cell).",
    )
    add_formula_field_sheet(
        wb, "Squared Error", truth, pred, x, y, "squared", 0.0, squared_max,
        "Each cell is =('U True' cell - 'U Pred' cell)^2. Average all cells to obtain MSE.",
    )
    for ws in wb.worksheets:
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 1
        ws.sheet_properties.outlinePr.summaryBelow = True
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT_PATH)

    # Compact programmatic verification.
    check = load_workbook(OUTPUT_PATH, read_only=False, data_only=False)
    expected_sheets = ["Triptych", "MSE Audit", "U True", "U Pred", "Absolute Error", "Squared Error"]
    assert check.sheetnames == expected_sheets, check.sheetnames
    assert check["Absolute Error"]["B5"].value == "=ABS('U True'!B5-'U Pred'!B5)"
    assert check["Squared Error"]["B5"].value == "=('U True'!B5-'U Pred'!B5)^2"
    assert check["MSE Audit"]["B14"].value == "=B13/B12"
    formula_errors = []
    for sheet_name in ("Absolute Error", "Squared Error", "MSE Audit"):
        for row in check[sheet_name].iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and any(token in cell.value for token in ("#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A")):
                    formula_errors.append((sheet_name, cell.coordinate, cell.value))
    assert not formula_errors, formula_errors[:10]
    print(f"OUTPUT={OUTPUT_PATH}")
    print(f"FRAME={frame_index}; SHAPE={truth.shape}; N={truth.size}")
    print(f"MSE={float(np.mean(squared_error)):.12e}")
    print(f"RMSE={float(np.sqrt(np.mean(squared_error))):.12e}")
    print(f"MAX_ABS_ERROR={error_max:.12e}")
    print(f"SIZE_BYTES={OUTPUT_PATH.stat().st_size}")


if __name__ == "__main__":
    main()
