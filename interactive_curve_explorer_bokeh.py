"""
Interactive explorer: select an ideal curve (dropdown) and compare
its values with the corresponding test points.

Outputs `bokeh_explorer.html` (open in any browser or via Jupyter).
"""

from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CustomJS, Select
from bokeh.plotting import figure
import pandas as pd

from data_utils import build_residual_table

# ---------- load ----------
test_df = pd.read_csv(
    "C:/Users/lanta/Desktop/python/assignment/dataset_test.csv")
ideal_df = pd.read_csv(
    "C:/Users/lanta/Desktop/python/assignment/dataset_ideal.csv")

table = build_residual_table(test_df, ideal_df)
first_curve = table["curve"].iloc[0]
source = ColumnDataSource(table[table["curve"] == first_curve])

# ---------- base figure ----------
plot = figure(
    title="Interactive Curve Explorer",
    x_axis_label="x",
    y_axis_label="y",
    width=800,
    height=400,
)
plot.circle("x", "y_test", source=source, size=5,
            color="blue", legend_label="test")
plot.line("x", "y_ideal", source=source, color="red", legend_label="ideal")

# ---------- dropdown widget ----------
dropdown = Select(
    title="Curve",
    value=first_curve,
    options=sorted(table["curve"].unique()),
)

dropdown.js_on_change(
    "value",
    CustomJS(
        args=dict(src=source, full_table=table.to_dict(orient="list")),
        code="""
        const curve = cb_obj.value;
        const data  = {x: [], y_test: [], y_ideal: []};
        for (let i = 0; i < full_table['curve'].length; i++) {
            if (full_table['curve'][i] === curve) {
                data['x'].push(full_table['x'][i]);
                data['y_test'].push(full_table['y_test'][i]);
                data['y_ideal'].push(full_table['y_ideal'][i]);
            }
        }
        src.data = data;
        src.change.emit();
    """,
    ),
)

# ---------- output ----------
output_file("bokeh_explorer.html", title="Interactive Curve Explorer")
show(column(dropdown, plot))
