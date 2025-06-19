# codeinhuman 🔴import pandas as pd
from data_utils import build_residual_table
🔴from bokeh.plotting import figure, show
🔴from bokeh.models import ColumnDataSource, Select, CustomJS
🔴from bokeh.layouts import column
🔴from bokeh.io import output_file

test_df = pd.read_csv("datasets/dataset_test.csv")
ideal_df = pd.read_csv("datasets/dataset_ideal.csv")

table = build_residual_table(test_df, ideal_df)
source = ColumnDataSource(table[table["curve"] == "y1"])

p = figure(
    title="Interactive Curve Explorer",
    x_axis_label="x",
    y_axis_label="y",
    width=800,
    height=400,
)
p.circle("x", "y_test", source=source, size=5,
         color="blue", legend_label="test")
p.line("x", "y_ideal", source=source, color="red", legend_label="ideal")

dropdown = Select(title="Curve", value="y1",
                  options=sorted(table["curve"].unique()))

dropdown.js_on_change(
    "value",
    CustomJS(
        args=dict(src=source, table=table.to_dict(orient="list")),
        code="""
        const curve = cb_obj.value;
        const data  = {x: [], y_test: [], y_ideal: []};
        for (let i = 0; i < table['curve'].length; i++) {
            if (table['curve'][i] === curve) {
                data['x'].push(table['x'][i]);
                data['y_test'].push(table['y_test'][i]);
                data['y_ideal'].push(table['y_ideal'][i]);
            }
        }
        src.data = data;
        src.change.emit();
    """,
    ),
)

layout = column(dropdown, p)
output_file("bokeh_explorer.html", title="Interactive Curve Explorer")
show(layout)
