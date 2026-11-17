'''
common.py: small helpers shared by the step views
'''
import pandas as pd
import panel as pn
from bokeh.models import HoverTool
from bokeh.plotting import figure as bokeh_figure
from matplotlib.figure import Figure

COLORS = {"upper": "#1f77b4", "lower": "#2ca02c", "camber": "#d62728", "points": "#555555",
          "removed": "#d62728", "le": "#1f77b4", "te": "#d62728", "slice": "#999999", "fill": "#9ecae1"}


def table(rows, **kw):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    return pn.pane.DataFrame(df, index=False, float_format=lambda v: "{:.6g}".format(v), **kw)


def mpl_figure(width=7, height=4):
    fig = Figure(figsize=(width, height))
    return fig, fig.add_subplot(111)


def mpl(fig):
    fig.tight_layout()
    return pn.pane.Matplotlib(fig, dpi=110, tight=True, format="png", sizing_mode="scale_width", max_width=900)


def bfig(title, x_label, y_label, height=300, equal=True, hover=None):
    '''Bokeh figure (zoom with the mouse wheel, pan, reset), equal axes by default.'''
    f = bokeh_figure(title=title, height=height, sizing_mode="stretch_width", match_aspect=equal,
                     tools="pan,wheel_zoom,box_zoom,reset,save", active_scroll="wheel_zoom")
    f.xaxis.axis_label, f.yaxis.axis_label = x_label, y_label
    if hover:
        f.add_tools(HoverTool(tooltips=hover))
    f.toolbar.logo = None
    return f


def bokeh(f):
    '''Panel pane of a Bokeh figure; the legend is placed below the plot (never over the curves).'''
    if f.legend:
        legend = f.legend[0]
        f.center = [r for r in f.center if r is not legend]
        legend.orientation, legend.click_policy = "horizontal", "hide"
        legend.label_text_font_size, legend.spacing = "11px", 12
        f.add_layout(legend, "below")
        f.height = f.height + 40
    return pn.pane.Bokeh(f, sizing_mode="stretch_width")


def note(text, kind="light"):
    return pn.pane.Alert(text, alert_type=kind)
